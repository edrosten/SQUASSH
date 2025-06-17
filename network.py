from typing import Tuple, List, Union, Protocol, cast, Callable
from pathlib import Path
from abc import abstractmethod
import math

from torch import nn
import torch
from torch import Tensor
import numpy
from pystrict import strict


from matrix import so3_6D, trn, scale_along_axis_and_expand_matrix
import matrix
import localisation_data
import save_ply
import render

def CBABlock(in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int)->nn.Module:
    '''Conv-batchnorm-Activation block'''
    return nn.Sequential(
         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
         nn.BatchNorm2d(out_channels),
         nn.SiLU()
         )

def EncoderDownsampleBlock(in_channels: int, out_channels: int)->nn.Module:
    '''
    Basic unit for the encoder. Does some processing then
    Reduces size by a factor of 2
    '''
    return CBABlock(in_channels, out_channels, 3, 2, 1)

OUTPUT_CHANNELS_RTS_VALID=11
def split_output_rts_valid(r: Tensor, max_translation: float, min_sigma: torch.Tensor)->Tuple[Tensor, Tensor, Tensor, Tensor]:
    '''
    Splits an input tensor of logits into translation, rotation, sigma, and validity
    '''
    assert r.ndim == 2
    assert r.shape[1] == OUTPUT_CHANNELS_RTS_VALID

    translation: Tensor = torch.tanh(r[:,0:3]) * max_translation
    sigma: Tensor = (nn.functional.softplus(r[:,3]) + 1) * min_sigma # pylint: disable=not-callable
    rotation_params: Tensor = r[:,4:10]
    is_valid = torch.sigmoid(r[:,10])

    return translation, so3_6D(rotation_params), sigma, is_valid



LATENT_SPACE_WIDTH=256
def LatentToOutput(output_channels: int)->nn.Module:
    """Network "head": convert latent space to output space (logits)"""
    return nn.Sequential(
        nn.Linear(LATENT_SPACE_WIDTH, 256), # hax 64x64 input
        nn.BatchNorm1d(256),
        nn.SiLU(),
        nn.Linear(256, output_channels)
    )
        
ENCODER_WIDTH=256
def Encoder()->nn.Module:
    '''
    Encoder takes in single channel image, multiple of 32 in size.
    Returns image 1/32 size, ENCODER_WIDTH channels
    '''
                                                        # Input size
    return nn.Sequential(                               # 128 64 32
            EncoderDownsampleBlock(1, 16),              # 64  32 16
            EncoderDownsampleBlock(16, 32),             # 32  16  8
            EncoderDownsampleBlock(32, 64),             # 16   8  4
            EncoderDownsampleBlock(64, 128),            # 8    4  2
            EncoderDownsampleBlock(128, ENCODER_WIDTH)  # 4    2  1
        )



class NetworkAny(nn.Module):
    '''Takes in projections, returns latent prediction''' 
    def __init__(self, sample_data: List[Tensor], output_channels: int):
        ''' 
        @param: sample_data A sample of the input data. Used by the network to compute the 
                            size of the output of the shared encoders.

        Very basic encode-process-output structure
        '''
        super().__init__()
        self._shared_encoder = Encoder()
        
        self._shared_encoder.eval()
        encoder_elements = self._encode_and_flatten(sample_data).shape[1]
        self._shared_encoder.train()

        self._process = nn.Sequential(
            nn.Linear(encoder_elements, LATENT_SPACE_WIDTH),
            nn.BatchNorm1d(LATENT_SPACE_WIDTH)
            # Yes lack of activation here is a bug
        )

        self._output = LatentToOutput(output_channels)

        self._input_sizes = [ i.shape for i in sample_data]

    def _encode_and_flatten(self, x:List[Tensor])->Tensor:
        return torch.cat([self._shared_encoder(i).flatten(start_dim=1) for i in x], 1)

    def forward(self, x: List[Tensor])->torch.Tensor:
        '''Standard forward method'''
        
        #Check the input sizes match the setup excluding the batch dimension 
        #but check the batch dimension for consistency
        for i,s in zip(x, self._input_sizes):
            assert i.shape[1:] == s[1:]
            assert i.shape[0] == x[0].shape[0]

        enc = self._encode_and_flatten(x)
        proc = self._process(enc)
        output_channels = cast(torch.Tensor, self._output(proc))

        return output_channels



class NetworkAnyWithoutBug(nn.Module):
    '''Takes in projections, returns latent prediction''' 
    def __init__(self, sample_data: List[Tensor], output_channels: int):
        ''' 
        @param: sample_data A sample of the input data. Used by the network to compute the 
                            size of the output of the shared encoders.

        Very basic encode-process-output structure
        '''
        super().__init__()
        self._shared_encoder = Encoder()
        
        self._shared_encoder.eval()
        encoder_elements = self._encode_and_flatten(sample_data).shape[1]
        self._shared_encoder.train()

        self._process = nn.Sequential(
            nn.Linear(encoder_elements, LATENT_SPACE_WIDTH),
            nn.BatchNorm1d(LATENT_SPACE_WIDTH),
            nn.SiLU(),
        )

        self._output = LatentToOutput(output_channels)

        self._input_sizes = [ i.shape for i in sample_data]

    def _encode_and_flatten(self, x:List[Tensor])->Tensor:
        return torch.cat([self._shared_encoder(i).flatten(start_dim=1) for i in x], 1)

    def forward(self, x: List[Tensor])->torch.Tensor:
        '''Standard forward method'''
        
        #Check the input sizes match the setup excluding the batch dimension 
        #but check the batch dimension for consistency
        for i,s in zip(x, self._input_sizes):
            assert i.shape[1:] == s[1:]
            assert i.shape[0] == x[0].shape[0]

        enc = self._encode_and_flatten(x)
        proc = self._process(enc)
        output_channels = cast(torch.Tensor, self._output(proc))

        return output_channels


def _as_module(func: Callable[[torch.Tensor],torch.Tensor])->nn.Module:
    class _ModFunc(nn.Module):
        def forward(self, arg: torch.Tensor)->torch.Tensor:
            '''forward function'''
            return func(arg)
    return _ModFunc()
    

def EncoderDownsampleBlock1D(in_channels: int, out_channels: int)->nn.Module:
    '''
    Basic unit for the encoder. Does some processing then
    Reduces size by a factor of 2
    '''
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, 2, 1),
        nn.BatchNorm1d(out_channels),
        nn.SiLU()
    )


MAX_TRANSLATION_FRACTION=0.1

class RenderFunc(Protocol):
    '''Protocol for packaged rendering function'''
    def __call__(self, centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->List[torch.Tensor]:...


def _inverse_sigmoid(x: torch.Tensor)->torch.Tensor:
    return -torch.log((1 / (x + 1e-16)) - 1)

# TODO this is a late addition, use it where this has been hand coded every time
def center_points(points: torch.Tensor, intensities: torch.Tensor)->torch.Tensor:
    '''Center the intensity weighted input points, i.e. remove the mean'''
    assert points.shape[0:2] == intensities.shape
    assert points.ndim == 3
    assert points.shape[2] == 3

    batch_size = points.shape[0]

    centre = (points * intensities.unsqueeze(2).expand_as(points)).sum(1) / intensities.sum(1).unsqueeze(1).expand(batch_size,3)
    return points - centre.unsqueeze(1).expand_as(points)


# pylint: disable=missing-function-docstring
class ModelParameterisation(nn.Module):
    '''Fixed protocol for model parameterisations to adhere to.
    Note that the _apply_parameterisation method should be overridden rather than 
    the forward method, since the forward method does all the consistency checks
    on the shape of the input tensors. You can safely assume that _apply_parameterisation
    has consistently shaped inputs.
    '''
    @abstractmethod 
    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path)->None:...

    @abstractmethod
    def number_of_parameters(self)->int:
        ...
    
    # TODO: should this have just been forward and some hooks?
    @abstractmethod
    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Takes the 3D model point cloud and the parameterisation logits and 
        applies the parameterisation to the model

        Returns a batch of models modified by the parametera and also parameters aggregated for loss computation

        This is only ever called via forward, so the dimensions of the inputs are already checked.
        '''


    def forward(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert model_points.ndim == 2
        assert model_points.shape[1] == 3
        assert parameters.ndim == 2
        assert parameters.shape[1] == self.number_of_parameters()
        assert model_intensities.ndim == 1
        assert model_intensities.shape[0] == model_points.shape[0]
        return self._apply_parameterisation(model_points, model_intensities, parameters)

    
    def __call__(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points, intensities, aggregate = cast(Tuple[Tensor, Tensor, Tensor], super().__call__(model_points, model_intensities, parameters))

        # Sanity check the results
        assert points.shape[0] == intensities.shape[0]  #Batch size
        assert points.shape[0] == aggregate.shape[0]
        assert points.shape[0] == parameters.shape[0]
        assert points.ndim == 3
        assert intensities.ndim == 2
        assert points.shape[1] == intensities.shape[1]
        assert points.shape[2] == 3
        return points, intensities, aggregate

@strict
class IdentityParameterisation(ModelParameterisation):
    '''Parameterise as a stretch along an axis and expansion normal to the axis.
    The principle axis is optimized as part of the model'
    '''
    def __init__(self)->None:
        super().__init__()

    def number_of_parameters(self)->int:
        return 0

    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]
        points = model_points.unsqueeze(0).expand(batch_size, Nv, 3)
        intensities = model_intensities.unsqueeze(0).expand(batch_size, Nv)
        return points, intensities, torch.ones(1, dtype=model_points.dtype, device=model_points.device)

    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path)->None:
        '''Dump out a visualisation'''
        to_write: list[Tensor | tuple[Tensor, tuple[int, int, int]]] = [ model_points.cpu() ]
        save_ply.save(name, to_write)



@strict
class AxialStretchRadialExpand(ModelParameterisation):
    '''Parameterise as a stretch along an axis and expansion normal to the axis.
    The principle axis is optimized as part of the model'
    '''
    def __init__(self)->None:
        super().__init__()
        #Principal axis is the axis of stretch and shrink, which is global
        #Stored as a 3 vector representing a direction
        self._principal_axis = torch.nn.parameter.Parameter(torch.rand(3))

        # TODO make these buffers
        self.max_stretch_factor_axis = 1.0
        self.max_stretch_factor_expand = 1.0


    def number_of_parameters(self)->int:
        return 2

    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]


        scale = self.max_stretch_factor_axis**torch.tanh(parameters[:,0])
        scale2 = self.max_stretch_factor_expand**torch.tanh(parameters[:,1])

        S = scale_along_axis_and_expand_matrix(self._principal_axis, scale, scale2)

        points = trn(S @ trn(model_points).unsqueeze(0).expand(batch_size, 3, Nv))
        intensities = model_intensities.unsqueeze(0).expand(batch_size, Nv)

        # TODO aggregate scale here. This is just compatibility with the old one
        return points, intensities, scale


    def get_axis(self)->Tensor:
        '''Return principal axis as unit vector'''
        return self._principal_axis / torch.sqrt((self._principal_axis**2).sum())

    def get_axis_points(self, length:torch.Tensor)->Tensor:
        '''Get some points along the main axis for visualisation'''
        N=100
        axis = torch.arange(start=-N, end=N+1, device=length.device)/N * length
        axis = axis.unsqueeze(1).expand(axis.shape[0], 3)
        return axis * self.get_axis().unsqueeze(0).expand(axis.shape)


    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path)->None:
        '''Dump out a visualisation'''

        to_write: list[Tensor | tuple[Tensor, tuple[int, int, int]]] = [ model_points.cpu(), (self.get_axis_points((model_points**2).sum(1).max().sqrt()), (255,0,0)) ]
        save_ply.save(name, to_write)




def _gaussian_cdf(x: torch.Tensor, sigma: torch.Tensor)->torch.Tensor:
    assert x.shape == sigma.shape or sigma.numel() == 1
    return 0.5 * ( 1 + torch.erf(x / (sigma * math.sqrt(2))))


def _approximate_gaussian_cdf(x: torch.Tensor, sigma: torch.Tensor)->torch.Tensor:
    assert x.shape == sigma.shape or sigma.numel() == 1
    # The CDF is pretty clost to a sigmoid in shape, but the derivatives die 
    # away less aggressively (exp, versus exp^2). The approximation can be
    # reasoned about as the gaussian in terms of size/sigma/FWHM, but may
    # be easier to optimize
    #
    # A logistic sigmoid L(x)=1/(1+exp(-x)) has derivative L'(x)=L(x)L(1-x). 
    # Plugging in L(0) =0.5 gives L'(0) = 1/4
    #
    # A Gaussian CDF has the gradient in the middle as a Gaussian, so 1/sqrt(2*pi*sigma^2)
    #
    # Equating these two to get the equivalent sigma for a good match for a GCDF to L(X) gives
    #         __
    # 𝜎ₑ = 4/√2π ≈ 1.596
    #
    # There are other approximations. Empirically:
    #
    #  𝜎ₑ ≈ 1.596  Same gradient (0.25) at x=0
    #  𝜎ₑ ≈ 1.696  minimize sum of absolute error
    #  𝜎ₑ ≈ 1.697  minimize ssum of squared error
    #  𝜎ₑ ≈ 1.701  minimise max absolute error (sum of inf'd error) < 0.01 max error
    #
    sigma_equivalent = 1.701
    return torch.nn.functional.sigmoid(x*sigma_equivalent / sigma)

    
def test_approximate_gaussian_cdf()->None:
    x = torch.arange(-36, 36, 0.001) # 18, sig=1 enough to fully saturate both functions
    g = _gaussian_cdf(x, torch.tensor(2.0))
    a = _approximate_gaussian_cdf(x, torch.tensor(2.0))

    assert (g-a).abs().max() < 0.01


@strict
class AxialGrowth(ModelParameterisation):
    '''Parameterise a cut along the principal axis.
    The principle axis is optimized as part of the model'
    '''
    def __init__(self, length_scale: float, cut_sigma: float):
        '''
        Initialise.
        length_scale: around the same as volume_cube_size. Small numbers, i.e. +/-1 ish will map to this scale
        cut_sigma: cut is made with a Gaussian CDF shape, using this sigma.
        '''
        super().__init__()
        #Principal axis is the axis of stretch and shrink, which is global
        #Stored as a 3 vector representing a direction
        self._principal_axis = torch.nn.parameter.Parameter(torch.rand(3))
        self.register_buffer('_cut_scale_factor', torch.tensor(length_scale))
        self.register_buffer('_cut_sigma', torch.tensor(cut_sigma))

        # generated via register_buffer, can't infer type. Why is it OK with the scale factor?
        self._cut_sigma: torch.Tensor


    def number_of_parameters(self)->int:
        return 1

    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]

        #Prevent cutting too far off the left hand side of the volume cube
        #cuts will have an interaction with the normalisation
        # If it's too far, the intensity can come out as 0 and nan 0 during normalisation
        # The normalization is fixed with n / (n.sum() + 1e-5), but if that triggers then 
        # derivatives are lost for that datum. Currently unclear if that matters.
        cut_point_nm = (torch.nn.functional.softplus(parameters[:,0]+1)-1) * self._cut_scale_factor # pylint: disable=not-callable
        #cut_point_nm = parameters[:,0] * self._cut_scale_factor

        axis = torch.nn.functional.normalize(self._principal_axis, dim=0)

        points = model_points.unsqueeze(0).expand(batch_size, Nv, 3) # batch_size x Nv x 3

        projections = matrix.inner_dot(points, axis.reshape(1,1,3).expand_as(points)) # batch, Nv

        cut_point_nm = cut_point_nm.reshape(batch_size, 1).expand_as(projections)
        # 1 - so it goes from on to off with positive direction
        cut_intensities = 1-_approximate_gaussian_cdf(projections-cut_point_nm, sigma=self._cut_sigma)

        intensities = model_intensities.unsqueeze(0).expand(batch_size, Nv)* cut_intensities

        # Shift to centre. This is because otherwise the subsequent rotation and shift
        # will have a substantial lever-arm effect which makes predicting the correct
        # translation somewhat harder.
        
        # could just do it in all dimensions?
        average_weighted_position = (cut_intensities * projections).sum(1) / (cut_intensities.sum(1)+0.001)

        shift = axis.reshape(1,1,3).expand_as(points) * average_weighted_position.reshape(batch_size, 1, 1).expand_as(points)
        #print("\n"*10)
        #print("param = ", parameters)
        #print("cp = ", cut_point_nm)
        #print("ave = ", average_weighted_position)

        # TODO aggregate scale here. This is just compatibility with the old one
        return points-shift, intensities, parameters[:,0]
        #return points-shift, model_intensities.unsqueeze(0).expand(batch_size, Nv), parameters[:,0]


    def get_axis(self)->Tensor:
        '''Return principal axis as unit vector'''
        return self._principal_axis / torch.sqrt((self._principal_axis**2).sum())

    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path)->None:
        '''Dump out a visualisation'''
        length=(model_points **2).sum(1).max().sqrt().item()
        N=100
        axis = torch.arange(start=-N, end=N+1)/N * length
        axis = axis.unsqueeze(1).expand(axis.shape[0], 3)
        axis1 = axis * self.get_axis().cpu().unsqueeze(0).expand(axis.shape)

        to_write: list[Tensor | tuple[Tensor, tuple[int, int, int]]] = [ model_points.cpu(), (axis1, (255,0,0)) ]
        save_ply.save(name, to_write)







class NetworkFactory(Protocol):
    '''Protocol to generate a network given sample data (for input size) and number of output channels'''
    def __call__(self, sample_data: List[torch.Tensor], output_channels: int)->nn.Module:
        ...




@strict
class GeneralPredictReconstruction(nn.Module):
    '''
    Given a set of images, predict a reconstruction,
    i.e. predict rotation, translation and sigma
    (per batch), then re-render the model using
    the rotation and translation

    The parameterisation and network generating function are supplied

    '''
    def __init__(self, model_size: int, volume_cube_size_nm:float, renderfunc: RenderFunc, parameterisation: ModelParameterisation, network_factory:NetworkFactory):
        super().__init__()
        # Note that the renderer takes positions in NM
        # with (0,0) being the centre.

        # Make a small batch of data to render. This is needed to initialise the network
        # so that it knows the image sizes
        tmp_sigma = torch.ones(1)
        tmp_pts = torch.zeros(1, model_size, 3)
        tmp_wts = torch.ones(1, model_size)*0.5
        tmp_data = renderfunc(centres=tmp_pts, weights=tmp_wts, sigma_nm=tmp_sigma)

        self.prediction_network = network_factory(sample_data=tmp_data, output_channels = OUTPUT_CHANNELS_RTS_VALID+parameterisation.number_of_parameters())
        self._max_translation=volume_cube_size_nm*MAX_TRANSLATION_FRACTION


        # We want points to be in the range +/-1 roughly to make them the same scale as convolution
        # kernel elements
        # Why is it 10 here?
        # If it were 1 then +/-.5 would go span the full size of the image
        # This means +/-0.05 spans the full range of the image
        # So it works, but why 10? seems arbitrary.
        self.model_scale = volume_cube_size_nm * 10

        #Scatter points to fill up the image, then scale.
        #Care is needed with the scale/learning rate over all
        # Surely we shouldn't be scaling here???
        self._model_points = torch.tensor([])
        self._model_intensities = torch.tensor([])

        # 0.25 here, so the points are basically scattered in the middle of the volume
        # rather than too close to the edge, so they are less likely to get ejected.
        # Not sure if that claim is true, but this strategy works well enough
        self.set_model(0.25*volume_cube_size_nm*(torch.rand(model_size,3)-.5), torch.ones(model_size)*0.5)

        self._render = renderfunc
        self._parameterisation = parameterisation
        
    def get_model(self)->Tuple[Tensor, Tensor]:
        '''Returns the 3D model as points, in nm'''
        return (self._model_points * self.model_scale, torch.nn.functional.sigmoid(100*self._model_intensities))

    def set_model(self, new_points:Tensor, new_intensities:Tensor)->None:
        '''Sets the model to the provided points, in  nm'''

        if self._model_points.shape != new_points.shape:
            self._model_points = torch.nn.parameter.Parameter(torch.zeros_like(new_points))
            
        self._model_points.requires_grad = False
        self._model_points.copy_(new_points / self.model_scale)
        self._model_points.requires_grad = True

        if self._model_intensities.shape != new_intensities.shape:
            self._model_intensities = torch.nn.parameter.Parameter(torch.zeros_like(new_intensities))
        

        self._model_intensities.requires_grad = False
        self._model_intensities.copy_(_inverse_sigmoid(new_intensities)/100)
        self._model_intensities.requires_grad = True
        
        assert self._model_points.ndim == 2
        assert self._model_intensities.ndim == 1
        assert self._model_points.shape[1] == 3
        assert self._model_intensities.shape[0] == self._model_points.shape[0]

    
    def process_input(self, x: Union[Tensor,List[Tensor]], min_sigma_nm: torch.Tensor)->Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''
        Processes input image(s) and outputs standard results (t, r, sigma, is_valid) and any remaining logits.
        '''
        x_list = [x] if isinstance(x, Tensor) else x

        logits = self.prediction_network(x_list)

        t, r, sigma, is_valid = split_output_rts_valid(logits[:, 0:OUTPUT_CHANNELS_RTS_VALID], 
            min_sigma=min_sigma_nm,
            max_translation=self._max_translation)
        
        parameters = logits[:, OUTPUT_CHANNELS_RTS_VALID:]

        batch_size = x_list[0].shape[0]
        assert sigma.shape == torch.Size((batch_size,))
        assert r.shape == torch.Size((batch_size,3, 3))
        assert t.shape == torch.Size((batch_size,3))
        assert parameters.shape == torch.Size((batch_size, self._parameterisation.number_of_parameters()))
        
        return t, r, sigma, is_valid, parameters
        

    
    def forward(self, x: Union[Tensor,List[Tensor]], min_sigma_nm: torch.Tensor)->Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        '''
        Standard forward method

        Note, sigma can be proveided in which case it will use that rather than
        the predicted sigma
        '''
        
        # TODO: min_sigma_nm should really be set as a property and stored in a buffer.
        # This would allow it to be de/serialised properly meaning a network would "just work"
        # without having to know sigma.

        t, r, sigma, is_valid, parameters = self.process_input(x, min_sigma_nm)
        batch_size = t.shape[0]

        # Apply the parameterisation
        points, intensities, parameter_aggregate_for_loss = self._parameterisation(*self.get_model(), parameters)
        
        # Note that the parameterisation can change the number of points.
        assert points.ndim == 3
        assert intensities.ndim == 2
        assert points.shape[0] == batch_size
        assert intensities.shape[0] == batch_size
        assert points.shape[1] == intensities.shape[1]
        assert points.shape[2] == 3

        Nv = points.shape[1]


        t_per_point = t.unsqueeze(1).expand(batch_size, Nv, 3)
        
        
        # Rotate and shift resulting aggregate
        points = trn(r @ trn(points)) + t_per_point

        #Render with sigma
        return self._render(centres=points, weights=intensities, sigma_nm=sigma), parameter_aggregate_for_loss, sigma, is_valid
    
    def __call__(self, x: Union[Tensor,List[Tensor]], min_sigma_nm: torch.Tensor)->Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        return cast(Tuple[List[Tensor], Tensor, Tensor, Tensor], super().__call__(x, min_sigma_nm))

    def save_ply_with_axes(self, name:Path)->None:
        '''Dump out a visualisation'''
        self._parameterisation.save_ply_with_axes(self.get_model()[0], name)

    def save_model_txt(self, path: Path)->None:
        '''Save positions, intensities and length'''
        points, intensities = self.get_model()
        points = points.detach().cpu()
        intensities = intensities.detach().cpu().unsqueeze(1)

        fullmodel = torch.cat((points, intensities), 1)
        numpy.savetxt(f"{path}_model.txt", fullmodel.numpy())



def ReconstructionWithStretch2(model_size: int, volume_cube_size_nm:float, renderfunc: RenderFunc)->Tuple[GeneralPredictReconstruction, AxialStretchRadialExpand]:

    parameterisation = AxialStretchRadialExpand()
    reconstructor=  GeneralPredictReconstruction(
        model_size, 
        volume_cube_size_nm, 
        renderfunc, 
        parameterisation,
        NetworkAny)

    return reconstructor, parameterisation
        
        

def PredictReconstructionStretch2D(model_size: int, nm_per_pixel: float, image_size:int)->GeneralPredictReconstruction:
    '''Predict R/t etc and rerender for a single XY image (i.e. 2D-3D image)'''
    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->List[torch.Tensor]:
        return [render.render_batch_weights(centres=centres[:,:,0:2], weights=weights, sigma_nm=sigma_nm, nm_per_pixel=nm_per_pixel, size=image_size).unsqueeze(1)]

    reconstructor, _ = ReconstructionWithStretch2(model_size=model_size, volume_cube_size_nm=image_size*nm_per_pixel, renderfunc=renderer)
    return reconstructor


def PredictReconstructionStretchExpandValidDan6(model_size: int, nm_per_pixel_xy: float, image_size_xy:int, image_size_z: int, z_scale: float, data: List[Tensor])->Tuple[GeneralPredictReconstruction, AxialStretchRadialExpand]:
    '''Predict R/t etc and rerender for a 6 plane rendering, also allow prediction of "opting out"'''
    d6render = localisation_data.RenderDan6(data)

    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->List[torch.Tensor]:
        return [i.unsqueeze(1) for i in d6render(
               centres=centres, 
               weights=weights,
               sigma_xy_nm=sigma_nm,
               nm_per_pixel_xy=nm_per_pixel_xy,
               z_scale=z_scale,
               xy_size=image_size_xy,
               z_size=image_size_z) ]
    
    return ReconstructionWithStretch2(model_size=model_size, volume_cube_size_nm=image_size_xy*nm_per_pixel_xy, renderfunc=renderer)
 




def PredictReconstructionStretchExpandValid(model_size: int, nm_per_pixel_xy: float, image_size_xy:int, image_size_z: int, z_scale: float)->Tuple[GeneralPredictReconstruction, AxialStretchRadialExpand]:
    '''Predict R/t etc and rerender for a 3 plane rendering, also allow prediction of "opting out"'''
    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->List[torch.Tensor]:
        return [i.unsqueeze(1) for i in 
            render.render_multiple_scale(
            centres=centres,
            sigma_xy_nm=sigma_nm,
            weights=weights,
            nm_per_pixel_xy=nm_per_pixel_xy,
            z_scale=z_scale,
            xy_size=image_size_xy,
            z_size=image_size_z)]

    return ReconstructionWithStretch2(model_size=model_size, volume_cube_size_nm=image_size_xy*nm_per_pixel_xy, renderfunc=renderer)
 
 
