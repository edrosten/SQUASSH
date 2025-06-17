from typing import Tuple, List, cast
from pathlib import Path

from pystrict import strict
import torch
import torch._dynamo
from torch import Tensor

import data_spectrin
import train
import network
from localisation_data import LocalisationDataSetMultipleDan6, RenderDan6
import device
import save_ply
from matrix import trn, scale_along_axis_and_expand_matrix, so3_6D
from circular_interpolation import circular_interpolation_fourier

_SMOOTHMAX_SCALE=10.0

def contingumax(repetition_logits: torch.Tensor)->torch.Tensor:
    '''
    A bit like softmax, but everything to the left of the argmax is also 1.
    '''
    assert repetition_logits.ndim==2
    batch_size = repetition_logits.shape[0]
    N = repetition_logits.shape[1]


    # Could use sigmoid here. Any 0-1 mapping would do?
    # softmax prevents too much weight which can push the logsumexp smooth max
    # to values a fair bit over 1
    # Except that log-sum-exp is a bit pants (see below)
    # r = torch.nn.functional.softmax(repetition_logits, dim=1)
    r = repetition_logits.sigmoid()
    #r = repetition_logits.softmax(dim=1)

    # Now make it so the repetitions must be contiguous so we get some number of
    # repetitions with not gaps
    #
    # R0 = max(r0, r1, r2, r3, r4)
    # R1 = max( 0, r1, r2, r3, r4)
    # R2 = max( 0,  0, r2, r3, r4)
    # R3 = max( 0,  0,  0, r3, r4)
    # R4 = max( 0,  0,  0,  0, r4)
    #
    # The computation is quadratic in N. Similar results could be achived with
    # recursive application of max, i.e.:
    #   R4 = r4
    #   R3 = max(r3, R4)
    #   R2 = max(r2, R3)
    #   etc
    # But the constant is small, and the quadratic method is both loop free and
    # much more parallelisable
    # 
    # Note that this is overparaemterised, since there's sort of an implicit
    # 1 for R0. It's possible to suppress it a little: if all the r's are equal
    # then after logsumexp, the result will be uniform and < 1, though that will
    # get bumped up effectively by the later image normalisation
    #
    # log-sum-exp doesn't give great results: the output for empty cells is always 
    # log(N), since it's a log of sum of N exp(0)'s, which is quite high

    # The logsumexp version:
    #return (_SMOOTHMAX_SCALE*r.unsqueeze(1).expand(batch_size, N, N).triu()).logsumexp(dim=2)/_SMOOTHMAX_SCALE

    tr = r.unsqueeze(1).expand(batch_size, N, N).triu()
    return (tr * torch.nn.functional.softmax(tr*_SMOOTHMAX_SCALE, dim=2)).sum(2)

_RADIAL_COMPONENTS=4
_RADIUS_SCALE=1.5


def compute_radii_multipliers_from_parameters(parameters: torch.Tensor, theta: torch.Tensor)->torch.Tensor:
    '''
    Compute radial scaling given an un-sliced parameter list
    '''
    radii_parameters = torch.tanh(parameters[:, 2:2+_RADIAL_COMPONENTS])
    return  cast(torch.Tensor, _RADIUS_SCALE ** circular_interpolation_fourier(radii_parameters, theta))



@strict
class AxialRepeatRadialExpand(network.ModelParameterisation):
    '''Parameterise as a stretch along an axis and expansion normal to the axis.
    The principle axis is optimized as part of the model'
    '''
    def __init__(self, min_repetitions: int, max_repetitions: int)->None:
        super().__init__()
        #Principal axis is the axis of stretch and shrink, which is global
        #Stored as a 3 vector representing a direction
        self._principal_axis = torch.nn.parameter.Parameter(torch.rand(3))
        self._secondary_axis = torch.nn.parameter.Parameter(torch.rand(3))

        
        #self._principal_axis.requires_grad=False
        #self._principal_axis[0]=1
        #self._principal_axis[1]=0
        #self._principal_axis[2]=0

        self.register_buffer("max_stretch_factor_expand", torch.tensor(1.0))
        self.register_buffer("min_repetition_length", torch.tensor(1.0))
        self.register_buffer("max_repetition_length", torch.tensor(2.0))
        self._min_repetitions = min_repetitions
        self._repetitions = max_repetitions

        self.max_stretch_factor_expand: torch.Tensor
        self.min_repetition_length: torch.Tensor
        self.max_repetition_length: torch.Tensor

    def number_of_parameters(self)->int:
        return 2 + self._repetitions - self._min_repetitions + _RADIAL_COMPONENTS



    def _compute_length(self, length_parameter: torch.Tensor)->torch.Tensor:
        ''' 
        (batched) maps length parameter (-inf, inf) to min/max length
        '''
        # Map -inf...inf to 0...1 to min...max but logarithmically
        #
        # l is the logit
        # s = sigmoid(l)
        # 
        # log_length = lerp(s, log(lo), log(hi)) = s * (log(hi)-log(lo)) + log(lo)
        # length = exp(log_length)
        #        = exp(s * log(hi/lo)) * lo
        #        =  (hi/lo)^s * lo
        assert length_parameter.ndim == 1
        return (self.max_repetition_length/self.min_repetition_length).pow(torch.sigmoid(length_parameter)) * self.min_repetition_length

    def compute_elongation_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        '''Given parameter list, compute the elongation'''
        return self._compute_length(parameters[:,0])
    
    def compute_expansion_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        '''Given parameter list, compute the elongation'''
        return cast(torch.Tensor, self.max_stretch_factor_expand**torch.tanh(parameters[:,1]))

    def get_R(self)->torch.Tensor:
        '''Get the rotation matrix'''
        return so3_6D(torch.cat([self._principal_axis, self._secondary_axis]).unsqueeze(0)).squeeze(0)

    def get_axis_aligned_and_stretched_points(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->torch.Tensor:
        '''Model conists of a set of points and a set of axes
        Get the points in the frame of those axes'''

        # First, centre the model points, weighted by intensity. This is so that the axis of
        # expansion goes through the middle, otherwise there's somewhat unpleasant coupling between
        # the axis and the shape change. Note that since later in the process (after this function)
        # translation and rotation is performed, having it fully centred reduces the coupling between
        # those as well. 
        #
        # Note that since distortion and length variation happens later, it will need to be recentred
        # a second time.

        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]
        axis = self.get_axis()
        expansion = self.compute_expansion_from_parameters(parameters)

        model_points_weighted_mean = (model_points * model_intensities.unsqueeze(1).expand(Nv, 3)).sum(0) / model_intensities.sum(0)
        centred_model_points = model_points - model_points_weighted_mean.unsqueeze(0).expand(Nv, 3)

        R = self.get_R()
        S = scale_along_axis_and_expand_matrix(axis, torch.ones_like(expansion), expansion)
        
        points = trn(S @ trn(centred_model_points).unsqueeze(0).expand(batch_size, 3, Nv))
        return R.unsqueeze(0).expand(batch_size, 3, 3) @ trn(points) #batch_size, 3, Nv

    

    def get_axis_aligned_and_reshaped_points(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->torch.Tensor:
        '''Get the points with the stretching and anisotropiuc radial scaling applied'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]

        zxy = self.get_axis_aligned_and_stretched_points(model_points, model_intensities, parameters)

        z = zxy[:,0,:]
        x = zxy[:,1,:]
        y = zxy[:,2,:]
        theta = torch.atan2(y, x)
        radius = (x**2 + y**2).sqrt()

        assert theta.shape == torch.Size((batch_size, Nv))
        assert radius.shape == torch.Size((batch_size, Nv))

        new_radii = radius * compute_radii_multipliers_from_parameters(parameters, theta)

        # Now reconstruct
        cos_theta = x / (radius + 1e-6)
        sin_theta = y / (radius + 1e-6)

        x_new = cos_theta * new_radii
        y_new = sin_theta * new_radii
        
        return torch.stack([z, x_new, y_new], 1) #batch_size, 3, Nv


    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]
        dev = model_points.device
        dtype = model_points.dtype
        
        length = self.compute_elongation_from_parameters(parameters)
        axis = self.get_axis()
        
        zyx_new = self.get_axis_aligned_and_reshaped_points(model_points, model_intensities, parameters)
        points = trn(self.get_R().permute(1,0).unsqueeze(0).expand(batch_size,3,3) @ zyx_new)



        if self._repetitions == self._min_repetitions:
            repetition_weights = torch.ones(batch_size, self._min_repetitions, device=parameters.device, dtype=parameters.dtype)
        else:
            repetition_logits = parameters[:,2+_RADIAL_COMPONENTS:]
            repetition_weights = torch.cat([torch.ones(batch_size, self._min_repetitions, device=parameters.device, dtype=parameters.dtype), contingumax(repetition_logits)], 1)


        intensities = model_intensities.unsqueeze(0).expand(batch_size, Nv)

        # Points are now: batch, Nv, 3, expand to Nv*repeate
        #    intensities: batch, Nv     expand similarly.
        #
        # Points are shifted along the axis by simple integer multiples of length
        scaled_lengths = torch.arange(self._repetitions, device=dev, dtype=dtype).unsqueeze(0).expand(batch_size, -1) * length.unsqueeze(1).expand(-1, self._repetitions)
        shifts = scaled_lengths.repeat_interleave(Nv, dim=1).unsqueeze(2).expand(-1, -1, 3) * axis.reshape(1, 1, 3).expand(batch_size, Nv*self._repetitions, 3)

        r_points = points.repeat((1, self._repetitions, 1)) + shifts
        r_intensitites = repetition_weights.repeat_interleave(Nv, dim=1) * intensities.repeat((1,self._repetitions))

        # Now shift the points to zero mean (intensity based)
        centre = (r_points * r_intensitites.unsqueeze(2).expand(batch_size, Nv*self._repetitions, 3)).sum(1) / r_intensitites.sum(1).unsqueeze(1).expand(batch_size,3)
        shifted_repeated_points = r_points - centre.unsqueeze(1).expand(batch_size, Nv*self._repetitions, 3)

        return shifted_repeated_points, r_intensitites, self.compute_expansion_from_parameters(parameters)


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




def _adhoc_test()->None:
    
    import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel

    # Somewhat ad-hoc testing coe
    #
    #
    #

    with torch.no_grad():
        #Generate a line of points
        
        angles = torch.tensor([60.]) * torch.pi / 180
        batch_size = 7
        length = 30
        N = 70
        R=2
        linear_pos = torch.arange(0,N)/(N-1) * length
        cs0 = torch.tensor([angles.cos(), angles.sin(), torch.zeros_like(angles)])
        
        pts = cs0.unsqueeze(0).expand(N, 3) * linear_pos.reshape(N, 1).expand(N, 3)
        intensities = linear_pos/length

        plt.clf()
        plt.ion()
        
        ar = AxialRepeatRadialExpand(2,2+R)
        ar.max_stretch_factor_expand = torch.tensor(1.)
        ar.min_repetition_length = torch.tensor(10.)
        ar.max_repetition_length = torch.tensor(10.)
        A = torch.tensor(45.) * torch.pi / 180
        ar._principal_axis.copy_(torch.tensor([A.cos(), A.sin(), 0])) # pylint:disable=protected-access
        print(ar.state_dict())
        
        logits = torch.randn(batch_size, R)*5
        parameters = torch.cat([torch.zeros(batch_size, 2), torch.zeros(batch_size, _RADIAL_COMPONENTS), logits], 1)

        cm = contingumax(logits)
        new_pts, new_intensities, _ = ar(pts, intensities, parameters)

        C=3
        for i in range(batch_size):
            plt.subplot(batch_size, C, i*C+1)
            plt.plot(logits[i], 'r')
            plt.ylabel('logit', color='r')
            plt.gca().twinx()
            plt.plot(cm[i])
            plt.axis((0, R, 0, 1.1))
            plt.ylabel('out', color='b')

            plt.subplot(batch_size, C, i*C+2)
            plt.scatter(pts[:, 0], pts[:,1], c=intensities, cmap='Greys')
            plt.axis('equal')

            plt.subplot(batch_size, C, i*C+3)
            plt.scatter(new_pts[i, :, 0], new_pts[i,:,1], c=new_intensities[i], cmap='Greys')
            plt.axis('equal')


        plt.show()


def PredictReconstructionRepetitionD6(model_size: int, nm_per_pixel_xy: float, image_size_xy:int, image_size_z: int, z_scale: float, data: List[Tensor])->Tuple[network.GeneralPredictReconstruction, AxialRepeatRadialExpand]:
    '''Predict R/t etc and rerender for a 6 plane rendering, also allow prediction of "opting out"'''
    d6render = RenderDan6(data)

    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->List[torch.Tensor]:
        return [i.unsqueeze(1) for i in d6render(
               centres=centres, 
               weights=weights,
               sigma_xy_nm=sigma_nm,
               nm_per_pixel_xy=nm_per_pixel_xy,
               z_scale=z_scale,
               xy_size=image_size_xy,
               z_size=image_size_z) ]

    parameterisation = AxialRepeatRadialExpand(5, 6)
    
    reconstructor = network.GeneralPredictReconstruction(
        model_size=model_size, 
        volume_cube_size_nm=image_size_xy*nm_per_pixel_xy, 
        renderfunc=renderer, 
        parameterisation=parameterisation,
        network_factory=network.NetworkAny)


    return reconstructor, parameterisation
 

def _main()->None:
    data3d = [t.to(device.device).half() for t in data_spectrin.load_3d()]

    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 25,
        z_scale = 1
    )


    dataset_initial = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=data3d, augmentations=15, device=device.device)
    dataset_initial.set_batch_size(5)


    torch._dynamo.config.cache_size_limit=512 # pylint: disable=protected-access # Do we still need this?
    for i in range(1):

        net, parameterisation = PredictReconstructionRepetitionD6(model_size=300, **vars(data_parameters), data=data3d)

        parameterisation.min_repetition_length = torch.tensor(160.)
        parameterisation.max_repetition_length = torch.tensor(240.)
        parameterisation.max_stretch_factor_expand = torch.tensor(2.0)

        net.to(device.device)
        
        params = train.TrainingParameters()
        params.batch_size = 40
        params.validity_weight=0.8

        params.schedule[0].epochs = 2000
        params.schedule[0].initial_psf = 250
        params.schedule[0].final_psf = 45
        params.schedule[0].psf_step_every= 200
        params.schedule[0].initial_lr= 0.0001
        params.schedule[0].final_lr= 0.0001
        

        fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
        train.retrain(fast, dataset_initial, params, f'run-{i:03}-phase_0')

if __name__ == "__main__":
    #_adhoc_test()
    _main()

