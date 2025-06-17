from typing import Tuple, List, Optional, cast
from pathlib import Path

import torch
from torch import Tensor

from pystrict import strict

import data_legant
import volumetric
import train
import network
import device
import render
import save_ply
from matrix import so3_6D, trn, scale_along_axis_and_expand_matrix, euler

@strict 
class RotatedDuplication(network.ModelParameterisation):
    '''Model as a diplication with rotations along an axis'''
    def __init__(self)->None:
        super().__init__()
        #Principal axis is the axis of stretch and shrink, which is global
        #Stored as a 3 vector representing a direction
        self._principal_axis = torch.nn.parameter.Parameter(torch.rand(3))
        self._secondary_axis = torch.nn.parameter.Parameter(torch.rand(3))

        
        self.register_buffer("max_expand", torch.tensor(1.0))
        self.register_buffer("min_spacing", torch.tensor(0.5))
        self.register_buffer("max_spacing", torch.tensor(2.0))
        self.register_buffer("max_rotation", torch.tensor(45 * torch.pi/180))

        self.max_expand: torch.Tensor
        self.min_spacing: torch.Tensor
        self.max_spacing: torch.Tensor
        self.max_rotation: torch.Tensor

    def number_of_parameters(self)->int:
        return 4

    def compute_spacing_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        ''' 
        (batched) maps spacing parameter (-inf, inf) to min/max length
        '''
        # l is the logit
        # s = sigmoid(l)
        # length =  (hi/lo)^s * lo
        return (self.max_spacing/self.min_spacing).pow(torch.sigmoid(parameters[:,0])) * self.min_spacing
    
    def compute_expansion_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        '''Given parameter list, compute the elongation'''
        return self.max_expand.pow(torch.tanh(parameters[:,1]))

    def compute_rotation_angles_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        '''Given parameter list, compute the additional rotation'''
        return self.max_rotation * torch.tanh(parameters[:,2:4])

    def get_R(self)->torch.Tensor:
        '''Get the rotation matrix'''
        return so3_6D(torch.cat([self._principal_axis, self._secondary_axis]).unsqueeze(0)).squeeze(0)

    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]
        
        spacing = self.compute_spacing_from_parameters(parameters)
        axis = self.get_axis()

        expansion = self.compute_expansion_from_parameters(parameters)

        # Centre the points
        model_pts_weighted_mean = (model_intensities.unsqueeze(1).expand(Nv,3) * model_points).sum(0) / model_intensities.sum()
        centred_model_points = model_points - model_pts_weighted_mean.unsqueeze(0).expand(Nv, 3)

        exp = scale_along_axis_and_expand_matrix(axis, torch.ones_like(expansion), expansion)

        expanded_model_points = trn(exp @ trn(centred_model_points.unsqueeze(0).expand(batch_size, Nv, 3)))

        #A bit janky: use Euler angles for the rotation. It's only 45 degrees max or so, 
        #so that's a long way from the problems. 
        #
        # The rotation has the primary axis as the first row, therefore it will rotate the model
        # so that 1,0,0 is the primary axis 
        # We want to transform the Euler rotations into the frame of the model.
        # The rotations we want are around Y and Z 
        #
        # Note that the maximum total rotation, i.e. rotation angle about some axis isn't 
        # quite uniform and depends on the combinations of rotations.

        Rt = trn(self.get_R()).unsqueeze(0).expand(batch_size, 3, 3)
        angles = self.compute_rotation_angles_from_parameters(parameters)
        

        # This is jank a.f. in that this does introduce small amounts of 
        # rotation around axis, but not much for the requires range of angles.
        # Not enough to warrant implementing the Rodrigues formula and do it properly
        R = Rt  @ euler(angles[:,0], 'y') @ euler(angles[:,1], 'z') @ trn(Rt)

        
        second_set = trn(R @ trn(expanded_model_points)) + spacing.reshape(batch_size,1,1).expand(batch_size, Nv, 3) * axis.reshape(1,1,3).expand(batch_size, Nv, 3)
        
        full_pts = torch.cat([expanded_model_points, second_set], 1)
        full_intensities = model_intensities.repeat(2).unsqueeze(0).expand(batch_size, Nv*2)

        return full_pts, full_intensities, torch.ones_like(spacing)

    def get_axis(self)->torch.Tensor:
        '''Return principal axis as unit vector'''
        return self._principal_axis / torch.sqrt((self._principal_axis**2).sum())

    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path, parameters: Optional[torch.Tensor]=None)->None:
        '''Dump out a visualisation'''
        length=(model_points **2).sum(1).max().sqrt().item()
        N=100
        axis = torch.arange(start=-N, end=N+1)/N * length
        axis = axis.unsqueeze(1).expand(axis.shape[0], 3)
        axis1 = axis * self.get_axis().cpu().unsqueeze(0).expand(axis.shape)

        npts, _, _ = self(model_points, torch.ones_like(model_points)[:,0], parameters if parameters is not None else torch.zeros(1,4, device=model_points.device))

        to_write: list[Tensor | tuple[Tensor, tuple[int, int, int]]] = [ (npts[0,0:model_points.shape[0], :], (0,128,0)), (npts[0,model_points.shape[0]:, :], (0,128,255)), (axis1, (255,0,0)) ]
        save_ply.save(name, to_write)


def PredictReconstruction(model_size: int, nm_per_pixel_xy: float, image_size_xy:int, image_size_z: int, z_scale: float)->Tuple[network.GeneralPredictReconstruction, RotatedDuplication]:
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

    parameterisation = RotatedDuplication()
    reconstructor=  network.GeneralPredictReconstruction(
        model_size, 
        image_size_xy*nm_per_pixel_xy, 
        renderer, 
        parameterisation,
        network.NetworkAny)

    return reconstructor, parameterisation


def _test()->None:
    # Ad-hoc tests. Verify by hand that it does a duplicate-and-shift
    parameterisation = RotatedDuplication()
    
    parameterisation._principal_axis.requires_grad=False # pylint: disable=protected-access
    parameterisation._principal_axis.copy_(torch.tensor([1.,1,0])) # pylint: disable=protected-access
    parameterisation.max_expand = torch.tensor(2.)
    parameterisation.max_spacing = torch.tensor(5.)
    parameterisation.min_spacing = torch.tensor(.1)

    # Make a disc of points
    N = 10000
    pts = torch.randn(N,3) * torch.tensor([.01, 1, .2]).unsqueeze(0).expand(N,3)
    pts = trn(trn(parameterisation.get_R()) @ trn(pts))
    #intensities = torch.ones_like(pts)[:,0]

    parameters = torch.tensor([[100, 0, 1, 1.]])

    parameterisation.save_ply_with_axes(pts, Path('hax/test.ply'), parameters)

    

def _main()->None:
    
    data, metadata = data_legant.load()

    dataset = volumetric.SimpleVolumetric3Plane(data, metadata, device.device)

    print(f'Res = {metadata}')

    params_initial = train.TrainingParameters()
    params_initial.batch_size = 4
    params_initial.validity_weight=1000
    params_initial.checkpoint_every=100

    params_initial.schedule[0].epochs = 1000
    params_initial.schedule[0].initial_psf = 10000
    params_initial.schedule[0].final_psf = 5000
    params_initial.schedule[0].psf_step_every= 100
    params_initial.schedule[0].initial_lr= 0.0001
    params_initial.schedule[0].final_lr= 0.0001



    params_refine = train.TrainingParameters()
    params_refine.batch_size = 16
    params_refine.validity_weight=1000
    params_refine.checkpoint_every=100

    params_refine.schedule[0].epochs = 10000
    params_refine.schedule[0].initial_psf = 5000
    params_refine.schedule[0].final_psf = 500
    params_refine.schedule[0].psf_step_every= 1000
    params_refine.schedule[0].initial_lr= 0.0001
    params_refine.schedule[0].final_lr= 0.00001


    for i in range(1):

        net, parameterisation = PredictReconstruction(
            model_size=500,
            nm_per_pixel_xy=metadata.xy_nm_pix,
            image_size_xy=32,
            z_scale=metadata.z_nm_pix/metadata.z_nm_pix,
            image_size_z=32)

        parameterisation.max_expand = torch.tensor(1.0)
        parameterisation.max_rotation = torch.tensor(0.0 * torch.pi/180.)
        parameterisation.max_spacing = torch.tensor(24000.)
        parameterisation.min_spacing = torch.tensor(8000.)
        net.to(device.device)
        
        fast = cast(train.GeneralPredictReconstruction, torch.compile(net))
        train.retrain(fast, dataset, params_initial, f'run-{i:03}-phase_0')

        parameterisation.max_expand = torch.tensor(1.5)
        parameterisation.max_rotation = torch.tensor(45.0 * torch.pi/180.)

        train.retrain(fast, dataset, params_refine, f'run-{i:03}-phase_1')

        
        
if __name__ == "__main__":
    _main()



