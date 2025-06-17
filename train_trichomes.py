from __future__ import annotations
from typing import cast
from pathlib import Path
import torch
from torch import Tensor
import torch._dynamo

import train
import device
import network
import data_littlejohn
import volumetric
import render
import save_ply
from data.volumetric import Metadata

# This seems very general purpose. Maybe move it?
def make_renderer(metadata: Metadata, dataset: volumetric.DataSet6Plane)->network.RenderFunc:
    '''Make a renderer'''
    z_scale = metadata.z_nm_pix / metadata.xy_nm_pix
    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->list[torch.Tensor]:
        return [ i.unsqueeze(1) for i in render.render_multiple_dan6(
            centres, 
            sigma_xy_z_nm=torch.stack([sigma_nm, sigma_nm*z_scale], -1), 
            weights=weights, 
            nm_per_pixel_xy=metadata.xy_nm_pix,
            nm_per_pixel_z=metadata.z_nm_pix,
            xy_size=dataset.xy_size(), 
            z_size=dataset.z_size(),
            max_abs_xy=dataset.projection_plane_position_nm()[0],
            max_abs_z=dataset.projection_plane_position_nm()[1]
            )]
    return renderer


class TrichomeParameterisation(network.ModelParameterisation):
    '''General purpose parameterisation, which uses vectors at the corners of the visible cube
    and linearly interpolates to find a distortion vector (additive) at an arbitrary point'''

    def __init__(self, volume_cube_size_nm: float)->None:
        super().__init__()
        self.register_buffer("max_scale_factor", torch.tensor(1.0))
        self.max_scale_factor: torch.Tensor

        self.register_buffer("max_distortion_factor", torch.tensor(1.0))
        self.max_distortion_factor: torch.Tensor

        self.register_buffer("volume_cube_size_nm", torch.tensor(volume_cube_size_nm))
        self.volume_cube_size_nm: torch.Tensor

    def number_of_parameters(self)->int:
        return 2*2*2*3+1


    def compute_scale_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        '''Compute the scale factor from the logits'''
        return cast(torch.Tensor, self.max_scale_factor ** parameters[:,0].tanh())

    def compute_distortions_from_parameters(self, parameters: torch.Tensor)->torch.Tensor:
        '''Compute the distortion cube'''
        batch_size = parameters.shape[0]
        return  self.max_distortion_factor * parameters[:,1:1+2*2*2*3].reshape(batch_size, 3, 2, 2, 2).tanh()

    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = parameters.shape[0]
        npts = model_points.shape[0]

        scale = self.compute_scale_from_parameters(parameters)


        # This will force the distortions to act around the centre
        model_points = network.center_points(model_points.unsqueeze(0), model_intensities.unsqueeze(0)).squeeze(0)

        # Parameterisation forms a 2x2x2 grid of x, y, z distortions

        # Need to map model points to [-1,1] for grid_sample (kinda)
        # volume_cube_size_nm is the size across, so [0,1] or [-.5, .5]
        #
        # Note that for 3D interpolation, a 3D output is expected
        lookup_positions = (2 * model_points / self.volume_cube_size_nm).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 1, -1, 3)
        
        # Need do predict x, y, z deltas 
        distortion_cube = self.compute_distortions_from_parameters(parameters)

        distortions = torch.nn.functional.grid_sample(distortion_cube, lookup_positions, padding_mode="border", align_corners=True)*self.volume_cube_size_nm

        # distortions.shape = B 3 1 1 N
        distortions = distortions.squeeze(3).squeeze(2)
        assert distortions.shape == torch.Size((batch_size, 3, npts))

        new_points = (model_points.unsqueeze(0).expand(batch_size, npts, 3) + distortions.permute(0,2,1))*scale.unsqueeze(1).unsqueeze(2).expand(batch_size, npts, 3)
        new_intensities = model_intensities.unsqueeze(0).expand(batch_size, npts)

        return network.center_points(new_points, new_intensities), new_intensities, scale
        

    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path)->None:
        '''Dump out a visualisation'''
        to_write: list[Tensor | tuple[Tensor, tuple[int, int, int]]] = [ model_points.cpu() ]
        save_ply.save(name, to_write)


def make_network_dan6(dataset: volumetric.DataSet6Plane, metadata: Metadata)->tuple[network.GeneralPredictReconstruction, TrichomeParameterisation]:
    '''Make a network and parameterisation pair for general trichome parameterisation'''
    volume_cube_size_nm = dataset.xy_size() * metadata.xy_nm_pix
    renderer = make_renderer(metadata, dataset)

    parameterisation = TrichomeParameterisation(volume_cube_size_nm)
    net = network.GeneralPredictReconstruction(
            model_size=500, 
            volume_cube_size_nm = volume_cube_size_nm,
            renderfunc = renderer,
            parameterisation = parameterisation,
            network_factory = network.NetworkAnyWithoutBug)

    net._max_translation *= 3 #From 10% to 30%? This should be done better and maybe with a buffer...
    
    return net, parameterisation

def _make_network_3plane(dataset: volumetric.GeneralPreprojectedVoulmetric3Plane, metadata: Metadata)->tuple[network.GeneralPredictReconstruction, TrichomeParameterisation]:
    volume_cube_size_nm = dataset.xy_size() * metadata.xy_nm_pix
    #renderer = make_renderer(metadata, dataset)

    z_scale = metadata.z_nm_pix / metadata.xy_nm_pix
    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->list[torch.Tensor]:
        return [ i.unsqueeze(1) for i in render.render_multiple(
            centres, 
            sigma_xy_z_nm=torch.stack([sigma_nm, sigma_nm*z_scale], -1), 
            weights=weights, 
            nm_per_pixel_xy=metadata.xy_nm_pix,
            nm_per_pixel_z=metadata.z_nm_pix,
            xy_size=dataset.xy_size(), 
            z_size=dataset.z_size(),
            )]

    parameterisation = TrichomeParameterisation(volume_cube_size_nm)
    net = network.GeneralPredictReconstruction(
            model_size=500, 
            volume_cube_size_nm = volume_cube_size_nm,
            renderfunc = renderer,
            parameterisation = parameterisation,
            network_factory = network.NetworkAnyWithoutBug)

    net._max_translation *= 3 #From 10% to 30%? This should be done better and maybe with a buffer...
    
    return net, parameterisation

def _main()->None:

    #trichomes, metadata = data_trichomes.load_dataset_1()
    trichomes, metadata = data_littlejohn.load()

    #trichomes=trichomes[[10,15],...]

    dataset = volumetric.DataSet6Plane(trichomes, metadata, device.device)
    print(metadata)

    print(f'Res = {metadata}')
    print(len(dataset))

    params = train.TrainingParameters()
    params.batch_size = 16
    params.validity_weight=1.0
    params.checkpoint_every=200
    params.normalize_by_group=True

    params.schedule[0].epochs = 8000
    params.schedule[0].initial_psf = 10
    params.schedule[0].final_psf = 10
    params.schedule[0].psf_step_every= 100
    params.schedule[0].initial_lr= 0.0001
    params.schedule[0].final_lr= 0.0001


    torch._dynamo.config.cache_size_limit=512 # pylint: disable=protected-access
    for i in range(1):

        net, parameterisation = make_network_dan6(dataset, metadata)
        parameterisation.max_scale_factor = torch.tensor(1.5)
        parameterisation.max_distortion_factor = torch.tensor(0.1)
        print(list(parameterisation.named_buffers()))

        net.to(device.device)

        state = torch.load('log/1736113205-9e0c47da754fce2736f55bd266cfa419dedca99e/run-000-phase_0/final_net.zip')
        state = {k[10:]:v for k,v in state.items()}
        net.load_state_dict(state)
        parameterisation.max_distortion_factor = torch.tensor(1.5)
        
        fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
        train.retrain(fast, dataset, params, f'run-{i:03}-phase_1') 
        
if __name__ == "__main__":
    _main()



