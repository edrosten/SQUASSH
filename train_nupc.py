from typing import cast
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
import torch._dynamo
from pystrict import strict

import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import data.nucporesim   # noqa pylint:disable=unused-import
import train
import network
import save_ply
import device
from matrix import trn, so3_6D
import localisation_data
from localisation_data import LocalisationDataSetMultipleDan6



INTERMEDIATE=4096

class _Scale(nn.Module):
    def __init__(self, f: float):
        super().__init__()
        self.register_buffer("factor", torch.tensor(f))
        self.factor: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor: #pylint: disable=missing-function-docstring
        return x * self.factor

@strict
class AxialStretchRadialExpandWithGeneralShift(network.ModelParameterisation):
    '''Parameterise as a stretch along an axis and separate expansions normal to the axis.
    In add completely general shift for each point
    '''
    def __init__(self, npts: int)->None:
        super().__init__()
        #Principal axis is the axis of stretch and shrink, which is global
        #Stored as a 3 vector representing a direction
        self.principal_axis = torch.nn.parameter.Parameter(torch.rand(3))
        self._secondary_axis = torch.nn.parameter.Parameter(torch.rand(3))

        self.register_buffer("max_stretch_factor_axis", torch.tensor(1.0))
        self.register_buffer("max_stretch_factor_expand", torch.tensor(1.0))
        self.register_buffer("shift_amount_nm", torch.tensor(0.0))
        self.max_stretch_factor_axis: torch.Tensor
        self.max_stretch_factor_expand: torch.Tensor
        self.shift_amount_nm: torch.Tensor


        self.shift_network = nn.Sequential(
            nn.Linear(INTERMEDIATE, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Linear(1024, 3* npts),
            _Scale(10),
            nn.Tanh()
        )
        
        self.per_point_shift = False


    def number_of_parameters(self)->int:
        return 3 + INTERMEDIATE

    def _apply_parameterisation_simple(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]

        
        # In this model have 3 full expansion axes rather than coupling 2 and 3 together
        scale1 = self.max_stretch_factor_axis**torch.tanh(parameters[:,0])
        scale2 = self.max_stretch_factor_expand**torch.tanh(parameters[:,1])
        scale3 = self.max_stretch_factor_expand**torch.tanh(parameters[:,2])

        diag_scale = torch.stack([scale1, scale2, scale3], 1).diag_embed()

        R = self.get_R().unsqueeze(0).expand(batch_size, 3, 3)
        S = trn(R) @ diag_scale @ R
        points = trn(S @ trn(model_points).unsqueeze(0).expand(batch_size, 3, Nv))
        intensities = model_intensities.unsqueeze(0).expand(batch_size, Nv)

        # TODO aggregate scale here. This is just compatibility with the old one
        return points, intensities, scale1


    def _apply_parameterisation_crazy(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Stretch and expand, then per-point shift''' 
        points, intensities, scale1 = self._apply_parameterisation_simple(model_points, model_intensities, parameters)
        #  
        batch_size = parameters.shape[0]
        shifts = self.shift_network(parameters[:,3:]).reshape(batch_size, -1, 3) * self.shift_amount_nm
        return points + shifts, intensities, scale1


    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.per_point_shift:
            return self._apply_parameterisation_crazy(model_points, model_intensities, parameters)
        return self._apply_parameterisation_simple(model_points, model_intensities, parameters)

    def get_R(self)->torch.Tensor:
        '''Get the rotation matrix'''
        return so3_6D(torch.cat([self.principal_axis, self._secondary_axis]).unsqueeze(0)).squeeze(0)

    def get_axis(self)->Tensor:
        '''Return principal axis as unit vector'''
        return self.principal_axis / torch.sqrt((self.principal_axis**2).sum())

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





def PredictReconstruction(initial_model_size: int, final_model_size: int, nm_per_pixel_xy: float, image_size_xy:int, image_size_z: int, z_scale: float, data: list[Tensor])->tuple[network.GeneralPredictReconstruction, AxialStretchRadialExpandWithGeneralShift]:
    '''Predict R/t etc and rerender for a 6 plane rendering, also allow prediction of "opting out"'''
    d6render = localisation_data.RenderDan6(data)

    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->list[torch.Tensor]:
        return [i.unsqueeze(1) for i in d6render(
               centres=centres, 
               weights=weights,
               sigma_xy_nm=sigma_nm,
               nm_per_pixel_xy=nm_per_pixel_xy,
               z_scale=z_scale,
               xy_size=image_size_xy,
               z_size=image_size_z) ]

    parameterisation = AxialStretchRadialExpandWithGeneralShift(final_model_size) # LOL
    reconstructor=  network.GeneralPredictReconstruction(
        initial_model_size, 
        image_size_xy*nm_per_pixel_xy,
        renderer, 
        parameterisation,
        network.NetworkAny)

    return reconstructor, parameterisation
 


def _main()->None:
    nupc3d = [t.to(device.device).half() for t in data.nucporesim.sim_nups(1000)]
    #nupc3d = [t.to(device.device).half() for t in resi_data.load_3d()]
    #nupc3d = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]

    initial_points=35
    mult = 20
    scatter = 0.01

    SCALE=1.3
    rejection = 1.0

    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 3*SCALE,
        z_scale = 2
    )
    
    params_initial = train.TrainingParameters()
    params_initial.batch_size = 160 
    params_initial.validity_weight=rejection
    params_initial.checkpoint_every=100

    params_initial.schedule[0].epochs = 90
    params_initial.schedule[0].initial_psf = 50*SCALE
    params_initial.schedule[0].final_psf = 26*SCALE
    params_initial.schedule[0].psf_step_every= 30
    params_initial.schedule[0].initial_lr= 0.0001
    params_initial.schedule[0].final_lr= 0.0001

    params_initial.schedule.append(train.TrainingSegment())
    params_initial.schedule[1].epochs = 300
    params_initial.schedule[1].initial_psf = 19*SCALE
    params_initial.schedule[1].final_psf = 10.0*SCALE
    params_initial.schedule[1].psf_step_every= 100
    params_initial.schedule[1].initial_lr= 0.0001
    params_initial.schedule[1].final_lr= 0.0001

    dataset_initial = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=8, device=device.device)


    params_refine = train.TrainingParameters()
    params_refine.batch_size = 10
    params_refine.validity_weight=rejection
    params_refine.checkpoint_every=100

    params_refine.schedule[0].epochs = 500
    params_refine.schedule[0].initial_psf = 10.0*SCALE
    params_refine.schedule[0].final_psf = 10.0*SCALE
    params_refine.schedule[0].psf_step_every= 300
    params_refine.schedule[0].initial_lr= 0.0002
    params_refine.schedule[0].final_lr= 0.00005

    params_final = train.TrainingParameters()
    params_final.batch_size = 10
    params_final.validity_weight=rejection
    params_final.checkpoint_every=100

    params_final.schedule[0].epochs = 500
    params_final.schedule[0].initial_psf = 10.0*SCALE
    params_final.schedule[0].final_psf = 10.0*SCALE
    params_final.schedule[0].psf_step_every= 300
    params_final.schedule[0].initial_lr= 0.00005
    params_final.schedule[0].final_lr= 0.00005




    dataset_refine = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)

    torch._dynamo.config.cache_size_limit=512  # pylint: disable=protected-access

    torch.compiler.reset()

    net, parameterisation = PredictReconstruction(initial_model_size=initial_points, final_model_size=initial_points*mult, **vars(data_parameters), data=nupc3d)
    parameterisation.max_stretch_factor_axis = torch.tensor(2.0)
    parameterisation.max_stretch_factor_expand = torch.tensor(1.0)
    net.to(device.device)
    
    net._model_intensities.requires_grad=False  # pylint: disable=protected-access


    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset_initial, params_initial, 'phase_0')
    
    scale = net.get_model()[0].abs().max().item()
    
    old_pts, old_weights = (j.detach() for j in net.get_model())

    new_pts = torch.nn.functional.interpolate(old_pts.unsqueeze(0).unsqueeze(0), scale_factor=[mult,1]).squeeze(0).squeeze(0)
    new_pts += torch.randn(new_pts.shape, device=device.device) * scale * scatter

    new_weights = torch.nn.functional.interpolate(old_weights.unsqueeze(0).unsqueeze(0), scale_factor=mult).squeeze(0).squeeze(0)
    
    net.set_model(new_pts, new_weights)
    net._model_intensities.requires_grad=True  # pylint: disable=protected-access

    # principal axis ought to have optimized OK by now. It has to be turned off for 
    # optimization at this point, otherwise with 3 generic axes, there's no 
    # real notion of overall orientation
    parameterisation.principal_axis.requires_grad = False
    parameterisation.max_stretch_factor_expand = torch.tensor(1.3, device=device.device)
    
    torch.compiler.reset() # Otherwise it crashes on torch 2.7
    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset_refine, params_refine, 'phase_1')

    
    torch.compiler.reset() # Otherwise it crashes on torch 2.7
    parameterisation.shift_amount_nm = torch.tensor(7)
    parameterisation.per_point_shift=True
    
    # Turn off gradients for everything
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    # Turn gradients back on only for the per-point shift
    parameterisation.shift_network.train()
    for p in parameterisation.shift_network.parameters():
        p.requires_grad = True

    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset_refine, params_final, 'phase_2')

if __name__ == "__main__":
    _main()

