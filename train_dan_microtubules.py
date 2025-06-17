from __future__ import annotations

from typing import cast
from pathlib import Path

import torch
from torch import Tensor
from pystrict import strict

from localisation_data import LocalisationDataSetMultipleDan6, RenderDan6
import data_dan_microtubules
import device
import train
import network
import matrix
from train_spectrin import contingumax
from matrix import trn
import save_ply

@strict
class AxialRepeat(network.ModelParameterisation):
    '''Parameterise as a stretch along an axis and expansion normal to the axis.
    The principle axis is optimized as part of the model'
    '''
    def __init__(self, min_repetitions: int, max_repetitions: int)->None:
        super().__init__()
        #Principal axis is the axis of stretch and shrink, which is global
        #Stored as a 3 vector representing a direction
        self._principal_axis = torch.nn.parameter.Parameter(torch.rand(3))
        self._expasion_axis = torch.nn.parameter.Parameter(torch.rand(3))
        self._spacing_parameter = torch.nn.parameter.Parameter(torch.rand(1))


        self.register_buffer("min_repetition_length", torch.tensor(1.0))
        self.register_buffer("max_repetition_length", torch.tensor(2.0))
        self.register_buffer("semi_radial_expand", torch.tensor(1.0))
        self.register_buffer("elongation", torch.tensor(1.0))
        self._min_repetitions = min_repetitions
        self._repetitions = max_repetitions

        self.min_repetition_length: torch.Tensor
        self.max_repetition_length: torch.Tensor
        self.semi_radial_expand: torch.Tensor

        # This only makes sense for a fixed model
        self._overall_scale_value = torch.nn.parameter.Parameter(torch.zeros(1))
        self.register_buffer('max_scale', torch.tensor([1.0]))

    def number_of_parameters(self)->int:
        return self._repetitions - self._min_repetitions + 2

    def get_global_scale(self)->torch.Tensor:
        '''Get global scale'''
        return torch.pow(self.max_scale, self._overall_scale_value.tanh())
    
    def get_spacing(self)->torch.Tensor:
        '''Get the spacing'''
        return (self.max_repetition_length/self.min_repetition_length).pow(torch.sigmoid(self._spacing_parameter)) * self.min_repetition_length
    

    def get_axis(self)->torch.Tensor:
        '''Return principal axis as unit vector'''
        return torch.nn.functional.normalize(self._principal_axis, dim=0)

    def get_expansion_axis(self)->Tensor:
        '''Return the expan axis (ortho to principal) as unit vector'''
        return matrix.normalized_gram_schmidt_reduce(self.get_axis().unsqueeze(0), self._expasion_axis.unsqueeze(0)).squeeze(0)

    def compute_repetition_weights_from_parameters(self, parameters: Tensor)->Tensor:
        '''Compute the weights on the repeaitng units'''
        batch_size = parameters.shape[0]
        if self._repetitions == self._min_repetitions:
            return torch.ones(batch_size, self._min_repetitions, device=parameters.device, dtype=parameters.dtype)
        
        repetition_logits = parameters[:,0:self._repetitions-self._min_repetitions]
        return torch.cat([torch.ones(batch_size, self._min_repetitions, device=parameters.device, dtype=parameters.dtype), contingumax(repetition_logits)], 1)    
    
    def compute_semi_radial_expansion_from_parameters(self, parameters: Tensor)->Tensor:
        '''Compute the expansion factor'''
        return cast(Tensor, self.semi_radial_expand ** parameters[:,self._repetitions-self._min_repetitions].tanh())

    def compute_elongation_from_parameters(self, parameters: Tensor)->Tensor:
        '''Compute the elongation factor'''
        return cast(Tensor, self.elongation ** parameters[:,self._repetitions-self._min_repetitions+1].tanh())

    def _apply_parameterisation(self, model_points: torch.Tensor, model_intensities: torch.Tensor, parameters: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''stretch and expand'''
        batch_size = parameters.shape[0]
        Nv = model_points.shape[0]
        dev = model_points.device
        dtype = model_points.dtype
        
        length = self.get_spacing()
        axis = self.get_axis()
        
        global_scale = self.get_global_scale()
        Sc = matrix.scale_along_axis_and_expand_matrix(self.get_axis(), torch.ones_like(global_scale), global_scale).squeeze(0)
        points = trn(Sc@trn(model_points)).unsqueeze(0).expand(batch_size, Nv, 3)

        repetition_weights = self.compute_repetition_weights_from_parameters(parameters)
        expansion = self.compute_semi_radial_expansion_from_parameters(parameters)
        elongation = self.compute_elongation_from_parameters(parameters)
        
        Sx = matrix.scale_along_axis_matrix(self.get_expansion_axis(), expansion)
        Se = matrix.scale_along_axis_matrix(self.get_axis(), elongation)


        points  = trn((Se @ Sx) @ trn(points))

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

        return shifted_repeated_points, r_intensitites, torch.ones(batch_size, device=points.device)


    def save_ply_with_axes(self, model_points: torch.Tensor, name:Path)->None:
        '''Dump out a visualisation'''
        length=(model_points **2).sum(1).max().sqrt().item()
        N=100
        axis = torch.arange(start=-N, end=N+1)/N * length
        axis = axis.unsqueeze(1).expand(axis.shape[0], 3)
        axis1 = axis * self.get_axis().cpu().unsqueeze(0).expand(axis.shape)

        to_write: list[Tensor | tuple[Tensor, tuple[int, int, int]]] = [ model_points.cpu(), (axis1, (255,0,0)) ]
        save_ply.save(name, to_write)


# pylint: disable=too-many-positional-arguments
def PredictReconstructionRepetitionD6(model_size: int, nm_per_pixel_xy: float, image_size_xy:int, image_size_z: int, z_scale: float, data: list[Tensor], min_repetitions:int=4, max_repetitions:int=6)->tuple[network.GeneralPredictReconstruction, AxialRepeat]:
    '''Predict R/t etc and rerender for a 6 plane rendering, also allow prediction of "opting out"'''
    d6render = RenderDan6(data)

    def renderer(centres: torch.Tensor, weights: torch.Tensor, sigma_nm: torch.Tensor)->list[torch.Tensor]:
        return [i.unsqueeze(1) for i in d6render(
               centres=centres, 
               weights=weights,
               sigma_xy_nm=sigma_nm,
               nm_per_pixel_xy=nm_per_pixel_xy,
               z_scale=z_scale,
               xy_size=image_size_xy,
               z_size=image_size_z) ]

    parameterisation = AxialRepeat(min_repetitions, max_repetitions)
    
    reconstructor = network.GeneralPredictReconstruction(
        model_size=model_size, 
        volume_cube_size_nm=image_size_xy*nm_per_pixel_xy, 
        renderfunc=renderer, 
        parameterisation=parameterisation,
        network_factory=network.NetworkAny)


    return reconstructor, parameterisation
def _main()->None:
    
    data3d = [u.to(device.device).half() for _,t in data_dan_microtubules.load_3d_3(segment_length=128).items() for u in t]


    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 2.0,
        z_scale = 1
    )

    dataset_initial = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=data3d, augmentations=2, device=device.device)
    dataset_initial.set_batch_size(1)
    torch._dynamo.config.cache_size_limit=512 # pylint: disable=protected-access # but probably don't need this anymore

    for i in range(1):

        net, parameterisation = PredictReconstructionRepetitionD6(
            model_size=280, 
            **vars(data_parameters), 
            data=data3d,
            min_repetitions = 3,
            max_repetitions = 5
        )

        parameterisation.min_repetition_length = torch.tensor(14.)
        parameterisation.max_repetition_length = torch.tensor(18.)
        parameterisation.semi_radial_expand = torch.tensor(1.2)

        net.to(device.device)
        
        params = train.TrainingParameters()
        params.batch_size = 20
        params.validity_weight=0.6

        params.schedule[0].epochs = 3000
        params.schedule[0].initial_psf = 20
        params.schedule[0].final_psf = 8
        params.schedule[0].psf_step_every= 100
        params.schedule[0].initial_lr= 0.0001
        params.schedule[0].final_lr= 0.0001
        

        fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
        train.retrain(fast, dataset_initial, params, f'run-{i:03}-phase_0')

if __name__ == "__main__":
    #_adhoc_test()
    _main()


