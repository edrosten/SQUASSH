from __future__ import annotations
from typing import List, Optional, Tuple, Callable
from abc import abstractmethod
import math

import torch
from torch import Tensor
import pystrict 
import tqdm

from localisation_data import GeneralLocalisationDataSet, fwhm_to_sigma
from data.volumetric import Metadata
from data import volumetric

@pystrict.strict
class GeneralVolumetricDataset(GeneralLocalisationDataSet):
    '''Dataset for volumetric data, with known xy ans z size'''
    @abstractmethod
    def xy_size(self)->int:
        '''Image size'''

    @abstractmethod
    def z_size(self)->int:
        '''Stack size'''


@pystrict.strict
class DataSet6Plane(GeneralVolumetricDataset):
    '''lol'''
    def __init__(self, data_cubes: Tensor, metadata: Metadata, device: Optional[torch.device]=None):

        # Tensor is a big ass-cube of data
        assert len(data_cubes.shape) == 4 # NZYX

        self._xysize = data_cubes.shape[2]
        assert data_cubes.shape[2] == data_cubes.shape[3]

        self._zsize = data_cubes.shape[1]

        self._metadata = metadata

        self._data, xy_plane_pos_px, z_plane_pos_px = volumetric.compute_volumetric_6_plane(data_cubes, device)

        self._xy_plane_pos_nm = xy_plane_pos_px * metadata.xy_nm_pix
        self._z_plane_pos_nm = z_plane_pos_px * metadata.z_nm_pix

        n = len(self._data)
        dtype = data_cubes.dtype

        # Allocate memory for rendered data.
        self._rendered_data: List[Tensor] = [ 
            torch.zeros(n, self._xysize, self._xysize, device=device, dtype=dtype),
            torch.zeros(n, self._xysize, self._xysize, device=device, dtype=dtype),
            torch.zeros(n,  self._zsize, self._xysize, device=device, dtype=dtype),
            torch.zeros(n,  self._zsize, self._xysize, device=device, dtype=dtype),
            torch.zeros(n,  self._zsize, self._xysize, device=device, dtype=dtype),
            torch.zeros(n,  self._zsize, self._xysize, device=device, dtype=dtype)
        ]

        self._sigma_xy: Optional[float] = None
        self._device=device
    
    def xy_size(self)->int:
        '''Image size'''
        return self._xysize

    def z_size(self)->int:
        '''Stack size'''
        return self._zsize

    
    def projection_plane_position_nm(self)->Tuple[float, float]:
        '''Postion of the xy and z projection planes'''
        return self._xy_plane_pos_nm, self._z_plane_pos_nm

    def __len__(self)->int:
        return len(self._data)
    
    def __getitem__(self, idx:int)->List[torch.Tensor]:
        if self._sigma_xy is None:
            raise RuntimeError('Rendering sigma not set')
        
        # Add channel dimension
        return [ i[idx, :, :].unsqueeze(0).float() for i in self._rendered_data]

    def get_augmentations(self)->int:
        return 1

    def set_sigma(self, sigma_nm: float)->None:
        """Updates the rendering sigma. Idempotent."""
        if sigma_nm == self._sigma_xy:
            return
        
        self._sigma_xy = sigma_nm

        target_sigma_xy = sigma_nm
        target_sigma_z = target_sigma_xy * self._metadata.z_nm_pix / self._metadata.xy_nm_pix

        min_sigma_xy = fwhm_to_sigma(self._metadata.xy_fwhm_nm)
        min_sigma_z = fwhm_to_sigma(self._metadata.z_fwhm_nm)
        
        additional_sigma_z = math.sqrt(max(0, target_sigma_z**2 - min_sigma_z**2))
        additional_sigma_xy= math.sqrt(max(0, target_sigma_xy**2 - min_sigma_xy**2))
        

        _SIGMAS=3

        #Create the 1D kernels
        k_xy = torch.ones(1)
        k_z = torch.ones(1)

        print('New sigma values (z, xy):')
        print('nm: ', additional_sigma_z, additional_sigma_xy) 

        if additional_sigma_xy > 0:
            sigma_xy_px = additional_sigma_xy / self._metadata.xy_nm_pix
            k_r_xy = int(math.ceil(sigma_xy_px*_SIGMAS))
            k_xy = torch.exp(-torch.arange(-k_r_xy, k_r_xy+1)**2/(2*sigma_xy_px**2))
            k_xy /= k_xy.sum()
            print('x px: ', sigma_xy_px)
    
        if additional_sigma_z > 0:
            sigma_z_px = additional_sigma_z / self._metadata.z_nm_pix
            k_r_z = int(math.ceil(sigma_z_px*_SIGMAS))
            k_z = torch.exp(-torch.arange(-k_r_z, k_r_z+1)**2/(2*sigma_z_px**2))
            k_z /= k_z.sum()
            print('z px: ', sigma_z_px)


        k_xy_xy = (k_xy.unsqueeze(0) * k_xy.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(self._device)
        k_z_xy = (k_z.unsqueeze(1) * k_xy.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(self._device)


        for i, (xy_0, xy_1, yz_0, yz_1, xz_0, xz_1) in enumerate(tqdm.tqdm(self._data, desc='Blurring')):

            xy_0 = torch.nn.functional.conv2d(xy_0.view(1,1,*xy_0.shape), k_xy_xy, padding='same').view(*xy_0.shape) # pylint: disable=not-callable
            xy_1 = torch.nn.functional.conv2d(xy_1.view(1,1,*xy_1.shape), k_xy_xy, padding='same').view(*xy_1.shape) # pylint: disable=not-callable
            
            yz_0 = torch.nn.functional.conv2d(yz_0.view(1,1,*yz_0.shape), k_z_xy, padding='same').view(*yz_0.shape) # pylint: disable=not-callable
            yz_1 = torch.nn.functional.conv2d(yz_1.view(1,1,*yz_1.shape), k_z_xy, padding='same').view(*yz_1.shape) # pylint: disable=not-callable

            xz_0 = torch.nn.functional.conv2d(xz_0.view(1,1,*xz_0.shape), k_z_xy, padding='same').view(*xz_0.shape) # pylint: disable=not-callable
            xz_1 = torch.nn.functional.conv2d(xz_1.view(1,1,*xz_1.shape), k_z_xy, padding='same').view(*xz_1.shape) # pylint: disable=not-callable

            
            self._rendered_data[0][i,:,:] = xy_0
            self._rendered_data[1][i,:,:] = xy_1
            self._rendered_data[2][i,:,:] = yz_0
            self._rendered_data[3][i,:,:] = yz_1
            self._rendered_data[4][i,:,:] = xz_0
            self._rendered_data[5][i,:,:] = xz_1


def _gaussian_kernel(sigma: float)->torch.Tensor:
    _SIGMAS=3
    k = torch.ones(1)
    if sigma > 0:
        r = int(math.ceil(sigma*_SIGMAS))
        k = torch.exp(-torch.arange(-r, r+1)**2/(2*sigma**2))
        k /= k.sum()
    return k


@pystrict.strict
class GeneralPreprojectedVoulmetric3Plane(GeneralVolumetricDataset):
    '''
    Volumetric 3-plane dataset with an unspecified projection function
    '''
    def __init__(self, data: torch.Tensor, metadata: Metadata, project: Callable[[torch.Tensor, int],torch.Tensor], device: torch.device|None=None):
        self._sigma_xy: Optional[float] = None
        self._device = device
        self._xy_nm_per_pix = metadata.xy_nm_pix 
        self._z_nm_per_pix = metadata.z_nm_pix
        self._xy_sigma_nm = fwhm_to_sigma(metadata.xy_fwhm_nm) #Resolution of the data, i.e. lowest meaningful sigma
        self._z_sigma_nm = fwhm_to_sigma(metadata.z_fwhm_nm)


        self._data = [
            project(data,1),  # NzYX
            project(data,3),  # NZYx
            project(data,2),  # NZyX
        ]


        self._rendered_data = [
            torch.zeros(self._data[0].shape, device=device),
            torch.zeros(self._data[1].shape, device=device),
            torch.zeros(self._data[2].shape, device=device),
        ]

    def xy_size(self)->int:
        '''Size of the returned data'''
        return self._data[0].shape[2]
    def z_size(self)->int:
        '''Size of the returned data'''
        return self._data[1].shape[1]

    def set_sigma(self, sigma_nm: float)->None:
        """Updates the rendering sigma. Idempotent."""
        if sigma_nm == self._sigma_xy:
            return
        self._sigma_xy = sigma_nm
        
        target_sigma_xy = sigma_nm
        target_sigma_z = target_sigma_xy * self._z_sigma_nm/self._xy_sigma_nm

        
        additional_sigma_z = math.sqrt(max(0, target_sigma_z**2 - self._z_sigma_nm**2))
        additional_sigma_xy= math.sqrt(max(0, target_sigma_xy**2 - self._xy_sigma_nm**2))

        #Create the 1D kernels
        k_xy = _gaussian_kernel(additional_sigma_xy / self._xy_nm_per_pix)
        k_z = _gaussian_kernel(additional_sigma_z / self._z_nm_per_pix)

        k_xy_xy = (k_xy.unsqueeze(0) * k_xy.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(self._device)
        k_z_xy = (k_z.unsqueeze(1) * k_xy.unsqueeze(0)).unsqueeze(0).unsqueeze(0).to(self._device)

        for i, (xy, yz, xz) in enumerate(zip(tqdm.tqdm(self._data[0], desc='Blurring'),*self._data[1:])):

            # Zero pad: assume background is effectively zero
            xy = torch.nn.functional.conv2d(xy.view(1,1,*xy.shape).to(self._device), k_xy_xy, padding='same').view(*xy.shape) # pylint: disable=not-callable
            yz = torch.nn.functional.conv2d(yz.view(1,1,*yz.shape).to(self._device), k_z_xy, padding='same').view(*yz.shape) # pylint: disable=not-callable
            xz = torch.nn.functional.conv2d(xz.view(1,1,*xz.shape).to(self._device), k_z_xy, padding='same').view(*xz.shape) # pylint: disable=not-callable

            self._rendered_data[0][i,...] = xy
            self._rendered_data[1][i,...] = yz
            self._rendered_data[2][i,...] = xz

    def get_augmentations(self)->int:
        return 1
    
    def  __len__(self)->int:
        return self._rendered_data[0].shape[0]

    def __getitem__(self, idx:int)->List[torch.Tensor]:
        if self._sigma_xy is None:
            raise RuntimeError('Rendering sigma not set')

        # Add singleton channel dimension
        return [ i[idx, :, :].unsqueeze(0).float() for i in self._rendered_data]


def SimpleVolumetric3Plane(data: torch.Tensor, metadata: Metadata, device: torch.device|None=None)->GeneralPreprojectedVoulmetric3Plane:
    '''Volumetric 3-plane data using 3 sum projections'''
    return GeneralPreprojectedVoulmetric3Plane(data, metadata, lambda d,i: d.sum(i), device)

def MaxVolumetric3Plane(data: torch.Tensor, metadata: Metadata, device: torch.device|None=None)->GeneralPreprojectedVoulmetric3Plane:
    '''Volumetric 3-plane data using 3 sum projections'''
    return GeneralPreprojectedVoulmetric3Plane(data, metadata, lambda d,i: d.amax(i), device)
