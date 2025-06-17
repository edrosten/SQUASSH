from __future__ import annotations
from typing import TypeVar
import math
from pathlib import Path
from dataclasses import dataclass

from bioio import BioImage
import bioio_czi
from torch import Tensor
import torch

from ..volumetric import Metadata

_FILES_10_17_2024 = [
    "OneDrive_1_10-17-2024/01.czi",
    "OneDrive_1_10-17-2024/02.czi",
    "OneDrive_1_10-17-2024/03.czi",
    "OneDrive_1_10-17-2024/04.czi",
    "OneDrive_1_10-17-2024/05.czi",
    "OneDrive_1_10-17-2024/06.czi",
    "OneDrive_1_10-17-2024/07.czi",
    "OneDrive_1_10-17-2024/08.czi",
    "OneDrive_1_10-17-2024/09.czi",
    "OneDrive_1_10-17-2024/10.czi",
    "OneDrive_1_10-17-2024/11.czi",
    "OneDrive_1_10-17-2024/12.czi",
]

_labels_10_17_2024 = Path(__file__).parent/'OneDrive_1_10-17-2024'/'labels.zip'

_FILES_Oct_28 = [
    "Oct_28th/10_28_01_128.czi",
    "Oct_28th/10_28_02_128.czi",
    "Oct_28th/10_28_03_128.czi",
    "Oct_28th/10_28_04_128.czi",
    "Oct_28th/10_28_05_128.czi",
    "Oct_28th/10_28_06_128.czi",
    "Oct_28th/10_28_07_128.czi",
    "Oct_28th/10_28_08_128.czi",
    "Oct_28th/10_28_09_128.czi",
    "Oct_28th/10_28_10_128.czi",
    "Oct_28th/10_28_11_128.czi",
    "Oct_28th/10_28_12_128.czi",
    "Oct_28th/10_28_13_128.czi",
    "Oct_28th/10_28_14_128.czi",
    "Oct_28th/10_28_15_128.czi",
    "Oct_28th/10_28_16_128.czi",
    "Oct_28th/10_28_17_128.czi",
    "Oct_28th/10_28_18_128.czi",
    "Oct_28th/10_28_19_128.czi",
    "Oct_28th/10_28_21_128.czi",
    "Oct_28th/10_28_22_128.czi",
    "Oct_28th/10_28_23_128.czi",
    "Oct_28th/10_28_24_128.czi",
    "Oct_28th/10_28_25_128.czi",
    "Oct_28th/10_28_26_128.czi",
    "Oct_28th/10_28_27_128.czi",
    "Oct_28th/10_28_28_128.czi",
    "Oct_28th/10_28_29_128.czi",
    "Oct_28th/10_28_30_128.czi",
    "Oct_28th/10_28_31_128.czi",
    "Oct_28th/10_28_32_128.czi",
    "Oct_28th/10_28_33_128.czi",
    "Oct_28th/10_28_34_128.czi",
    "Oct_28th/10_28_35_128.czi",
    "Oct_28th/10_28_36_128.czi",
    "Oct_28th/10_28_37_128.czi",
    "Oct_28th/10_28_38_128.czi",
    "Oct_28th/10_28_39_128.czi",
    "Oct_28th/10_28_40_128.czi",
    "Oct_28th/10_28_41_128.czi",
]
_labels_Oct_28 = Path(__file__).parent/'Oct_28th'/'labels.zip'

_FILES_Nov_26 = [
    "Nov_26th/11_19_01.czi",
    "Nov_26th/11_19_02.czi",
    "Nov_26th/11_19_03.czi",
    "Nov_26th/11_19_04.czi",
    "Nov_26th/11_19_05.czi",
    "Nov_26th/11_19_06.czi",
    "Nov_26th/11_19_07.czi",
    "Nov_26th/11_19_08.czi",
    "Nov_26th/11_19_09.czi",
    "Nov_26th/11_19_10.czi",
    "Nov_26th/11_19_11.czi",
    "Nov_26th/11_19_12.czi",
    "Nov_26th/11_19_13.czi",
    "Nov_26th/11_19_14.czi",
    "Nov_26th/11_19_15.czi",
    "Nov_26th/11_19_16.czi",
    "Nov_26th/11_19_17.czi",
    "Nov_26th/11_19_18.czi",
    "Nov_26th/11_19_19.czi",
    "Nov_26th/11_19_20.czi",
    "Nov_26th/11_19_21.czi",
    "Nov_26th/11_19_22.czi",
    "Nov_26th/11_19_23.czi",
    "Nov_26th/11_19_24.czi",
    "Nov_26th/11_19_25.czi",
    "Nov_26th/11_19_26.czi",
    "Nov_26th/11_19_27.czi",
    "Nov_26th/11_19_28.czi",
    "Nov_26th/11_19_29.czi",
    "Nov_26th/11_19_30.czi",
    "Nov_26th/11_19_31.czi",
    "Nov_26th/11_19_32.czi",
    "Nov_26th/11_19_33.czi",
    "Nov_26th/11_19_34.czi",
    "Nov_26th/11_19_35.czi",
    "Nov_26th/11_19_36.czi",
    "Nov_26th/11_19_37.czi",
    "Nov_26th/11_19_38.czi",
    "Nov_26th/11_19_39.czi",
    "Nov_26th/11_19_40.czi",
    "Nov_26th/11_19_41.czi",
    "Nov_26th/11_19_42.czi",
    "Nov_26th/11_19_43.czi",
    "Nov_26th/11_19_44.czi",
    "Nov_26th/11_19_45.czi",
    "Nov_26th/11_19_46.czi",
    "Nov_26th/11_19_47.czi",
    "Nov_26th/11_19_48.czi",
    "Nov_26th/11_19_49.czi",
    "Nov_26th/11_19_50.czi",
    "Nov_26th/11_19_51.czi",
    "Nov_26th/11_19_52.czi",
    "Nov_26th/11_19_53.czi",
    "Nov_26th/11_19_54.czi",
    "Nov_26th/11_19_55.czi",
    "Nov_26th/11_19_56.czi",
    "Nov_26th/11_19_57.czi",
    "Nov_26th/11_19_58.czi",
    "Nov_26th/11_19_59.czi",
    # "Nov_26th/11_19_60.czi", # This one has a weird resolution, 10% off 
    "Nov_26th/11_19_61.czi",
    "Nov_26th/11_19_62.czi",
    "Nov_26th/11_19_63.czi",
]
_labels_Nov_26 = Path(__file__).parent/'Nov_26th'/'labels.zip'


@dataclass
class Label:
    """Forgot to put _, too late to make intenal now lol"""
    angle: float
    point_1: Tensor|None = None
    point_2: Tensor|None = None


def _R_Y(angle: float, device:torch.device|None=None)->Tensor:
    c = math.cos(angle)
    s = math.sin(angle)

    R = torch.tensor([
        [  c,  0,  s],
        [  0,  1,  0],
        [ -s,  0,  c],
    ], device=device)

    return R


# Note vol coords * _vol_nm_scale_xyz should give the same results as
# (torch.arange(shape[2])-(shape[2]-1)/2) * metadata.xy_nm_pix
def _vol_coords_xyz(shape: torch.Size, d:torch.device|None=None)->Tensor:
    # Coordinate lookup is always done in the range [-1,1], as per OpenGL
    xs = torch.arange(shape[2], device=d).unsqueeze(0).unsqueeze(0).expand(shape) / (shape[2]-1) * 2 - 1
    ys = torch.arange(shape[1], device=d).unsqueeze(0).unsqueeze(2).expand(shape) / (shape[1]-1) * 2 - 1
    zs = torch.arange(shape[0], device=d).unsqueeze(1).unsqueeze(2).expand(shape) / (shape[0]-1) * 2 - 1
    return torch.stack([xs, ys, zs], 3)


def _vol_nm_scale_xyz(shape: torch.Size, metadata: Metadata)->torch.Tensor:
    # xscale goes from -1 to 1
    # -1 corresponts to -x_size/2 * nm_per_pix
    # 1 corresponds to  x_size/2 * nm_per_pix

    # Scale volume coords to nanometers
    x_scale = (shape[2]-1)/2 * metadata.xy_nm_pix
    y_scale = (shape[1]-1)/2 * metadata.xy_nm_pix
    z_scale = (shape[0]-1)/2 * metadata.z_nm_pix
    return torch.tensor([x_scale, y_scale, z_scale])





def _cut_volume(volume: Tensor, metadata: Metadata, angle: float, point_1: Tensor, point_2: Tensor)->Tensor:
    d = volume.device
    
    R = _R_Y(angle, d).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*volume.shape, 3, 3)
    scale = _vol_nm_scale_xyz(volume.shape, metadata).reshape(1,1,1,3).expand(*volume.shape, 3)
    xyz_nm = _vol_coords_xyz(volume.shape, d) * scale
    rotated_coords_nm = (R @ xyz_nm.unsqueeze(-1)).squeeze(-1) 

    point_1_nm = (point_1 - (torch.tensor([volume.shape[2], volume.shape[1]])/2-1)) * metadata.xy_nm_pix
    point_2_nm = (point_2 - (torch.tensor([volume.shape[2], volume.shape[1]])/2-1)) * metadata.xy_nm_pix

    vec2 = point_1_nm - point_2_nm

    vec3 = torch.cat([vec2, torch.zeros(1)]) #This vector is on the plane
    other_vec3 = torch.tensor([0,0,1.]).to(vec3) # The plane goes in Z, in the rotated frame
    
    normal = torch.linalg.cross(vec3, other_vec3) # pylint: disable=not-callable

    plane_point = torch.cat([point_1_nm, torch.zeros(1)]).reshape(1,1,1,3).expand_as(rotated_coords_nm)

    cut  =  (normal.reshape(1,1,1,1,3).expand(*volume.shape, 1, 3) @ (rotated_coords_nm - plane_point).unsqueeze(-1)) < 0
    cut = cut.squeeze(-1).squeeze(-1)

    cut_volume = volume.clone()
    cut_volume[cut] = 0
    return cut_volume



T = TypeVar('T')

def _not_none(a: T|None)->T:
    if a is None:
        raise RuntimeError('Expected value, got None')
    return a


def _approx_eq(a: float, b: float, epsilon: float)->bool:
    return abs(a-b)/(a+b) < epsilon

def _load_file(filename: Path)->tuple[Tensor, Metadata]:
    img = BioImage(filename, reader=bioio_czi.Reader)

    if img.physical_pixel_sizes.Y is None or img.physical_pixel_sizes.X is None:
        raise RuntimeError(f'Resolution missing in {filename}')

    if not _approx_eq(img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X, 1e-8):
        raise RuntimeError(f'Resolution mismatch in {filename}')

    fwhm_ratio = 1/3. # effective PSF of the resulting data cube
    rx = _not_none(img.physical_pixel_sizes.X)
    rz = _not_none(img.physical_pixel_sizes.Z)
    metadata = Metadata(
        xy_nm_pix = rx, 
        z_nm_pix = rz,
        xy_fwhm_nm = rx * fwhm_ratio,
        z_fwhm_nm = rz * fwhm_ratio,
    )

    return torch.tensor(img.data[0,0,...]), metadata

def _pad(stack: Tensor, xy_size: int, z_size: int)->Tensor:

    
    padded = torch.zeros(z_size, xy_size, xy_size)
    off = (torch.tensor(padded.shape)-torch.tensor(stack.shape))//2
    slices = [ slice(i, i+j) for i,j in zip(off, stack.shape)]
    #print(slices)
    padded[*slices] = stack
    return padded

def _concat(loaded: list[tuple[Tensor, Metadata]], xy_size: int, z_size: int)->tuple[Tensor, Metadata]:
     
    xy_nm_pix = loaded[0][1].xy_nm_pix
    z_nm_pix = loaded[0][1].z_nm_pix
    good = [ _approx_eq(xy_nm_pix, i[1].xy_nm_pix, 5e-3) and _approx_eq(z_nm_pix, i[1].z_nm_pix, 1e-3) for i in loaded]

    if not all(good):
        for (_,m),g in zip(loaded, good):
            print(m, "" if g else "*"*10)
        raise RuntimeError('Metadata mismatch')
    
    return torch.stack([_pad(i[0], xy_size, z_size) for i in loaded], 0), loaded[0][1]

def _half_xy(stacks: Tensor, metadata: Metadata)->tuple[Tensor, Metadata]:
    scaled_stacks = torch.nn.functional.avg_pool2d(stacks, 2) # pylint: disable=not-callable

    scaled_metadata = Metadata(
        xy_nm_pix = metadata.xy_nm_pix * 2,
        z_nm_pix = metadata.z_nm_pix,
        xy_fwhm_nm = metadata.xy_fwhm_nm * 2,
        z_fwhm_nm = metadata.z_fwhm_nm
    )
    return scaled_stacks, scaled_metadata


def _load_local_list(files: list[str])->list[tuple[Tensor, Metadata]]:
    return [ _load_file(Path(__file__).parent/i) for i in files]

def load_10_17_2024()->tuple[Tensor, Metadata]:
    '''Load the 10_17_2024 dataset'''
    loaded = _load_local_list(_FILES_10_17_2024)
    return _half_xy(*_concat(loaded, 128, 32))


_FINAL_SIZE = (178,60)

def _load_and_reshape(files: list[str])->tuple[Tensor, Metadata]:
    loaded = _load_local_list(files)
    return _half_xy(*_concat(loaded, *_FINAL_SIZE))


def _process_labels(dataset: torch.Tensor, metadata: Metadata, labels: list[Label|None])->Tensor:
    cuts = dataset.clone()

    for label, i in zip(labels, range(len(dataset))):
        if label is not None:
            angle = label.angle
            point_1 = label.point_1
            point_2 = label.point_2
            if angle is not None and point_1 is not None and point_2 is not None:
                cuts[i] = _cut_volume(dataset[i], metadata, angle*torch.pi/180, point_1, point_2)
    
    return cuts

def _load_labelled_data(files: list[str], label_file: Path)->tuple[Tensor, Metadata]:
    dataset, metadata = _load_and_reshape(files)
    labels = torch.load(label_file)
    return _process_labels(dataset, metadata, labels), metadata


def load_all()->tuple[Tensor, Metadata]:
    '''Load all the data. Duh'''

    all_data_files = [
        (_FILES_10_17_2024, _labels_10_17_2024),
        (_FILES_Oct_28, _labels_Oct_28),
        (_FILES_Nov_26, _labels_Nov_26)
    ]

    all_data_and_metadata = [ _load_labelled_data(i, j) for i, j in all_data_files]

    all_data = [i[0] for i in all_data_and_metadata]

    if not all(i[1] == all_data_and_metadata[0][1] for i in all_data_and_metadata):
        raise RuntimeError('Metadata mismatch')

    return torch.cat(all_data, 0), all_data_and_metadata[0][1]
