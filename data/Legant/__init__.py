import zipfile
from pathlib import Path
from typing import Tuple

import tqdm
import tifffile
import torch
import numpy as np

from ..volumetric import Metadata 
# From this publication
# https://www.nature.com/articles/s41592-023-02126-0

_PIXEL_SIZE_XY=108
_PIXEL_SIZE_Z=200
_FWHM_XY=200
_FWHM_Z=600 #??


class _ZipFileDataset(torch.utils.data.Dataset):
    def __init__(self, colour: str)->None:
        super().__init__()

        self._zip_path = Path(__file__).parent/'DataForSusan.zip'
        zf = zipfile.Path(self._zip_path) / 'DataForSusan'

        self._file_list = [i.filename.relative_to(self._zip_path) for i in zf.iterdir() if colour in i.name] # type: ignore[attr-defined]

    def __len__(self)->int:
        return len(self._file_list)

    def __getitem__(self, idx:int )->torch.Tensor:
        with (zipfile.Path(self._zip_path)/self._file_list[idx]).open('rb') as data:
            image = tifffile.tifffile.imread(data)
            return torch.tensor(image.astype(np.float32))

def _centroid(s: torch.Tensor)->torch.Tensor:
    zs = torch.arange(s.shape[0], device=s.device).reshape(-1,  1,  1).expand_as(s)
    ys = torch.arange(s.shape[1], device=s.device).reshape( 1, -1,  1).expand_as(s)
    xs = torch.arange(s.shape[2], device=s.device).reshape( 1,  1, -1).expand_as(s)

    indices = torch.stack((zs, ys, xs), 3)
    return (indices*s.unsqueeze(-1).expand(-1,-1,-1,3)).sum((0, 1, 2)) / s.sum()


def _load_colour(colour: str)->list[torch.Tensor]:
    zf = _ZipFileDataset(colour)
    loader = torch.utils.data.DataLoader(zf, batch_size=1, shuffle=False, num_workers=6)
    return [ d.squeeze(0) for d in  tqdm.tqdm(loader, desc="Unzipping")]

def _centroid_stack(stacks: list[torch.Tensor])->torch.Tensor:
    return torch.stack([ _centroid(i) for i in tqdm.tqdm(stacks, desc="Centroiding")], 0)
    
def _load_and_crop_extra(colour: str)->Tuple[torch.Tensor, Metadata, torch.Tensor, Metadata]:
    stacks =  _load_colour(colour)
    centroids = _centroid_stack(stacks)
    return _crop_extra(stacks, centroids)

def _crop_extra(stacks: list[torch.Tensor], centroids: torch.Tensor)->Tuple[torch.Tensor, Metadata, torch.Tensor, Metadata]:
    big=[]
    bigger=[]

    # Sizes to crop out
    S=128
    Sz=64
    # Result is a 32x32x32 cube
    downsample_Z=4
    downsample_XY=8

    downsample = (downsample_Z, downsample_XY, downsample_XY)

    for stack, c in zip(stacks,tqdm.tqdm(centroids, 'Cropping')):

        # Paste into a big ass-image, and then crop that, since sometimes the necessary crop
        # needs to hang over the edge. Images are low background so this hack works well enough
        N=512
        z = torch.zeros(N, N, N)
        s = stack.shape
        
        #Align the centre with the centroid
        slices = []
        for si, ci in zip(s, c.floor().to(torch.int32)):
            start = N//2-ci
            end = start + si
            slices.append(slice(start, end))

        z[*slices] = stack

        zc = z[N//2-Sz:N//2+Sz,N//2-S:N//2+S,N//2-S:N//2+S] 
        zc /= zc.sum()
        bigger.append(zc)
        big.append(torch.nn.functional.avg_pool3d(zc.unsqueeze(0), downsample).squeeze(0)) # pylint: disable=not-callable
    

    # Approximate ideal FWHM for arbitrary pixel sizes.
    # Note that due to the heavy downsampling since the 
    # original data is not heavily oversampled, the effect
    # of the imaging PSF is essentially lost by this point
    fwhm_ratio = 5/8

    m = Metadata(
        xy_nm_pix = _PIXEL_SIZE_XY * downsample_XY,
        xy_fwhm_nm = _PIXEL_SIZE_XY * downsample_XY * fwhm_ratio,
        z_nm_pix = _PIXEL_SIZE_Z * downsample_Z,
        z_fwhm_nm = _PIXEL_SIZE_Z * downsample_Z * fwhm_ratio
    )

    mb = Metadata(
        xy_nm_pix = _PIXEL_SIZE_XY,
        xy_fwhm_nm = _FWHM_XY,
        z_nm_pix = _PIXEL_SIZE_Z,
        z_fwhm_nm = _FWHM_Z
    )


    return torch.stack(big, 0), m, torch.stack(bigger,0), mb


def load_and_crop()->Tuple[torch.Tensor, Metadata]:
    '''Load the 560nm data'''
    return _load_and_crop_extra("560nm")[0:2]

def load_and_crop_tubulin()->Tuple[torch.Tensor, Metadata]:
    '''Load the 642nm data'''
    return _load_and_crop_extra("642nm")[0:2]

def load_and_crop_both()->Tuple[torch.Tensor, torch.Tensor, Metadata, torch.Tensor, torch.Tensor, Metadata]:
    '''Load the 560nm and 642nm data jointly aligned to the 560 data'''
    data_560 = _load_colour("560nm") 
    data_642 = _load_colour("642nm") 

    centroid_560 = _centroid_stack(data_560)

    cropped_560, metadata, cropped_560_large, metadata_large = _crop_extra(data_560, centroid_560)
    cropped_642, _ , cropped_642_large, _ = _crop_extra(data_642, centroid_560)

    return cropped_560, cropped_642, metadata, cropped_560_large, cropped_642_large, metadata_large
