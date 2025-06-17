'''Functions for dealing with volumetric data'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

import torch
import tqdm
from torch import Tensor

def _fraction_index_symmetric(data: Tensor, fraction: float)->Tensor:
    # symmatrize
    # cs is complement symmetric, in that 1-cs is equal to cs.flip()
    # except that the cumulative sum is incomplete in that that it goes to
    # 1 (all the elements) but not 0 (none). The relation holds only if 
    # cs is symmetric in that it either has a leading zero or the leading 1 is
    # removed.
    assert data.ndim==1
    data = data + data.flip(0)
    data /= data.sum()
    cs = data.cumsum(0)
    index_low = (cs >= fraction).nonzero()[0]
    assert data[0:index_low].sum() <=  fraction
    assert data[0:index_low+1].sum() >= fraction
    assert (data[0:index_low].sum() - data[data.numel()-index_low:].sum()).abs() < 1e-6
    return index_low


def find_projection_planes_px(vol_sum: Tensor, percentile:float)->Tuple[float, float]:
    '''Take in an averaged (or summed) volume for all the data
    and finds the XY and Z plane positions for Dan6 rendering

    Positions are in pixels'''

    assert vol_sum.shape[1] == vol_sum.shape[2]
    xysize = vol_sum.shape[1]
    z_size = vol_sum.shape[0]


    #xs = torch.arange(0, vol_sum.shape[2]).reshape(1,1,-1).expand(vol_sum.shape)
    #ys = torch.arange(0, vol_sum.shape[1]).reshape(1,-1,1).expand(vol_sum.shape)
    #zs = torch.arange(0, vol_sum.shape[0]).reshape(-1,1,1).expand(vol_sum.shape)

    #coords = torch.stack([zs, ys, xs], 3)
    #centre = (vol_sum.unsqueeze(3).expand(-1, -1, -1, 3)* coords).sum((0,1,2))/vol_sum.sum()

    # Check that the data are centred roughly.
    # Do we need this?
    #print(centre)
    #print(centre - (torch.tensor(vol_sum.shape)-1)/2)
    #assert ((centre - (torch.tensor(vol_sum.shape)-1)/2).abs() < .5).all()


    # The centring should bring it on average to the middle of the image 
    # so for an image size 64, the centre should be at 31.5
    #
    # The rendering algorithm maps centres to pixels using
    #  c = (size-1)/2
    #  pos_list = (arange(0,size)-c)*nm_per_pix
    #
    #  So for size = 64
    #  c = (64-1)/2 = 63/2 = 31.5
    #  pos_list = [0,64)-c = [0,63]-c = [-31.5,31.5]
    #
    #  Points are essentially sampled to the "centre" of the pixel using the mapping
    #  So in this configuration, a point at ((31.5,-31.5) with r->0 will light up
    #  exactly one pixel.
    # 
    #  Combined with the centring above, the final centre should be 0nm which falls
    #  exactly between two pixels since it is of even size.

    # Do the inverse for this

    # Probably best to put the plane rendering boundary on the edge of a pixel, 
    # so it doesn't bisect a pixel. However the difference is likely so slight
    # that it won't affect the final result since it'll affect only a tiny slice
    # of pixels centred on the 90th percentile


    # Assume after the centering
    x_projection = vol_sum.sum((0,1))
    y_projection = vol_sum.sum((0,2))
    z_projection = vol_sum.sum((1,2))




    # This captures >= 90%
    # Put the plane half a pixel off in order to not split a pixel
    xy_index_low = _fraction_index_symmetric(x_projection+y_projection, 1-percentile/100)
    xy_plane_pos = (xy_index_low - (xysize-1)/2 -.5).abs().item()


    z_index_low = _fraction_index_symmetric(z_projection, 1-percentile/100)
    z_plane_pos = (z_index_low - (z_size-1)/2 -.5).abs().item()


    print("positions: ", xy_plane_pos, z_plane_pos)

    return xy_plane_pos, z_plane_pos


def _cap_01(z: torch.Tensor)->torch.Tensor:
    '''Limit input to between 0, 1'''
    return torch.minimum(torch.ones_like(z), torch.maximum(torch.zeros_like(z), z))

def _assignment(pos: torch.Tensor, plane: float)->torch.Tensor:
    '''Assignment to plane'''
    # Assumption is there are planes at +- plane, 
    # and weight is assigned linearly between them and 
    # constant afterwards
    #
    # 1           _______
    #            /┊
    #           / ┊
    #          /┊ ┊
    # 0 ______/ ┊ ┊
    #         ┊ ┊ ┊         
    #   -plane╯ ┊ ┊
    #         0 ╯ ┊
    #       +plane╯
    #
    #   plane/(2*plane) + .5 = 1
    #  -plane/(2*plane) + .5 = 0
    return _cap_01(pos * .5 / plane + .5)


# Consider the following
#  8 pixels.
#  nm/pix = 1
# index = 2
# plane = +-2.0
# nm/pix = 1
#                   !<--                 -->!
# Pixel:|   0 |   1 !  2  |  3  |  4  |  5  !  6  |  7  
# NM:   | -3.5| -2.5!-1.5 | -.5 | .5  | 1.5 ! 2.5 | 3.5 
#Assign |   0 |   0 !  .2 | .4  | .6  | .8  !  1  |  1
#                   |***
# Distances to      |********
# the plane         |**************
#                   |********************
#
# The distances to the plane are 1/4/2, 1/4 + 1/4/2, 2/4+1/4/2
# etc, i.e. 1/8., 3/8, 5/8, 7/8
# Check this:
# shift to center is (size-1)/2
assert (_assignment(torch.arange(8)-(8-1)/2, 2.0) == torch.tensor([0, 0, 1/8, 3/8, 5/8, 7/8, 1, 1])).all()




def project_volume_dan6(s: Tensor, xy_plane_pos_px: float, z_plane_pos_px: float, shift: torch.Tensor)->Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    Perform a 6 plane projection of a volume. Note that the 
    centre about which the planes sit is provided

    Note that 

    Keyword arguments
    s -- the volume
    shift -- shift to the (i.e. zero position) of the volume
    xy_plane_pos_px -- position of the plane in pixel coordinates
    z_plane_pos_px -- position of the plane in pixel coordinates

    Note that of everything is centred then the shift is:
    (dimension_size-1)/2, 
    which makes the coordinated symmetric around the centre
    
    additional shifts can be added on
    '''

    # Coordinates for x, y, z pixels
    z_px = torch.arange(0, s.shape[0], device=s.device) - shift[0]
    y_px = torch.arange(0, s.shape[1], device=s.device) - shift[1]
    x_px = torch.arange(0, s.shape[2], device=s.device) - shift[2]
    
    xy = s.sum(0)

    xy_0 = (_assignment(z_px, z_plane_pos_px).unsqueeze(1).unsqueeze(2).expand_as(s) * s).sum(0)
    xy_1 = ((1-_assignment(z_px, z_plane_pos_px)).unsqueeze(1).unsqueeze(2).expand_as(s) * s).sum(0)

    xz = s.sum(1)
    xz_0 = (_assignment(y_px, xy_plane_pos_px).unsqueeze(0).unsqueeze(2).expand_as(s) * s).sum(1)
    xz_1 = ((1-_assignment(y_px, xy_plane_pos_px)).unsqueeze(0).unsqueeze(2).expand_as(s) * s).sum(1)
    
    yz = s.sum(2)
    yz_0 = (_assignment(x_px, xy_plane_pos_px).unsqueeze(0).unsqueeze(1).expand_as(s) * s).sum(2)
    yz_1 = ((1-_assignment(x_px, xy_plane_pos_px)).unsqueeze(0).unsqueeze(1).expand_as(s) * s).sum(2)
    
    assert( (xy - (xy_0 + xy_1)).abs().max() /s.max() < 1e-5)
    assert( (xz - (xz_0 + xz_1)).abs().max() /s.max() < 1e-5)
    assert( (yz - (yz_0 + yz_1)).abs().max() /s.max() < 1e-5)
    
    return xy_0, xy_1, yz_0, yz_1, xz_0, xz_1




def compute_volumetric_6_plane(stax:torch.Tensor, device: torch.device|None)->Tuple[List[List[Tensor]], float, float]:
    '''
    Given an input stack, compute the 6 plane projection, and retrn both that and 
    the plane positions (IN PIXELS!!!)
    '''

    #Find the brightness centroid and reslice
    # Ordering is z,y,x 
    
    assert stax.ndim == 4
    assert stax.shape[2] == stax.shape[3]

    #Calculate the average of the cropped volumes, in order to find
    # projection planes
    vol_sum = stax.sum(0).to(torch.float32)

    _PERCENTILE = 70
    xy_plane_pos_px, z_plane_pos_px = find_projection_planes_px(vol_sum, _PERCENTILE)
    
    # plane pos is in pixel coordinates. This is based on the assumption that the
    # image coordinares are  not [0-N), but centred around zero.
    
    print(stax.shape)
    print("xy_plane_pos_px: ", xy_plane_pos_px)
    print("z_plane_pos_px: ", z_plane_pos_px)

    proj_dan6_uncropped: List[List[torch.Tensor]] = []

    for s in tqdm.tqdm(stax, '6-plane projection'):
        s=s.to(device)
        shift = (torch.tensor(stax.shape[1:]) - 1)/2
        proj_dan6_uncropped.append(list(project_volume_dan6(s, xy_plane_pos_px, z_plane_pos_px, shift)))

    return proj_dan6_uncropped, xy_plane_pos_px, z_plane_pos_px


@dataclass(frozen=True)
class Metadata:
    '''Metadata for loaded images'''
    xy_nm_pix: float
    z_nm_pix: float
    xy_fwhm_nm: float
    z_fwhm_nm: float
