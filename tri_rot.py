import torch
import tifffile
from torch import Tensor
from data.volumetric import Metadata

import data_littlejohn
import generate_data

all_trichomes, metadata = data_littlejohn.load()

def _rotate_cube(cube: Tensor, fmetadata: Metadata, rot:Tensor)->Tensor:
    
    n_cubes = cube.shape[0]
    n_rot = rot.shape[0]
    
    assert cube.ndim == 4
    assert rot.ndim == 3
    assert rot.shape == torch.Size([n_rot, 3, 3])

    assert n_rot % n_cubes  == 0
    n_rot_per_cube = n_rot // n_cubes

    #Image size in nm.
    sx = cube.shape[-1] * fmetadata.xy_nm_pix
    #sy = cube.shape[-2] * fmetadata.xy_nm_pix
    sz = cube.shape[-3] * fmetadata.z_nm_pix

   # Need to scale z (and y but it's the same as x) to match x
    z = sz/sx
    z_scale = torch.tensor([
        [ 1., 0, 0],
        [ 0., 1, 0],
        [ 0., 0, z],
    ])

    iz_scale = torch.tensor([
        [ 1., 0,   0],
        [ 0., 1,   0],
        [ 0., 0, 1/z],
    ])
    # Construct an affine transform matrix
    A = torch.cat([iz_scale @ rot @ z_scale, torch.zeros(rot.shape[0], 3,1)], -1)
    grid = torch.nn.functional.affine_grid(A, [rot.shape[0],1,*cube.shape[1:]])
    
    cube = cube.unsqueeze(1).repeat_interleave(n_rot_per_cube, 0)
    print(cube.shape)
    resampled = torch.nn.functional.grid_sample(cube, grid, align_corners=False).squeeze(1)
    
    return resampled




generate_data.random_rotation_matrix(1)
r = generate_data.random_rotation_matrix(30)
new_cuuube = _rotate_cube(all_trichomes[2:5], metadata, r)
augmented = torch.cat([all_trichomes[2].unsqueeze(0), new_cuuube], 0)


tifffile.imwrite('hax/augg.tiff', augmented.numpy(), imagej=True, resolution=(1/metadata.xy_nm_pix, 1/metadata.xy_nm_pix),  metadata={'axes': 'TZYX', "unit":"nm", 'spacing':metadata.z_nm_pix})


