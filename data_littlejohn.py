import tifffile
import torch

import data.trichomes_littlejohn
import volumetric
import montage
import device
from localisation_data import fwhm_to_sigma

load = data.trichomes_littlejohn.load_all

def _mainfunc()->None:
    stack, metadata  = load()
    print(metadata)
    print(stack.shape)
    dataset_sum = volumetric.SimpleVolumetric3Plane(stack*1.0, metadata, device.device)
    dataset_max = volumetric.MaxVolumetric3Plane(stack*1.0, metadata, device.device)

    dataset_sum.set_sigma(fwhm_to_sigma(1))
    dataset_max.set_sigma(fwhm_to_sigma(1))

    m_sum = montage.make_stack_images([ dataset_sum[i] for i in range(len(dataset_sum)) ])
    m_max = montage.make_stack_images([ dataset_max[i] for i in range(len(dataset_max)) ])
    tifffile.imwrite('hax/trichomes_sum.tiff', (torch.stack(m_sum, 0).permute(0,2,3,1)*255).char().numpy())
    tifffile.imwrite('hax/trichomes_max.tiff', (torch.stack(m_max, 0).permute(0,2,3,1)*255).char().numpy())
    
    tifffile.imwrite('hax/trichomes_cubes.tiff', stack.numpy(), imagej=True, resolution=(1/metadata.xy_nm_pix, 1/metadata.xy_nm_pix),  metadata={'axes': 'TZYX', "unit":"nm", 'spacing':metadata.z_nm_pix})


if __name__ == "__main__":
    _mainfunc()
