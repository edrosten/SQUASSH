import torch
import tifffile
import tqdm

import data.dan_microtubules
import montage
import git
import device



load_3d = data.dan_microtubules.load
load_3d_2 = data.dan_microtubules.load2
load_3d_3 = data.dan_microtubules.load_3

def _render_cube(centres: torch.Tensor, sigma_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel: float, size:int)->torch.Tensor:
    assert centres.ndim == 3
    assert centres.shape[2] == 3

    assert sigma_nm.ndim == 1
    assert sigma_nm.shape[0] == centres.shape[0]
    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]

    device = centres.device
    batch = centres.shape[0]
    n_spots = centres.shape[1]
    sigma_px = sigma_nm / nm_per_pixel

    Z: torch.Tensor = (1.0 / (2*torch.pi*sigma_px**2)).reshape(batch, 1, 1, 1).expand(batch, size, size, size)
    sigma_px = sigma_px.reshape(batch, 1, 1, 1).expand(batch, size, size, size)


    # Pixel positions (centered)
    # Make p a 2D grid of (x,y) pairs
    px = torch.arange( -(size-1)/2, 1+(size-1)/2, device=device, dtype=centres.dtype).unsqueeze(0).unsqueeze(0).expand(size,size,size)
    py = px.permute(0, 2, 1)
    pz = px.permute(2, 1, 0)
    p = torch.stack((px, py, pz), 3)


    # P is batch of lists of 2D grids of (x,y z) triples, i.e. 5D
    p = p.unsqueeze(0).unsqueeze(0).expand(batch, n_spots, size, size, size, 3)


    # Centres is a batch of lists of x, y pairs
    # Make it a batch of lists of grids of x,y pairs
    centres = centres.unsqueeze(2).unsqueeze(2).unsqueeze(2).expand(batch, n_spots, size, size, size, 3) / nm_per_pixel

    weights = weights.reshape(batch, n_spots, 1, 1, 1).expand(batch, n_spots, size, size, size)

    return (torch.exp(-((p - centres)**2).sum(5) / (2*sigma_px**2)) * weights).sum(1) * Z

if __name__ == "__main__":
    

    all_points = load_3d_3()

    for name, points in all_points.items():
        if points:

            e = torch.ones(1).half().to(device.device)
            
            nm_per_pix = 2.0

            list_of_cubes = [ _render_cube(p.unsqueeze(0).to(e), torch.tensor([2.0]).to(e), torch.ones(1, p.shape[0]).to(e), nm_per_pix, 64) for p in tqdm.tqdm(points)]
            list_of_cubes = [ i/i.max() * 255.99 for i in list_of_cubes]
    
            stack_of_cubes = torch.stack(list_of_cubes, 0).squeeze(1).float().cpu().to(torch.uint8)
            
            tifffile.imwrite(f'hax/dan-cube-{git.shorthash}-params2-{name[:-5]}.tiff', stack_of_cubes.numpy(), imagej=True,metadata={'axes':'TZYX', 'unit':"nm", 'spacing': nm_per_pix},compression='zlib', compressionargs={'level':9}, resolution=(1/nm_per_pix,1/nm_per_pix))

            stack = (torch.stack(montage.make_stack_multiple(points, 2., 2.0, 64, 1), 0).permute(0, 2, 3, 1)*255).char().numpy()
            tifffile.imwrite(f'hax/dan-3plane-{git.shorthash}-params2-{name[:-5]}.tiff', stack, compression='zlib', compressionargs={'level':9}, resolution=(1/nm_per_pix,1/nm_per_pix))

