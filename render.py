from typing import Tuple, List
import torch



def render_3d(centres: torch.Tensor, sigma_nm_xy_z: torch.Tensor, weights: torch.Tensor, nm_per_pixel_xy: float, nm_per_pixel_z: float, size_xy:int, size_z: int)->torch.Tensor:
    '''
    Render a list of spots. Passed in as:
    0 = batch
    1 = index
    2 = [x, y, z]

    Sigma
    0 = batch
    1 = 0: xy 1: z

    Weights
    0 = batch
    1 = index

    nm_per_pixel


    This function respects the dtype of `centres`.
    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 3
    assert sigma_nm_xy_z.ndim == 2
    assert sigma_nm_xy_z.shape[0] == centres.shape[0]
    assert sigma_nm_xy_z.shape[1] == 2
    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]

    device = centres.device
    batch = centres.shape[0]
    n_spots = centres.shape[1]
    sigma_px_xy = sigma_nm_xy_z[:,0] / nm_per_pixel_xy
    sigma_px_z = sigma_nm_xy_z[:,1] / nm_per_pixel_z

    Z3_scale = 1/((2*torch.pi)**(3/2))

    size_zyx = torch.Size((size_z, size_xy, size_xy))
    size_bnzyx = torch.Size((batch, n_spots, *size_zyx))

    Z: torch.Tensor = (Z3_scale/(sigma_px_xy**2 * sigma_px_z)).reshape(batch, 1, 1, 1).expand(batch, *size_zyx)

    sigma_px_xyz = torch.stack((sigma_px_xy, sigma_px_xy, sigma_px_z), 1)
    sigma_px_xyz = sigma_px_xyz.reshape(batch, 1, 1, 1, 1, 3).expand(*size_bnzyx, 3)


    # Pixel positions (centered)
    # Make p a 2D grid of (x,y) pairs
    px = torch.arange( -(size_xy-1)/2, 1+(size_xy-1)/2, device=device, dtype=centres.dtype).reshape(1,1,size_xy).expand(size_zyx)
    py = px.transpose(1,2)
    pz = torch.arange(-(size_z-1)/2, 1+(size_z-1)/2, device=device, dtype=centres.dtype).reshape(size_z,1,1).expand(size_zyx)

    p = torch.stack((px, py, pz), 3).unsqueeze(0).unsqueeze(0).expand(*size_bnzyx, 3)

    # Centres is a batch of lists of x, y pairs
    # Make it a batch of lists of grids of x,y pairs
    nm_per_pixel = torch.tensor([nm_per_pixel_xy, nm_per_pixel_xy, nm_per_pixel_z], dtype=centres.dtype, device=centres.device)

    centres_px = centres / nm_per_pixel.reshape(1, 1, 3).expand(batch, n_spots, 3)
    centres_px = centres_px.reshape(batch, n_spots, 1, 1, 1, 3).expand(*size_bnzyx, 3)

    weights = weights.reshape(batch, n_spots, 1, 1, 1).expand(size_bnzyx)
    e = torch.exp(-((p - centres_px)**2 / (2*sigma_px_xyz**2)).sum(5))
    return (e * weights).sum(1)*Z


def render_batch_weights(centres: torch.Tensor, sigma_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel: float, size:int)->torch.Tensor:
    '''
    Render a list of spots. Passed in as:
    0 = batch
    1 = index
    2 = [x, y]

    Sigma
    0 = batch

    This function respects the dtype of `centres`.
    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 2

    assert sigma_nm.ndim == 1
    assert sigma_nm.shape[0] == centres.shape[0]
    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]

    device = centres.device
    batch = centres.shape[0]
    n_spots = centres.shape[1]
    sigma_px = sigma_nm / nm_per_pixel

    Z: torch.Tensor = (1.0 / (2*torch.pi*sigma_px**2)).unsqueeze(1).unsqueeze(1).expand(batch, size, size)
    sigma_px = sigma_px.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch, n_spots, size, size)


    # Pixel positions (centered)
    # Make p a 2D grid of (x,y) pairs
    px = torch.arange( -(size-1)/2, 1+(size-1)/2, device=device, dtype=centres.dtype).unsqueeze(0).expand(size,size)
    py = px.transpose(0,1)
    p = torch.stack((px, py), 2)


    # P is batch of lists of 2D grids of (x,y) pairs, i.e. 5D
    p = p.unsqueeze(0).unsqueeze(0).expand(batch, n_spots, size, size, 2)


    # Centres is a batch of lists of x, y pairs
    # Make it a batch of lists of grids of x,y pairs
    centres = centres.unsqueeze(2).unsqueeze(2).expand(batch, n_spots, size, size, 2) / nm_per_pixel

    weights = weights.reshape(batch, n_spots, 1, 1).expand(batch, n_spots, size, size)

    return (torch.exp(-((p - centres)**2).sum(4) / (2*sigma_px**2)) * weights).sum(1) * Z



def render_batch_anisotropic_with_sigmas_2D(centres: torch.Tensor, sigma_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel_x: float, nm_per_pixel_y: float, size:Tuple[int,int])->torch.Tensor:
    '''
    Render 2D with per-spot, per-axis uncertainty.
      
    centres passed in as:
    0 = batch
    1 = index
    2 = [x, y]

    Sigma
    0 = batch
    1 = index
    2 = axis (x, y)

    weights
    0 = batch
    1 = index

    This function respects the dtype of `centres`.
    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 2
    assert sigma_nm.shape == centres.shape
    assert weights.shape == centres.shape[0:2]

    device = centres.device
    dtype = centres.dtype
    batch = centres.shape[0]
    n_spots = centres.shape[1]
    rows, cols = size
    
    nm_per_pixel = torch.tensor([nm_per_pixel_x, nm_per_pixel_y], device=device, dtype=dtype)

    sigma_px = sigma_nm / nm_per_pixel.reshape(1,1,2).expand_as(sigma_nm)
        
    Z: torch.Tensor = (1.0 / (2*torch.pi*sigma_px.prod(2))).reshape(batch, n_spots, 1, 1).expand(batch, n_spots, rows, cols)
    sigma_px = sigma_px.reshape(batch, n_spots, 1, 1, 2).expand(batch, n_spots, rows, cols, 2)


    # Pixel positions (centered)
    # Make p a 2D grid of (x,y) pairs
    px = torch.arange( -(cols-1)/2, 1+(cols-1)/2, device=device, dtype=dtype).unsqueeze(0).expand(rows,cols)
    py = torch.arange( -(rows-1)/2, 1+(rows-1)/2, device=device, dtype=dtype).unsqueeze(1).expand(rows,cols)

    p = torch.stack((px, py), 2)


    # P is batch of lists of 2D grids of (x,y) pairs, i.e. 5D
    p = p.unsqueeze(0).unsqueeze(0).expand(batch, n_spots, rows, cols, 2)


    # Centres is a batch of lists of x, y pairs
    # Make it a batch of lists of grids of x,y pairs in pixel space

    centers_px = (centres / nm_per_pixel.reshape(1, 1, 2).expand(batch, n_spots, 2)).reshape(batch, n_spots, 1, 1, 2).expand(batch, n_spots, rows, cols, 2)

    weights = weights.reshape(batch, n_spots, 1, 1).expand(batch, n_spots, rows, cols) * Z


    inner = ((p-centers_px)**2 / (2*sigma_px**2)).sum(4)

    return (torch.exp(-inner) * weights).sum(1)




def render_batch_anisotropic(centres: torch.Tensor, sigma_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel_x: float, nm_per_pixel_y: float, size:Tuple[int,int])->torch.Tensor:
    '''
    Render a list of spots. Passed in as:
    0 = batch
    1 = index
    2 = [x, y]

    Sigma per batch or per spot
    0 = batch
    1 = axis
    or
    0 = batch
    1 = spot
    2 = axis


    NM per pixel. Note using it as a tensor, so changes do not trigger a recompile
    0 = axis

    size:
    (rows, cols)

    This function respects the dtype of `centres`.
    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 2
    
    if sigma_nm.ndim == 2:
        assert sigma_nm.ndim == 2
        assert sigma_nm.shape[0] == centres.shape[0]
        assert sigma_nm.shape[1] == 2
        sigma_nm_per_spot = sigma_nm.unsqueeze(1).expand_as(centres)
    else:
        assert centres.shape == sigma_nm.shape
        sigma_nm_per_spot = sigma_nm

    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]

    return render_batch_anisotropic_with_sigmas_2D(centres, sigma_nm_per_spot, weights, nm_per_pixel_x, nm_per_pixel_y, size)


def render_multiple(centres: torch.Tensor, sigma_xy_z_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel_xy: float, nm_per_pixel_z: float, xy_size: int, z_size: int)->List[torch.Tensor]:
    '''
    Render a list of spots. Passed in as:
    0 = batch
    1 = index
    2 = [x, y, z]

    Sigma
    0 = batch
    1 = axis, as in sigma for x/y axes and for z axis
    OR
    0 = batch
    1 = index
    2 = axis as in sigma for x/y axes and for z axis

    NM per pixel for x/y axis and z axis
    Size for x/y axis and z axis

    This function respects the dtype of `centres`.


    Returns:
    renderings from multiple views.

    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 3

    if sigma_xy_z_nm.ndim == 2:
        assert sigma_xy_z_nm.ndim == 2
        assert sigma_xy_z_nm.shape[0] == centres.shape[0]
        assert sigma_xy_z_nm.shape[1] == 2
    else:
        assert sigma_xy_z_nm.shape == torch.Size([*centres.shape[0:2],2])

    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]
    
    sigma_xy = sigma_xy_z_nm[...,0]
    sigma_xy = sigma_xy.unsqueeze(-1).expand(*sigma_xy.shape, 2)

    r_xy = render_batch_anisotropic(centres[:,:,0:2], sigma_xy, weights, nm_per_pixel_xy, nm_per_pixel_xy, (xy_size, xy_size))
    
    # YZ -> xy [cols, rows]
    # Z maps to rows
    r_yz = render_batch_anisotropic(centres[:,:,1:3], sigma_xy_z_nm, weights, nm_per_pixel_xy, nm_per_pixel_z, (z_size, xy_size))

    r_xz = render_batch_anisotropic(centres[:,:,0:3:2], sigma_xy_z_nm, weights, nm_per_pixel_xy, nm_per_pixel_z, (z_size, xy_size))

    return [r_xy, r_yz, r_xz]


def _cap_01(z: torch.Tensor)->torch.Tensor:
    '''Limit input to between 0, 1'''
    return torch.minimum(torch.ones_like(z), torch.maximum(torch.zeros_like(z), z))


def render_multiple_dan6(centres: torch.Tensor, sigma_xy_z_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel_xy: float, nm_per_pixel_z: float, xy_size: int, z_size: int, max_abs_xy: float, max_abs_z: float)->List[torch.Tensor]:
    '''
    Render a list of spots. Passed in as:
    0 = batch
    1 = index
    2 = [x, y, z]

    Sigma
    0 = batch
    1 = axis, as in sigma for x/y axes and for z axis
    OR
    0 = batch
    1 = index
    2 = axis as in sigma for x/y axes and for z axis

    NM per pixel for x/y axis and z axis
    Size for x/y axis and z axis

    This function respects the dtype of `centres`.

    max_abs_xy, max_abs_z: the expected size in the xy and z directions, with the planes at +/- max_abs


    Returns:
    renderings from multiple views.

    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 3

    if sigma_xy_z_nm.ndim == 2:
        assert sigma_xy_z_nm.ndim == 2
        assert sigma_xy_z_nm.shape[0] == centres.shape[0]
        assert sigma_xy_z_nm.shape[1] == 2
    else:
        assert sigma_xy_z_nm.shape == torch.Size([*centres.shape[0:2],2])

    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]
    
    sigma_xy = sigma_xy_z_nm[...,0]
    sigma_xy = sigma_xy.unsqueeze(-1).expand(*sigma_xy.shape, 2)
    
    # Compute weightings based on the distance between the planes, with the + plane being 1
    # and the - plane being 0. Weights are capped to be between 0 and 1
    weights_z = _cap_01(centres[:,:,2]*.5/max_abs_z + .5)

    r_xy_0 = render_batch_anisotropic(centres[:,:,0:2], sigma_xy, weights * weights_z, nm_per_pixel_xy, nm_per_pixel_xy, (xy_size, xy_size))
    r_xy_1 = render_batch_anisotropic(centres[:,:,0:2], sigma_xy, weights * (1-weights_z), nm_per_pixel_xy, nm_per_pixel_xy, (xy_size, xy_size))
    
    # YZ -> xy [cols, rows]
    # Z maps to rows
    weights_x = _cap_01(centres[:,:,0]*.5/max_abs_xy + .5)
    r_yz_0 = render_batch_anisotropic(centres[:,:,1:3], sigma_xy_z_nm, weights * weights_x, nm_per_pixel_xy, nm_per_pixel_z, (z_size, xy_size))
    r_yz_1 = render_batch_anisotropic(centres[:,:,1:3], sigma_xy_z_nm, weights * (1-weights_x), nm_per_pixel_xy, nm_per_pixel_z, (z_size, xy_size))

    weights_y = _cap_01(centres[:,:,1]*.5/max_abs_xy + .5)
    r_xz_0 = render_batch_anisotropic(centres[:,:,0:3:2], sigma_xy_z_nm, weights * weights_y, nm_per_pixel_xy, nm_per_pixel_z, (z_size, xy_size))
    r_xz_1 = render_batch_anisotropic(centres[:,:,0:3:2], sigma_xy_z_nm, weights * (1-weights_y), nm_per_pixel_xy, nm_per_pixel_z, (z_size, xy_size))

    return [r_xy_0, r_xy_1, r_yz_0, r_yz_1, r_xz_0, r_xz_1]


def render_multiple_scale(centres: torch.Tensor, sigma_xy_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel_xy: float, z_scale: float, xy_size: int, z_size: int)->List[torch.Tensor]:
    ''' 
    See render_multiple 
    This is in terms of x/y only and a scale for z

    '''
    assert centres.ndim == 3
    assert centres.shape[2] == 3

    assert sigma_xy_nm.ndim == 1
    assert sigma_xy_nm.shape[0] == centres.shape[0]

    assert weights.ndim == 2
    assert weights.shape[0] == centres.shape[0]
    assert weights.shape[1] == centres.shape[1]
    
    return render_multiple(centres, torch.stack([sigma_xy_nm, sigma_xy_nm*z_scale], -1), weights, nm_per_pixel_xy, nm_per_pixel_xy*z_scale, xy_size, z_size)





