from typing import Tuple, Any
import torch
from matrix import trn
from render import render_3d, render_batch_anisotropic_with_sigmas_2D, render_batch_weights, render_batch_anisotropic, render_multiple_scale 

def _render_spot_slow(centres: torch.Tensor, sigma_nm: torch.Tensor, nm_per_pixel: float, size:int)->torch.Tensor:
    ''' Slowest reference implementation of rendering'''
    batch = centres.shape[0]

    c = (size-1)/2
    result = torch.zeros((batch,size,size))


    for b in range(centres.shape[0]):

        sigma_px = sigma_nm[b]/nm_per_pixel
        Z = 1.0 / (2*torch.pi*sigma_px**2)

        for x in range(size):
            for y in range(size):
                for s in range(centres.shape[1]):
                    cx = centres[b][s][0]
                    cy = centres[b][s][1]
                    xx = (x - c) * nm_per_pixel
                    yy = (y - c) * nm_per_pixel
                    result[b][y][x] += torch.exp( -((cx-xx)**2 + (cy-yy)**2)/(2*sigma_nm[b]**2)) * Z

    return result



def _render_3D_veryslow(centres: torch.Tensor, weights: torch.Tensor, sigma_xy: torch.Tensor, sigma_z:torch.Tensor, size_xy: int, size_z: int, nm_per_pixel_xy: float, nm_per_pixel_z: float)->Tuple[torch.Tensor, Any]:
    batch = centres.shape[0]

    c_xy = (size_xy-1)/2
    c_z = (size_z-1)/2
    result = torch.zeros((batch,size_z,size_xy,size_xy))
    

    xr = [ (x, (x-c_xy)*nm_per_pixel_xy) for x in range(size_xy) ]
    yr = xr
    zr = [ (z, (z-c_z)*nm_per_pixel_z) for z in range(size_z) ]

    for b in range(centres.shape[0]):

        sigma_xy_px = sigma_xy[b]/nm_per_pixel_xy
        sigma_z_px = sigma_z[b]/nm_per_pixel_z
        Z = 1.0 / (2*torch.pi*sigma_xy_px**2) * 1.0/(2*torch.pi*sigma_z_px**2)**.5

        for x, xx in xr:
            for y, yy in yr:
                for z, zz in zr:
                    for s in range(centres.shape[1]):
                        cx = centres[b][s][0]
                        cy = centres[b][s][1]
                        cz = centres[b][s][2]

                        result[b][z][y][x] += weights[b,s] * torch.exp( -((cx-xx)**2 + (cy-yy)**2)/(2*sigma_xy[b]**2) - (cz-zz)**2/(2*sigma_z[b]**2)) * Z

    return result, ([i for _,i in xr],[i for _,i in xr],[i for _,i in zr])


def _render_spot_loop(centres: torch.Tensor, sigma_nm: torch.Tensor, nm_per_pixel: float, size:int)->torch.Tensor:
    '''
    Render a list of spots. Passed in as:
    0 = batch
    1 = index
    2 = [x, y]

    '''
    device = centres.device
    batch = centres.shape[0]

    # Pixel positions (centered)
    px = torch.arange( -(size-1)/2, 1+(size-1)/2, device=device)

    xs = px.unsqueeze(0).unsqueeze(0).expand(batch, size, size)


    result:torch.Tensor = torch.zeros((batch,size,size), dtype=torch.float, device=device)


    sigma_px = sigma_nm / nm_per_pixel
    Z: torch.Tensor = 1.0 / (2*torch.pi*sigma_px**2).unsqueeze(1).unsqueeze(1).expand(batch, size, size)

    #Iterates over rows, i.e. spots
    for c in range(centres.shape[1]):

        # Expand the current centre to the batch worth of shifted tensors
        cx = centres[:,c,0] / nm_per_pixel
        cy = centres[:,c,1] / nm_per_pixel


        #Get these into the shape batch,x,y
        batch_cx = cx.unsqueeze(1).unsqueeze(2).expand(batch, size, size)
        batch_cy = cy.unsqueeze(1).unsqueeze(2).expand(batch, size, size)

        dx = batch_cx - xs
        dy = batch_cy - trn(xs)


        spot = torch.exp(-(dx**2 + dy**2)/(2*sigma_px**2).unsqueeze(1).unsqueeze(1).expand(batch, size, size))

        result+=spot

    # Why?
    return result * Z


def _render_batch(centres: torch.Tensor, sigma_nm: torch.Tensor, nm_per_pixel: float, size:int)->torch.Tensor:
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

    return torch.exp(-((p - centres)**2).sum(4) / (2*sigma_px**2)).sum(1) * Z



def test_render_1()->None:
    ''' Test the various implementations give the same results'''
    print("test_render_1 ", end="")
    batch = 2
    spots = 20
    sigmas = torch.tensor([10, 20.])

    data = (torch.rand(batch, spots, 2) - 0.5)*150

    r1 = _render_spot_loop(data, sigmas, 10, 32)
    r2 = _render_spot_slow(data, sigmas, 10, 32)
    assert ((r1-r2).abs().max()) < 1e-5
    print("done")


def test_render_slow_vs_3d()->None:
    ''' Compare 3D to 2D renderer'''
    
    print("test_render_slow_vs_3d ", end="")

    batch = 2
    spots = 3
    sigmas = torch.tensor([10, 20.])

    data = (torch.rand(batch, spots, 2) - 0.5)*150
    data = torch.cat((data, torch.zeros(batch, spots, 1)), 2)
    assert data.shape[2] == 3

    r1 = _render_spot_slow(data[:,:,0:2], sigmas, 10, 32)
    r1 = r1 / r1.sum((1,2)).reshape(2, 1, 1).expand_as(r1)  # Normalize

    r2,_ = _render_3D_veryslow(
        centres = data,
        weights = torch.ones(batch, spots),
        sigma_xy = sigmas,
        sigma_z = sigmas*0+1000, # Shouldn't matter
        size_xy = 32,
        size_z = 1,
        nm_per_pixel_xy = 10,
        nm_per_pixel_z = 10) #Shouldn't matter

    r2 = r2 / r2.sum((2,3)).reshape(2, 1, 1, 1).expand_as(r2)  # Normalize
    r2 = r2[:,0,:,:]
    assert ((r1-r2).abs().max()) < 1e-5
    print("done")


def test_render_slow_vs_3d_2()->None:
    ''' Compare 3D to 2D renderer'''
    
    print("test_render_slow_vs_3d_2 ", end="")

    batch = 2
    spots = 3
    sigmas = torch.tensor([10, 20.])

    data = (torch.rand(batch, spots, 3) - 0.5)*150

    r1 = _render_spot_slow(data[:,:,0:2], sigmas, 10, 32)
    r1 = r1 / r1.sum((1,2)).reshape(2, 1, 1).expand_as(r1)  # Normalize

    r2,_ = _render_3D_veryslow(
        centres = data,
        weights = torch.ones(batch,spots),
        sigma_xy = sigmas,
        sigma_z = sigmas*0+1e6, # Shouldn't matter
        size_xy = 32,
        size_z = 1,
        nm_per_pixel_xy = 10,
        nm_per_pixel_z = 10) #Shouldn't matter

    r2 = r2 / r2.sum((2,3)).reshape(2, 1, 1, 1).expand_as(r2)  # Normalize
    r2 = r2[:,0,:,:]
    assert ((r1-r2).abs().max()) < 1e-5
    print("done")

def test_render_slow_vs_3d_3()->None:
    ''' Compare 3D to 2D renderer with weights'''
    
    print("test_render_slow_vs_3d_3 ", end="")

    batch = 2
    spots = 10
    sigmas = torch.tensor([10, 20.])

    data = (torch.rand(batch, spots, 3) - 0.5)*150
    weights = torch.rand(batch, spots)

    r1 = render_batch_weights(data[:,:,0:2], sigmas, weights, 10, 32)
    r1 = r1 / r1.sum((1,2)).reshape(2, 1, 1).expand_as(r1)  # Normalize

    r2,_ = _render_3D_veryslow(
        centres = data,
        weights = weights,
        sigma_xy = sigmas,
        sigma_z = sigmas*0+1e6, # Shouldn't matter
        size_xy = 32,
        size_z = 1,
        nm_per_pixel_xy = 10,
        nm_per_pixel_z = 10) #Shouldn't matter

    r2 = r2 / r2.sum((2,3)).reshape(2, 1, 1, 1).expand_as(r2)  # Normalize
    r2 = r2[:,0,:,:]
    assert ((r1-r2).abs().max()) < 1e-5
    print("done")


def test_render_3d_1()->None:
    ''' Test 3D renderer compared to very slow version'''
    
    print("test_render_3d_1 ", end="")

    batch = 2
    spots = 3
    sigmas = torch.tensor([10, 20.][:batch])

    data = (torch.rand(batch, spots, 3) - 0.5)*200
    weights = torch.rand(batch, spots)

    xy_size = 32
    z_size = 4
    nm_per_pix_xy = 10
    nm_per_pix_z = nm_per_pix_xy * xy_size/z_size

    r1, _ = _render_3D_veryslow(
        centres = data,
        weights = weights,
        sigma_xy = sigmas,
        sigma_z = sigmas * xy_size/z_size,
        size_xy = xy_size,
        size_z = z_size,
        nm_per_pixel_xy = nm_per_pix_xy,
        nm_per_pixel_z = nm_per_pix_z)


    r2 = render_3d(
        centres = data, 
        sigma_nm_xy_z = torch.stack((sigmas, sigmas*xy_size/z_size), 1),
        weights = weights,
        nm_per_pixel_xy = nm_per_pix_xy,
        nm_per_pixel_z = nm_per_pix_z,
        size_xy = xy_size,
        size_z = z_size)

    assert (r1-r2).abs().max() < 1e-5
    print("done")

def draw_render_3d()->None:
    ''' Test 3D renderer compared to very slow version'''
    
    print("draw_render_3d ", end="")

    batch = 4
    spots = 10

    data = ((torch.rand(batch, spots, 3) - 0.5)*200).cuda()
    weights = torch.ones(batch, spots).to(data)
    sigmas = torch.tensor([10, 10., 20, 30][:batch]).to(data)

    xy_size = 32
    z_size = 6
    nm_per_pix_xy = 10
    nm_per_pix_z = nm_per_pix_xy * xy_size/z_size

    r2 = render_3d(
        centres = data, 
        sigma_nm_xy_z = torch.stack((sigmas, sigmas*xy_size/z_size/2), 1),
        weights = weights,
        nm_per_pixel_xy = nm_per_pix_xy,
        nm_per_pixel_z = nm_per_pix_z,
        size_xy = xy_size,
        size_z = z_size).cpu()

    data=data.cpu()
    xr = [ (x-(xy_size-1)/2)*nm_per_pix_xy for x in range(xy_size) ]
    yr=xr
    zr = [ (z-(z_size-1)/2)*nm_per_pix_z for z in range(z_size) ]
    # pylint: disable=wrong-import-position,[import-outside-toplevel]
    from matplotlib.pyplot import subplot, axis, gca, title, scatter, clf, imshow
    clf()
    subplot(batch, z_size+3, 1)
    ext = (xr[0], xr[-1], yr[0], yr[-1])
    for b in range(batch):
        for i in range(z_size):
            subplot(batch, z_size+3, 1+i+(z_size+3)*b)
            imshow(r2[b,i,...], extent=ext)

        subplot(batch, z_size+3, 1+(z_size+3)*b + z_size)
        scatter(data[b, :, 0], data[b, :, 1], c=list(range(spots)))
        axis([xr[0], xr[-1], yr[0], yr[-1]])
        gca().invert_yaxis()
        title('xy')
        subplot(batch, z_size+3, 1+(z_size+3)*b + z_size+1)
        scatter(data[b, :, 0], data[b, :, 2],c=list(range(spots)))
        axis([xr[0], xr[-1], zr[0], zr[-1]])
        gca().invert_yaxis()
        title('xz')
        subplot(batch, z_size+3, 1+(z_size+3)*b + z_size+2)
        scatter(data[b, :, 1], data[b, :, 2],c=list(range(spots)))
        axis([yr[0], yr[-1], zr[0], zr[-1]])
        gca().invert_yaxis()
        title('yz')




def test_per_spot_sigma()->None:
    '''Do a test rendering with variable sigmas. Eyeball it for sanity'''
    # pylint: disable=import-outside-toplevel]
    import astropy.io.fits
    import tifffile 
    centres = torch.tensor([[
        [-25.5, -25.5],
        [-25.5,  24.5],
        [ 24.5,  24.5],
        [ 24.5, -25.5]]])

    sigmas = torch.tensor([[
        [2, 2],
        [2, 4],
        [4, 2],
        [1, 6.]]])


    weights = torch.tensor([[1, 1, 1, 1.]])
    

    # Render them all together and chop them apart and compare to individually rendered ones
    img = render_batch_anisotropic_with_sigmas_2D(centres, sigmas, weights, 1.0, 1.0, (100,100))

    imgs = [
        img[:,0:50,0:50],
        img[:, 50:,0:50],
        img[:, 50:, 50:],
        img[:,0:50, 50:],
    ]
    
    imgs2=[]
    for i in range(4):
        imgs2.append(render_batch_anisotropic(torch.tensor([[[-.5, -.5]]]), sigmas[:, i], torch.ones(1,1), 1.0, 1.0, (50,50)))

    diff = (torch.stack(imgs, 0)-torch.stack(imgs2, 0)).abs().max()
    assert diff < 1e-5

    astropy.io.fits.writeto("hax/rendering.fits", img[0].numpy(), overwrite=True)
    tifffile.imwrite('hax/rendering-all.tiff', torch.stack(imgs, 0).numpy())
    tifffile.imwrite('hax/rendering-all2.tiff', torch.stack(imgs2, 0).numpy())


def test_render()->None:
    ''' Test the various implementations give the same results'''
    batch = 5
    spots = 50

    data3 = (torch.rand(batch, spots, 3) - 0.5)*400
    data = data3[:,:,0:2]
    sigmas = torch.tensor([10., 20, 30, 40, 50])

    r1 = _render_spot_loop(data, sigmas, 10, 128)
    r2 = _render_batch(data, sigmas, 10, 128)
    r3 = _render_batch(data.half().cuda(), sigmas.half().cuda(), 10, 128).cpu()
    r4 = render_batch_weights(data, sigmas, torch.ones(batch, spots), 10, 128)

    r5 = render_batch_anisotropic(data, sigmas.unsqueeze(1).expand(sigmas.shape[0], 2), torch.ones(batch, spots), 10.0, 10.0, (128, 128))

    assert ((r1-r2).abs().max()) < 1e-5
    assert ((r1-r3).abs().max()) < 2e-3
    assert ((r1-r4).abs().max()) < 1e-5
    assert ((r1-r5).abs().max()) < 1e-5

    # Check for asymetric sized rendering 
    r1a = r1[:,:,1:-1] #cut off the first and last columns
    r5a = render_batch_anisotropic(data, sigmas.unsqueeze(1).expand(sigmas.shape[0], 2), torch.ones(batch, spots), 10.0, 10.0, (128, 126))
    assert ((r1a-r5a).abs().max()) < 1e-5

    # Check for asymmetric sigmas
    # Double sigma, np per pix and spacing in one axis 
    b_sigmas = torch.stack((sigmas, sigmas*2), 1)
    b_data = data * torch.tensor([1., 2.]).reshape(1,2).expand_as(data)
    r5b = render_batch_anisotropic(b_data, b_sigmas, torch.ones(batch, spots), 10, 20, (128, 126))

    assert ((r1a-r5b).abs().max()) < 1e-5

    # Check for multi-rendering
    [xy, yz, _] = render_multiple_scale(data3, sigmas, torch.ones(batch, spots), 10.0, 2.0, 128, 64)
    r6 = render_batch_anisotropic(data3[:,:,1:3], b_sigmas, torch.ones(batch, spots), 10.0, 20.0, (64,128))

    assert (xy - r1).abs().max() < 1e-5
    assert (yz - r6).abs().max() < 1e-5
    

render = render_batch_weights

def _bench()->None:

    # pylint: disable=import-outside-toplevel]
    import tqdm
    import time
    import generate_data
    from torch.utils.data import DataLoader, Dataset
    
    exemplar = torch.zeros(1).cuda().half()

    dataset_vertices = generate_data.load_ply_vertices('test_data/teapot.ply')

    teapots = [ t.to(exemplar) for t in 
                generate_data.pointcloud_dataset(dataset_vertices, 
                                                 size=10000, 
                                                 dropout=0,
                                                 teapot_size_nm=800, 
                                                 seed=0, 
                                                 scatter_xy_nm_sigma=10, 
                                                 anisotropic_scale_3_sigma=0) ]
    class _TeapotsDataset(Dataset):
        def __len__(self)->int:
            return len(teapots)
        def __getitem__(self, idx:int)->torch.Tensor:
            return teapots[idx]

        

    batch_size = 32
    loader = DataLoader(_TeapotsDataset(), batch_size=batch_size, drop_last=True)

    sigmas = torch.ones(batch_size).to(exemplar)*20

    t1 = time.time()
    for i in tqdm.tqdm(loader, unit_scale=batch_size):
        _render_batch(i, sigmas, 10, 64)
    t2 = time.time()
    print(f"_render_batch {t2-t1}")

    t1 = time.time()
    weights = torch.ones(batch_size, teapots[0].shape[0]).to(exemplar)
    for i in tqdm.tqdm(loader, unit_scale=batch_size):
        render_batch_weights(i, sigmas, weights, 10, 64)
    t2 = time.time()
    print(f"render_batch_weights {t2-t1}")

    #fast = torch.compile(_render_spot_slow)
    #for i in tqdm.tqdm(loader, unit_scale=batch_size):
    #    fast(i, sigmas, 10, 64)
    #t1 = time.time()
    #for i in tqdm.tqdm(loader, unit_scale=batch_size):
    #    fast(i, sigmas, 10, 64)
    #t2 = time.time()
    #print(f"render_slow compiled {t2-t1}")



if __name__ == "__main__":
    test_render_1()
    test_render_slow_vs_3d()
    test_render_slow_vs_3d_2()
    test_render_slow_vs_3d_3()
    test_render_3d_1()
    test_render()
    test_per_spot_sigma()
    print("Done test render")
    _bench()



