'''Bring in microtubule data from 
https://www.nature.com/articles/s41592-022-01464-9
https://zenodo.org/record/6315338
Fluorogenic_3D_MT_locs.hdf
'''

from typing import List, Dict, Any, Tuple, Optional, Iterable, TypeVar, cast
from pathlib import Path
import h5py
import numpy as np
from numpy.typing import NDArray
import torch
import torchvision
import cv2
import tqdm

def _load_local_hdf5_file(filename: str)->Dict[str, torch.Tensor]:
    with h5py.File(Path(__file__).parent / filename) as f:
        hdfdata = np.array(f['Localizations'])

    ret = {}
    useful = ['x', 'y', 'z']
    for n in useful:
        if hdfdata.dtype.names is None:
            raise RuntimeError('Error loading HDF5 file: no named data types')

        i = hdfdata.dtype.names.index(n)
        print("Loading ", i, n)
        ret[n] = torch.tensor([ f[i] for f in hdfdata])

    return ret

def _coordinate_scale(xy: torch.Tensor, image_size: int)->float:
    divisor = xy.max().item()+1
    return image_size/divisor

def _to_image_coords_scale(xy: torch.Tensor, image_size: int, scale: float)->torch.Tensor:
    return (xy*scale).floor().to(torch.int32)

def _to_image_coords(xy: torch.Tensor, image_size: int)->torch.Tensor:
    return _to_image_coords_scale(xy, image_size, _coordinate_scale(xy, image_size))


def _xy_to_pixel_index_scale(xy: torch.Tensor, image_size: int, scale: float)->torch.Tensor:
    im_coords = _to_image_coords_scale(xy, image_size, scale)
    indices = im_coords[:,1]*image_size + im_coords[:,0]
    return indices

def _xy_to_pixel_index(xy: torch.Tensor, image_size: int)->torch.Tensor:
    return _xy_to_pixel_index_scale(xy, image_size, _coordinate_scale(xy, image_size))


def _stack(data: Dict[str, torch.Tensor], indices:List[str])->torch.Tensor:
    return torch.stack([data[i] for i in indices], 1)

def _project_xy_points(xy: torch.Tensor, size: int)->torch.Tensor:
    indices = _xy_to_pixel_index(xy, size)

    indices = indices.sort().values
    count: torch.Tensor
    indices, count = indices.unique(return_counts=True) #type: ignore

    base = torch.zeros(size, size)
    base.reshape(-1)[indices] += count
    return base

def _project_to_image(xy: torch.Tensor, filename: Path, size: int, power:float=1)->None:

    scale_nm_to_pixel = _coordinate_scale(xy, size) 
    print("Output nm/pix = ", 1/scale_nm_to_pixel)

    base = _project_xy_points(xy, size)
    rgb8 = ((base / base.max()).pow(power) * 255).to(torch.uint8).unsqueeze(0).expand(3, -1, -1)
    torchvision.io.write_png(rgb8, str(filename), compression_level=9)

def _r903x3(i: NDArray[Any])->NDArray[Any]:
    return i.transpose()[:,::-1]

def _morph_hitmiss(im: NDArray[np.uint8], selem: np.ndarray)->NDArray[np.uint8]:
    # Helper to get the types right
    return  cast(NDArray[np.uint8], cv2.morphologyEx(im, cv2.MORPH_HITMISS, selem))

def _morphological_skeletonize(dr: NDArray[np.uint8])->NDArray[np.uint8]:
    print("starting skeleton")
    # The two base hit-and-miss kernels for removable edge points
    k1 = np.array([
    [-1, -1, -1],
    [ 0,  1,  0],
    [ 1,  1,  1]])

    k2 = np.array([
    [ 0, -1, -1],
    [ 1,  1, -1],
    [ 0,  1,  0]])
    
    # The set of 8 kernels
    k_all = [k1, k2]
    k_all += [_r903x3(i) for i in k_all]
    k_all += [_r903x3(_r903x3(i)) for i in k_all]
    
    # Thin until you can thin no more.
    while True:
        count = dr.sum()
        for k in k_all:
            # Thinning
            dr = dr * (1-_morph_hitmiss(dr, k)) # dr ∩ ¬hitmiss(dr)
        if dr.sum() == count:
            break

    return dr


def _morphological_prune(A: NDArray[np.uint8], N: int)->NDArray[np.uint8]:
    #https://en.wikipedia.org/wiki/Pruning_(morphology)

    # The hit-or-miss kernels for endpoints
    k1 = np.array([
        [ 0, -1, -1],
        [ 1,  1, -1],
        [ 0, -1, -1]])
    k2 = np.array([
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1, -1]])

    k_all = [k1, k2]
    k_all += [_r903x3(i) for i in k_all]
    k_all += [_r903x3(_r903x3(i)) for i in k_all]
    
    X1 = A
    # Thin off the end points N times
    for i in range(N):
        for k in k_all:
            X1 = X1 * (1-_morph_hitmiss(X1, k))

    #This can sometimes leave it overconnected
    X1 = _morphological_skeletonize(X1)

    #Find the remaining endpoints
    X2 = X1*0
    for k in k_all:
        X2 = np.logical_or(X2, cv2.morphologyEx(X1, cv2.MORPH_HITMISS, k)).astype(np.uint8)

    # Dilate around the new endpoints and intersect with the original image, to
    # get back the ends that were thinned away
    for i in range(N):
        X2 = cv2.dilate(X2, np.ones((3,3), dtype=np.uint8)) * A # type: ignore

    final  = (X2 + X1)>0
    return _morphological_skeletonize(final.astype(np.uint8))

def _find_lines(thinned_np: NDArray[np.uint8])->List[torch.Tensor]:

    # Private copy
    thinned = torch.tensor(thinned_np).clone()
    # Thinned is a skeleton

    # The conv basically counts the number of things in a 3x3 window,
    # the multiply limits results to where there are points, so this 
    # gives the neighbour counts of all points. 
    neighbour_counts = (torch.nn.functional.conv2d(input=thinned.unsqueeze(0).unsqueeze(0), weight=torch.ones(1,1,3,3,dtype=torch.uint8), bias=None, padding=1).squeeze(0).squeeze(0)-1) * thinned
    
    if neighbour_counts.max() >= 3:
        breakpoint()
        raise RuntimeError("Found a branch in the labelling :(")

    endpoints = torch.nonzero((neighbour_counts.squeeze()==1))
    
    lines: List[List[Tuple[int,int]]] = []
    # For now assume there's nothing at the edge, lol
    for r, c in endpoints:
        if not thinned[r,c]:
            continue
        
        current: List[Tuple[int,int]] = [] 
        while True:
            #Record as x,y pairs
            current.append((c.item(),r.item()))
            thinned[r,c] = 0
        
            window3x3 = thinned[r-1:r+2, c-1:c+2]

            if window3x3.sum() == 0:
                break
            
            assert window3x3.sum() == 1 
            dr, dc = torch.nonzero(window3x3).squeeze(0)

            r += dr-1
            c += dc-1
        lines.append(current)
    
    return [torch.tensor(i) for i in lines]

def _get_segments(im: NDArray[np.uint8], data: Dict[str, torch.Tensor], image_size: int, segment_length: float, segment_width: float)->List[torch.Tensor]:
    xy = torch.stack((data['x'], data['y']), 1)
    scale = _coordinate_scale(xy, image_size)
    return _get_segments_scale(im, data, image_size, segment_length, segment_width, scale)


def _get_segments_scale(im: NDArray[np.uint8], data: Dict[str, torch.Tensor], image_size: int, segment_length: float, segment_width: float, scale: float)->List[torch.Tensor]:
    # Blue channels stores the labelling disc
    # green the data
    # red the labels
    b = torch.tensor(im[:,:,0])/255.0
    r = torch.tensor(im[:,:,2]) > 0 
    g = torch.tensor(im[:,:,1])

    #imshow(r)
    #waitforbuttonpress()

    xs = torch.arange(0, b.shape[1]).unsqueeze(0).expand(b.shape[0], -1)
    ys = torch.arange(0, b.shape[0]).unsqueeze(1).expand(-1, b.shape[1])


    x_hi = int((xs*b).max())
    x_lo = int(xs.shape[1] - ((xs.shape[1] - xs)*b).max())

    y_hi = int((ys*b).max())
    y_lo = int(ys.shape[0] - ((ys.shape[0] - ys)*b).max())
    
    # The markup brush is stored in the blue channel
    structuring_element = b[y_lo:y_hi+1,x_lo:x_hi+1].to(torch.uint8)

    structuring_element=structuring_element.numpy()
    # Shrink the structuring element: for some reason there are gaps with the full sized one
    #structuring_element = cv2.erode(structuring_element.numpy(), np.ones((3,3),dtype=np.uint8))
    
    # This will give a very good approximation of the centreline
    dr: NDArray[np.uint8] = cv2.erode(r.numpy().astype(np.uint8), structuring_element) # type: ignore
    #clf()
    #imshow(dr)
    #waitforbuttonpress()
   
    skel1 = _morphological_skeletonize(dr)
    skeleton = _morphological_prune(skel1, 10)

    #clf()
    #imshow(skeleton)
    #imshow(_morphological_prune(skeleton, 5))
    #waitforbuttonpress()

    lines = _find_lines(skeleton)


    xy = torch.stack((data['x'], data['y']), 1)
    im_coords = _to_image_coords_scale(xy, image_size, scale)
    #scale = _coordinate_scale(xy, image_size)


    # Walk along the line, in chunks of segment_length, generating segments
    # Nested lists to record the segment number. 
    segment_length *= scale 
    segment_width *= scale
    segments: List[List[torch.Tensor]] = []
    for line in lines:
        old: Optional[torch.Tensor] = None
        segments.append([])
        for point in line:
            if old is None or ((old - point)**2).sum().sqrt() >= segment_length:
                if old is not None:
                    segments[-1].append(torch.stack([old, point], 0))
                old = point

    U = TypeVar('U')
    def cat(iterable: Iterable[Iterable[U]])->Iterable[U]:
        for i in iterable:
            for j in i:
                yield j

    
    #clf()
    #imshow(im)
    #ctr = 0

        
    xyz = torch.stack((data['x'], data['y'], data['z']), 1)
    
    # Remove all data that isn't positively masked
    data_indices = _xy_to_pixel_index_scale(xy, image_size, scale)

    mask = r.reshape(-1)[data_indices]
    xyz = xyz[mask,:]
    xy = xy[mask,:]
    im_coords = im_coords[mask,:]

    allpts = []
    for segment in tqdm.tqdm(cat(segments), total=sum(len(i) for i in segments)):
        l1 = segment[0,:]
        l2 = segment[1,:]

        line = l2*1.0-l1
        l_hat = torch.nn.functional.normalize(line, dim=0)
        l_len = (line*line).sum().sqrt()

        n_hat = torch.cat([l_hat[1].reshape(1),-l_hat[0].reshape(1)], 0) # Normal

        
        dist_l1 = im_coords-l1
        alpha = (dist_l1 * l_hat).sum(1) / l_len #Position along the segment
        d = ((im_coords - l1)*n_hat).sum(1) # Distance to the segment
        mask = (alpha >= 0).logical_and(alpha <= 1).logical_and(d.abs() <= segment_width)
        
        allpts.append(xyz[torch.nonzero(mask).squeeze(), :])
       
        # # Uncomment this to plot:
        #lc = (l1+l2)/2
        #L=2
        #pts = xy[torch.nonzero(mask).squeeze(), :]*scale
        #plot(pts[:,0], pts[:,1], '.')
        #plot(segment[:,0], segment[:,1], 'w' if ctr%2==0 else 'r')
        #plot([lc[0], lc[0] + n_hat[0]*L], [lc[1], lc[1] + n_hat[1]*L], 'y')
        #ctr += 1


    #pause(.01)
    return allpts

