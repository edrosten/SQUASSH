# pylint: disable=missing-function-docstring
from __future__ import annotations
from typing import cast
from pathlib import Path

import tqdm
import h5py
import torch
import cv2
from torch import Tensor
import numpy as np

from ..segment_markup import _project_to_image, _coordinate_scale, _get_segments_scale

_PIXEL_SIZE_NM = 108 

_IMG_SIZE=2048
_IMG_SIZE_2=8192
_MIN_SIZE=50

print('importing')

def _load_local_hdf5_parts(filename: Path, parts: list[str])->dict[str, Tensor]:
    with h5py.File(filename) as f:
        hdfdata = np.array(f['locs'])

    ret = {}
    print(hdfdata.dtype)
    for n in parts:
        if hdfdata.dtype.names is None:
            raise RuntimeError('Error loading HDF5 file: no named data types')

        i = hdfdata.dtype.names.index(n)
        print("Loading ", i, n)
        
        # torch lacks unsigned types. In practice, uint32s come as
        # indices and frame numbers, so much less than 2e+09
        newtype = hdfdata[0][i].dtype
        if newtype == np.uint32:
            newtype = np.int32

        ret[n] = torch.tensor([f[i].astype(newtype) for f in hdfdata])

    return ret


def _project_file_to_image(file:Path, size: int)->None:
    
    points = _load_local_hdf5_parts(file, ['x', 'y'])
    outfile = file.parent/(file.stem + "_projected22.png")
    pts = torch.stack([points['x'], points['y']], 1)
    _project_to_image(pts, outfile, size, power=0.2)

def project_ro_image()->None:
    file = Path(__file__).parent/'320pM_R4_1_cropped_locs_filter_render_drift_corrected.hdf5'
    _project_file_to_image(file, _IMG_SIZE)

def project_set_2_to_images()->None:
    files = [
        "01082024_50pm_1_sp0_driftcorr_filter.hdf5",
        "04072024_320pm_r2_1_driftcorr_filter.hdf5",
        "04072024_320pm_r4_1__driftcorr_filter.hdf5",
        "040724_240pm_r4_1_driftcorr_filtered_filter.hdf5",
        "040724_320pm_r2_2_driftcorr_filter.hdf5",
    ]

    for f in files:
        file = Path(__file__).parent/'2'/f
        _project_file_to_image(file, _IMG_SIZE_2)

def project_set_3_to_images()->None:
    files = [
        "100pM_R4_1_locs_driftcorr.hdf5",
        #"200pM_R4_1_locs_driftcorr.hdf5",
        #"20240806_DNA_PAINT_7xR4_25pM_locs_driftcorr.hdf5",
        #"20240806_DNA_PAINT_Seeds_7xR4_25pM_2_locs_driftcorr.hdf5",
        #"20240806_DNA_PAINT_Seeds_7xR4_25pM_3_locs_driftcorr.hdf5",
        #"20240807_Segmented_seeds_5xR2_18pM_1_locs_driftcorr.hdf5",
        #"20240807_Segmented_seeds_5xR4_25pM_1_locs_driftcorr.hdf5",
        #"202408087_SegmentedSeeds_7xR4_25pM_1_locs_driftcorr.hdf5",
        #"202408087_SegmentedSeeds_7xR5_18pM_1_locs_driftcorr2.hdf5",
        #"400pM_R4_1_locs_driftcorr.hdf5",

    ]

    for f in files:
        file = Path(__file__).parent/'3'/f
        _project_file_to_image(file, _IMG_SIZE_2)



def _split_by_frame(xy_frame_index: torch.Tensor, max_frame: int)->list[Tensor]:
    # fields being x, y, frame, index, group, photons
    assert xy_frame_index.ndim == 2
    assert xy_frame_index.shape[1] == 6
    assert max_frame >= xy_frame_index[:,2].max()
    

    framedata: list[list[Tensor]] = [[] for _ in range(max_frame+1)]
    for datum in tqdm.tqdm(xy_frame_index, 'Splitting by frame'):
        framedata[int(datum[2])].append(datum)

    
    return [torch.stack(i, 0) for i in framedata]




def load_dataset(particles: Path, size:int)->tuple[list[Tensor], list[Tensor]]:
    return _load_dataset_general(particles, particles.parent/(particles.stem + "_projected_markup.png"), size, 64)

def _load_dataset_general(particles: Path, markup: Path, size: int, segment_length: float)->tuple[list[Tensor], list[Tensor]]:
    # This function loads and processes, so doesn't give the raw data. It only
    # gives the chained data which has fewer points, and the points are in different positions
    # And some at the far edges may not be there. 
    data = load_cache_hdf_file(particles)

    # The projection was originally made on the unchained (raw) xy data, so we need the raw data
    # in order to get back to the original scale, because scaling is done using the most extreme
    # points in position. Yes that was not great, but I ain't redoing all the markup. 
    raw_xy = _load_local_hdf5_parts(particles, ['x', 'y'])
    xy = torch.stack([raw_xy['x'], raw_xy['y']], 1)
    scale = _coordinate_scale(xy, size)/_PIXEL_SIZE_NM
    print(scale)


    im = cv2.imread(str(markup))
    if im.dtype != np.uint8:
        raise RuntimeError('image has the wrong type')

    im8 = cast(np.typing.NDArray[np.uint8], im)
    

    # Spplit into named channels
    xyz = {
        'x': data[:,0],
        'y': data[:,1],
        'z': data[:,2],
    }
    
    segments = _get_segments_scale(im8, xyz, size, segment_length=segment_length, segment_width=100, scale=scale)
    print(f'NUM = {len(segments)}')
    
    good_segments = [ i-i.mean(0) for i in segments if i.shape[0] >= _MIN_SIZE]
    good_means    = [   i.mean(0) for i in segments if i.shape[0] >= _MIN_SIZE]
    print(f'Number of segments = {len(good_segments)}')

    return good_segments, good_means


def load()->list[Tensor]:
    return load_dataset(Path(__file__).parent/"320pM_R4_1_cropped_locs_filter_render_drift_corrected.hdf5", _IMG_SIZE)[0]


def load2()->dict[str, list[Tensor]]:
    dataset = load_dataset(Path(__file__).parent/"2"/"040724_320pm_r2_2_driftcorr_filter.hdf5", _IMG_SIZE_2)[0]
    # Markup from Susan
    good = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 42, 43, 44,
            52, 53, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 85,
            86, 87, 88, 89, 101, 105, 106, 117, 122, 123, 124, 126, 128, 129, 130, 131,
            132, 133, 137, 142, 145, 146, 148, 150, 151, 157, 169, 170, 177, 181, 182,
            183, 184, 185, 186, 187, 188, 189, 194, 195, 196, 199, 200, 201, 203, 204,
            205, 206, 207, 208, 209, 211, 213, 214, 216, 217, 218, 237, 245, 247, 248,
            249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
            265, 266, 267, 269, 271, 274, 277, 283, 288, 289, 290, 292, 293, 294, 295,
            298, 299, 300, 302, 303, 305, 308, 309, 310, 311, 312, 313, 314, 316, 317,
            318, 319, 320, 325, 326, 327, 328, 329, 330, 331, 334, 337, 346, 347, 348,
            349, 350, 352, 353, 356, 362, 364, 365, 366, 367, 388, 389, 390, 392, 393,
            395, 396, 397, 398, 399, 401, 402, 404, 405, 407, 408, 410, 411, 412, 415,
            416, 418, 419, 420, 422, 423, 424, 425, 427, 431, 432, 435, 441, 442, 446,
            450, 451, 453, 456, 464, 466, 472, 473, 474, 475, 476, 477, 485, 486, 498,
            501, 509, 515, 520, 522, 523, 531, 533, 545, 546, 548, 554, 555, 556, 557,
            558, 560, 561, 563, 565, 567, 568, 570, 571, 573, 574, 575, 576, 578, 594,
            599, 605, 606, 607, 609, 616]

    return {"040724_320pm_r2_2_driftcorr_filter": [ dataset[i] for i in good]}

def load_01082024_50pm_1_sp0()->list[Tensor]:
    return  load_dataset(Path(__file__).parent/"2"/"01082024_50pm_1_sp0_driftcorr_filter.hdf5", _IMG_SIZE_2)[0]

def _parse_filter_file(f: Path)->list[int]:
    with f.open() as handle:
        lines = handle.readlines()

    return [ int(i.strip()) for i in lines if i.strip() != "" ]

def load_3(segment_length:int=64, do_filter:bool=True)->dict[str, list[Tensor]]:
    files = [
        "100pM_R4_1_locs_driftcorr.hdf5",
        "200pM_R4_1_locs_driftcorr.hdf5",
        "20240806_DNA_PAINT_7xR4_25pM_locs_driftcorr.hdf5",
        "20240806_DNA_PAINT_Seeds_7xR4_25pM_2_locs_driftcorr.hdf5",
        "20240806_DNA_PAINT_Seeds_7xR4_25pM_3_locs_driftcorr.hdf5",
        "20240807_Segmented_seeds_5xR2_18pM_1_locs_driftcorr.hdf5",
        "20240807_Segmented_seeds_5xR4_25pM_1_locs_driftcorr.hdf5",
        "202408087_SegmentedSeeds_7xR4_25pM_1_locs_driftcorr.hdf5",
        "400pM_R4_1_locs_driftcorr.hdf5",
    ]

    ret: dict[str, list[Tensor]] = {}
    
    for f in files:
        print(f"Loading {f}")
        filter_file = Path(__file__).parent/'3'/('dan-stacks-26e28ec4-params2-' + f[:-5] + '.tex')

        if filter_file.exists():
            particles = Path(__file__).parent/'3'/f
            markup = particles.parent/(particles.stem + "_projected_markup.png")

            loaded = _load_dataset_general(particles, markup, _IMG_SIZE_2, segment_length)[0]

            if do_filter and segment_length == 64: # Only did data filtering for length 64 lol
                good_items = _parse_filter_file(filter_file)
                # Note indexing in ImageJ and so the list of good items rocks it FORTRAN style 
                print(filter_file) 
                print(len(loaded))

                ret[f] = [ loaded[i-1] for i in good_items]
            else:
                ret[f] = loaded
        else:
            print("Nope.")
    return ret

# This function is very poorly named. Someone should do something about that
def load_cache_hdf_file(filename: Path)->Tensor:


    cache_file = Path('cache') / (filename.stem+".pt")
    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        tag=cache_file.parent / "CACHEDIR.TAG"
        if not tag.exists():
            with tag.open("w") as tagfile:
                print("Signature: 8a477f597d28d172789f06886806bc55", file=tagfile)
    else:
        print("Loading from cache")
        loaded = torch.load(cache_file)
        if type(loaded) == Tensor: # pylint: disable=unidiomatic-typecheck
            return loaded
        raise RuntimeError("Bad cache")

    groups = load_hdf_file(filename)
    torch.save(groups, cache_file)
    return groups
    

def load_hdf_file(filename: Path)->Tensor:
    hdfdata = _load_local_hdf5_parts(filename, ['x', 'y', 'frame', 'photons'])
    
    # Hardcoded for now, numbers from Dan. 
    # Not sure if these vary
    pixel_size = _PIXEL_SIZE_NM
    max_r_nm = 10
    l_tirf = 100
    intensity_0 = 1800
    alpha = 0.15
    
    groups = _group_hdfdata(hdfdata, pixel_size, max_r_nm, l_tirf, intensity_0, alpha)
    return groups

def _group_hdfdata(hdfdata: dict[str,Tensor], pixel_size:float, max_r_nm: float, l_tirf: float, intensity_0: float, alpha: float)->Tensor:
    # Concat everything as float64. Maybe inefficient, but float64 can 
    # represent integers up to 2**48 perfectly and it is convenient.
    INDEX=3
    GROUP=4
    PHOTONS=5
    
    xy_frame_index_group = torch.stack([
        hdfdata['x'], 
        hdfdata['y'], 
        hdfdata['frame'].to(torch.float64), 
        torch.arange(hdfdata['x'].numel()),
        -torch.ones_like(hdfdata['x']),
        hdfdata['photons']
    ], 1)
    xy_frame_index_group[:, 0:2] *= pixel_size

    frames =  _split_by_frame(xy_frame_index_group, int(hdfdata['frame'].max()))
    
    # This modifies frames, inserting the group number
    _chain_framedata(frames, max_r_nm, GROUP)

    chained = torch.cat(frames)

    num_groups = int(chained[:, GROUP].max()+1)
    group_indices: list[list[Tensor]] = [ [] for _ in range(num_groups)]

    for point in tqdm.tqdm(chained, 'Grouping'):
        if point[GROUP] != -1:
            group_indices[int(point[GROUP])].append(point[INDEX])

    
    # Now collect by groups, filtering out small groups (i.e. less than 3)
    # and removing the first and last which may be partial frames
    groups: list[Tensor] = []
    for gi in group_indices:
        if len(gi) > 2:
            groups.append(chained[torch.stack(gi[1:-1]).to(torch.int32)])
    
    means_x_y_photons = torch.stack([ x[:, [0, 1, PHOTONS]].mean(0) for x in groups ], 0)

    # Intensity = Intensity_0 * (alpha + (1 - alpha) * exp(-z / l_tirf))
    # i/i0 = a + (1-a) * exp(-z/lt)
    # (i/i0-a)/(1-a) = exp(-z/lt)
    # -lt * (log(i/i0-a) - (1-a)) = lt


    # Convert photons to Z
    #means_x_y_photons[:,2] = -l_tirf * torch.log(means_x_y_photons[:,2]/intensity_0)
    means_x_y_photons[:,2] = -l_tirf * (torch.log(means_x_y_photons[:,2]/intensity_0 - alpha) - (1-alpha))

    return means_x_y_photons

def _chain_framedata(frames: list[Tensor], max_r_nm: float, GROUP: int)->None:
    current_group=0
    for frame_index, frame in enumerate(tqdm.tqdm(frames, 'Chaining')):
        for point in frame:
            if point[GROUP] == -1:
                xy = point[0:2] 
                point[GROUP] = current_group

                # Chain forwards in time, using the first point as an anchor
                for next_frame in frames[frame_index+1:]:
                    distances2 = ((xy.unsqueeze(0).expand(next_frame.shape[0], 2) - next_frame[:,0:2])**2).sum(1)

                    thresholded = distances2 <= max_r_nm**2

                    if not thresholded.any():
                        break
                    
                    # Pick the first. A bit arbitrary, but there should very rarely be > 1
                    next_frame[thresholded.nonzero()[0],GROUP] = current_group

                current_group+=1
