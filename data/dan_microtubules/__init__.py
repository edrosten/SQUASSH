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
from ..download_file_with_hash import ensure_cached_files_exist, cache_dir

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


def _parse_filter_file(f: Path)->list[int]:
    with f.open() as handle:
        lines = handle.readlines()

    return [ int(i.strip()) for i in lines if i.strip() != "" ]


_files = {
    "0ebb2773f30e4b0f363da59cddbe8c4c6de0a3eea1e19640a3e8e0148643cf19":"100pM_R4_1_locs_driftcorr.hdf5",
    "a8febb3592fb4e5efc7705a3d71e69401fd2e0896da7a9af33a8a585a9e9baa0":"200pM_R4_1_locs_driftcorr.hdf5",
    "f15852e9245d919677c72ffde8a13e46740c043aff4831db42525ff39661042f":"20240806_DNA_PAINT_7xR4_25pM_locs_driftcorr.hdf5",
    "89a886c33b7f8bf79d1ed1f3acd6849673af3bd03ccbeeb2315e1cfd8a69d917":"20240806_DNA_PAINT_Seeds_7xR4_25pM_2_locs_driftcorr.hdf5",
    "4f425fee45a48533a462ea449768023508bd8d7304cf154a7bd1c414400ebd78":"20240806_DNA_PAINT_Seeds_7xR4_25pM_3_locs_driftcorr.hdf5",
    "a4d65a4bf1f3402dfc35b3ce8511e891dd48f55f1591801626b82771ea0e0eb0":"20240807_Segmented_seeds_5xR2_18pM_1_locs_driftcorr.hdf5",
    "e426b085e9b82dc461c585b786dc4016a45deda26bd72c05dbd0cc14b7853881":"20240807_Segmented_seeds_5xR4_25pM_1_locs_driftcorr.hdf5",
    "1d6c360ec38a24d225fba31c5811d9942a66755523fdd456bea9c5dcea60829f":"202408087_SegmentedSeeds_7xR4_25pM_1_locs_driftcorr.hdf5",
    "742a4b82fbe12e3d12291ede3f8c09e488717cba7e0fb44df384685b0cb42015":"202408087_SegmentedSeeds_7xR5_18pM_1_locs_driftcorr2.hdf5",
    "f11b849390259df2952bcdd627ee9edce370cef4451c46ee77444006951df898":"400pM_R4_1_locs_driftcorr.hdf5",
    "880b2820125d5f0f1637dac1b96e10132feeb89d6828aea6f924739d39c1576a":"100pM_R4_1_locs_driftcorr_projected_markup.png",
    "a6977e9b856389de49ed6d3ccc0695de898dd0d621a7343daccbf408f06588d9":"200pM_R4_1_locs_driftcorr_projected_markup.png",
    "698f7fedc1b6cc20814893b280f571f5ff5e3f012ccd8555984fc8e737844023":"20240806_DNA_PAINT_7xR4_25pM_locs_driftcorr_projected_markup.png",
    "4ca7d3c4965531efbf748d75b1dad0d88a3599e4c17bcbc80486b973ad565325":"20240806_DNA_PAINT_Seeds_7xR4_25pM_2_locs_driftcorr_projected_markup.png",
    "df8e5894e0c5a726009968264f31602d8c5e27a2c4670ed027aa03645c7b0192":"20240806_DNA_PAINT_Seeds_7xR4_25pM_3_locs_driftcorr_projected_markup.png",
    "707a1e1affbe04c4a00ca7cc984c57761cba813b73b0f23035ec22fb08c8c515":"20240807_Segmented_seeds_5xR2_18pM_1_locs_driftcorr_projected_markup.png",
    "5a436219707188ff3be779e7a50da5b6dfa4b5458e171102fc658983804cbacc":"20240807_Segmented_seeds_5xR4_25pM_1_locs_driftcorr_projected_markup.png",
    "22f00a6546d4cadecd3f2eb842a7c4f1d3fbb84d417400a4454288a1010a7fc6":"202408087_SegmentedSeeds_7xR4_25pM_1_locs_driftcorr_projected_markup.png",
    "5057e108b085b0faa5552a5d2101a1a1b3bad8d13cea019ab367a6e32ab4a121":"400pM_R4_1_locs_driftcorr_projected_markup.png",
}


def _ensure_cache()->None:
    ensure_cached_files_exist({h: 'dan_microtubules/3/'+n for h,n in _files.items()})



def load_3(segment_length:int=64, do_filter:bool=True)->dict[str, list[Tensor]]:
    _ensure_cache()

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
            particles = cache_dir/'dan_microtubules'/'3'/f
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
        if type(loaded) == Tensor: # pylint: disable=unidiomatic-typecheck # noqa: E721
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
