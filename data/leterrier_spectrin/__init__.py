"""
Data from Christophe Leterrier
https://www.nature.com/articles/s41592-020-0962-1

Also new unpublished data
"""

import lzma
from pathlib import Path
from typing import Dict, List, Tuple, cast
import tqdm

import torch
import cv2
import numpy
import pyarrow.csv

from ..segment_markup import _project_to_image, _get_segments
from ..download_file_with_hash import ensure_cached_files_exist, cache_dir

_IMG_SIZE=2048

def _load_local_csv_xz(filename: str)->Dict[str, torch.Tensor]:
    file = cache_dir / "leterrier_spectrin" / filename
    with lzma.open(file, 'rb') as xzfile:
        csv_data = pyarrow.csv.read_csv(xzfile)

        return {k:torch.tensor(numpy.array(csv_data[k])) for k in csv_data.column_names}


_DATA_FILES_2=[
    'data2/div35_C2_N2_div35_b2s-add-m2_b2s647_647_8519K_DC1C_TS3D.csv.xz',
    'data2/MPS#1-2_C7_ N3b_div9_b2s-add-b3t-m2_b2s647_647_1098K_DC1C_TS3D.csv.xz',
    'data2/TestMPS#1_C4_N3_b2s-TM2(100)-NF_b2s647_647_5320K_DC1C0250_TS3D.csv.xz',
]
_MARKUP_FILES_2=[
    'data2/div35_C2_N2_div35_b2s-add-m2_b2s647_647_8519K_DC1C_TS3D_projected_markup.png',
    'data2/MPS#1-2_C7_ N3b_div9_b2s-add-b3t-m2_b2s647_647_1098K_DC1C_TS3D_projected_markup.png',
    'data2/TestMPS#1_C4_N3_b2s-TM2(100)-NF_b2s647_647_5320K_DC1C0250_TS3D_projected_markup.png',
]

_files = {
    "47e467cfd601b75e025c07d0326689d9e62a2212ebe7f3c2c528f66458e268ac":"leterrier_spectrin/data2/MPS#1-2_C7_ N3b_div9_b2s-add-b3t-m2_b2s647_647_1098K_DC1C_TS3D.csv.xz",
    "250e43f040dede9bd76032c30abd884874d4f72ab3aeb808ae9c8438bedc0431":"leterrier_spectrin/data2/TestMPS#1_C4_N3_b2s-TM2(100)-NF_b2s647_647_5320K_DC1C0250_TS3D.csv.xz",
    "e83fc2cc831c5bebfd3e0d81ad64d87ebab708896c5c3d7245083253c1498cc3":"leterrier_spectrin/data2/div35_C2_N2_div35_b2s-add-m2_b2s647_647_8519K_DC1C_TS3D.csv.xz",
    "ccdbd8a5fb924fad0c7107fa8bd97637d075c45ac195e169598730f27a70ee5a":"leterrier_spectrin/data2/MPS#1-2_C7_ N3b_div9_b2s-add-b3t-m2_b2s647_647_1098K_DC1C_TS3D_projected.png",
    "b5de210f907d948756b70e31c10a72cbb9c96c05158560686b0721b882f10a26":"leterrier_spectrin/data2/MPS#1-2_C7_ N3b_div9_b2s-add-b3t-m2_b2s647_647_1098K_DC1C_TS3D_projected_markup.png",
    "04a22fc6695cf16def09abbfbbfefd68941f70f24c06034c27958364a2de60ff":"leterrier_spectrin/data2/TestMPS#1_C4_N3_b2s-TM2(100)-NF_b2s647_647_5320K_DC1C0250_TS3D_projected.png",
    "b80b239f83b4ba4d66b8992a8ab69a21985f60b21192faa212e83583f7cfbfa0":"leterrier_spectrin/data2/TestMPS#1_C4_N3_b2s-TM2(100)-NF_b2s647_647_5320K_DC1C0250_TS3D_projected_markup.png",
    "2117f647d4a6bd4ebdb271949ba40665908af5144b5739b6b7f118705a553d5b":"leterrier_spectrin/data2/div35_C2_N2_div35_b2s-add-m2_b2s647_647_8519K_DC1C_TS3D_projected.png",
    "4f12b728fa688a2e6c79a6b2c1450228509c401dd114ac0b6bd89bf4389835b3":"leterrier_spectrin/data2/div35_C2_N2_div35_b2s-add-m2_b2s647_647_8519K_DC1C_TS3D_projected_markup.png",
}


def _load_unfiltered(file:str)->torch.Tensor:
    data = _load_local_csv_xz(file)
    fields = ['x [nm]', 'y [nm]', 'z [nm]']

    return torch.stack([data[i] for i in fields], 1)

def project_to_image2()->None:
    '''Load and project the data, for markup'''
    for file in tqdm.tqdm(_DATA_FILES_2):
        data = _load_local_csv_xz(file)
        fields = ['x [nm]', 'y [nm]']
        pts2d = torch.stack([data[i] for i in fields], 1)
        
        outfile = Path(__file__).parent / file
        directory = cache_dir/'leterrier_spectrin'/Path(file).parent
        filename = str(outfile.with_suffix('').with_suffix('').name)+"_projected.png"
        

        _project_to_image(pts2d, directory/filename, _IMG_SIZE, power=1.0)
        

def load_dataset(particles: str, markup: str)->Tuple[List[torch.Tensor], List[torch.Tensor]]:
    '''Load the 3D spectrin data and positions'''
    data = _load_local_csv_xz(particles)
    im = cv2.imread(str(cache_dir/'leterrier_spectrin'/markup))
    
    print(f'Markup image: {im.shape}')
    data = { i:data[i+' [nm]'] for i in 'xyz' }

    if im.dtype != numpy.uint8:
        raise RuntimeError('image has the wrong type')

    im8 = cast(numpy.typing.NDArray[numpy.uint8], im)

    segments = _get_segments(im8, data, _IMG_SIZE, segment_length=1000, segment_width=1500)
    print(f'Number of segments = {len(segments)}')
    means: List[torch.Tensor] = []

    for i in segments:
        m = i.mean(0)
        i -= m
        means.append(m)

    return segments, means


def load_3d_and_means_2()->List[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    '''Load the newer dataset, everything split by image'''
    ensure_cached_files_exist(_files)
    return [ load_dataset(i, j) for i,j in zip(_DATA_FILES_2, _MARKUP_FILES_2) ]

def load_unfiltered_3d_2()->List[torch.Tensor]:
    '''Load all the particles from the newer dataset'''
    ensure_cached_files_exist(_files)
    return [ _load_unfiltered(f) for f in _DATA_FILES_2]

def load_3d_2()->List[torch.Tensor]:
    '''Load just the selected particles from the newer dataset'''
    return sum((i for i,_ in load_3d_and_means_2()), [])

