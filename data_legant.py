from typing import List
import itertools

import tifffile
import torch
from pystrict import strict

import data.Legant
from localisation_data import GeneralLocalisationDataSet
import volumetric 

load = data.Legant.load_and_crop
load_chromatin = data.Legant.load_and_crop
load_tubulin = data.Legant.load_and_crop_tubulin

@strict
class ConcatDataset(GeneralLocalisationDataSet):
    def __init__(self, *datasets: volumetric.GeneralPreprojectedVoulmetric3Plane):
        
        assert datasets
        for d in datasets:
            assert d.xy_size() == datasets[0].xy_size()
            assert d.z_size() == datasets[0].z_size()
            assert len(d) == len(datasets[0])
            assert d.get_augmentations() == datasets[0].get_augmentations()


        self._datasets = datasets

    def xy_size(self)->int:
        '''Size of the returned data'''
        return self._datasets[0].xy_size()

    def z_size(self)->int:
        '''Size of the returned data'''
        return self._datasets[0].z_size()

    def set_sigma(self, sigma_nm: float)->None:
        """Updates the rendering sigma. Idempotent."""
        for d in self._datasets:
            d.set_sigma(sigma_nm)
            
    def get_augmentations(self)->int:
        return self._datasets[0].get_augmentations()

    def  __len__(self)->int:
        return len(self._datasets[0])

    def __getitem__(self, idx:int)->List[torch.Tensor]:
        return list(itertools.chain.from_iterable(d[idx]for d in self._datasets))


def _mainfunc()->None:


    _, _, _, _, data_fullsize, fmetadata = data.Legant.load_and_crop_both()

    tifffile.imwrite('hax/legant_cubes.tiff', data_fullsize.numpy(), imagej=True, resolution=(1/fmetadata.xy_nm_pix, 1/fmetadata.xy_nm_pix),  metadata={'axes': 'TZYX', "unit":"nm", 'spacing':fmetadata.z_nm_pix})

if __name__ == "__main__":
    _mainfunc()
