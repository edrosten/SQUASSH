import math
from typing import List, Union, Optional, Protocol, Callable, TypeVar, Iterator, cast
from abc import abstractmethod

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import render
from matrix import trn


T = TypeVar('T')

class FloatDivisible(Protocol):
    '''Protocol for arithmetic type (incomplete!)'''
    @abstractmethod
    def __truediv__(self: T, other: float)->T:...

Numeric = TypeVar('Numeric', bound=FloatDivisible)

def fwhm_to_sigma(fwhm: Numeric)->Numeric:
    '''Convert FWHM to sigma for a Gaussian'''
    return fwhm / (2 * math.sqrt(2 * math.log(2)))

def sigma_to_fwhm(sigma: Numeric)->Numeric:
    '''Convert sigma to FWHM for a Gaussian'''
    return sigma / fwhm_to_sigma(1.0)

class GeneralLocalisationDataSet(Dataset):
    '''lol'''
    @abstractmethod
    def set_sigma(self, sigma_nm: float)->None:
        """Updates the rendering sigma. Idempotent."""

    @abstractmethod
    def __len__(self)->int:...

    @abstractmethod
    def get_augmentations(self)->int:
        '''Returns the number of augmentations'''
        
class RenderFunc(Protocol):
    '''Function protocol for a renderer used by the dataset'''
    def __call__(self, centres: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor, datum_index: int)->List[torch.Tensor]:...
    # Note the datum index is a bit of a horrible hack for sending in side channel data.

class PointLocalisationDataSet(GeneralLocalisationDataSet):
    '''
    Localisation dataset with renderings from multiple angles

    Incoming data is a collection of (x,y) pairs per image. Assume distances
    in nm, centred about 0.

    The dataset class presents this as reconstructed images, returned in the Torch style as c,w,h with c=1

    '''

    def __init__(self,
                 data: List[torch.Tensor],
                 augmentations: int,
                 render_func: RenderFunc,
                 device: Union[None,torch.device]=None,
                 auto_compile:bool=True):

        assert data[0].shape[1] == 3, "Data pounts must be 3D"
        assert augmentations > 0, 'Must be > 0 augmentations'
        auto_compile=False

        self._data = data
        self._augmentations = augmentations
        # Ugh h4x. Should never have had auto compiling but it's here now for now.
        self._render = cast(RenderFunc, torch.compile(render_func, dynamic=True)) if auto_compile else render_func
        self._device = device
        self._batch_size = augmentations
        
        d= data[0]
        tmp_sigma_xy = torch.ones(1, dtype=d.dtype, device=d.device)
        tmp_weights = torch.ones(1, d.shape[0], dtype=d.dtype, device=d.device)
        result = render_func(centres= d.unsqueeze(0), sigma=tmp_sigma_xy, weights=tmp_weights, datum_index=0)
        
        # Allocate memory based on whatever render returns
        self._rendered_data = [ torch.zeros(len(data)*augmentations, *i.shape[1:], device=device, dtype=d.dtype)  for i in result]

        self._sigma_xy: Optional[float] = None

    def set_batch_size(self, batch_size: int)->None:
        '''Set the batch size for rerendering. This may need to be smaller than augmentations
        for large data elements to prevent out of memory errors'''
        assert self._augmentations % batch_size == 0, 'Batch size muse divide into augmentations'
        self._batch_size = batch_size

    def set_sigma(self, sigma_nm: float)->None:
        """Updates the rendering sigma. Idempotent."""
        if sigma_nm == self._sigma_xy:
            return

        torch.cuda.empty_cache()
        self._sigma_xy = sigma_nm

        print("Re-rendering dataset")
        for i in tqdm(range(len(self._data)), unit_scale = self._augmentations):
            pts = self._data[i].to(self._device).unsqueeze(0).expand(self._augmentations, self._data[i].shape[0], 3)

            if self._augmentations > 1:
                angles = torch.rand(self._augmentations, 1, 1, device=self._device, dtype=self._data[0].dtype) * 2 * torch.pi
                c = torch.cos(angles)
                s = torch.sin(angles)
                O = torch.zeros_like(c) #noqa
                l = torch.ones_like(c) #noqa

                rots = torch.cat((
                    torch.cat(( c, s, O), 2),
                    torch.cat((-s, c, O), 2),
                    torch.cat(( O, O, l), 2)),
                    1)

                pts = trn(rots @ trn(pts))

            # Render in batches.
            sigma_nm_batch = torch.tensor(sigma_nm, dtype=self._data[0].dtype, device=self._device).unsqueeze(0).expand(self._batch_size)
            weights = torch.ones((self._batch_size, pts.shape[1]), dtype=pts.dtype, device=pts.device)
            
            bs = self._batch_size
            for b in range(0, self._augmentations, self._batch_size):
                result = self._render(centres=pts[b:b+bs,...], sigma=sigma_nm_batch, weights=weights, datum_index=i) 

                start=i*self._augmentations +b
                end=start + bs
                for i,v in enumerate(result):
                    self._rendered_data[i][start:end:,...] = v

    def get_augmentations(self)->int:
        return self._augmentations
    
    def  __len__(self)->int:
        return self._rendered_data[0].shape[0]

    def __getitem__(self, idx:int)->List[torch.Tensor]:
        if self._sigma_xy is None:
            raise RuntimeError('Rendering sigma not set')

        # Add singleton channel dimension
        return [ i[idx, :, :].unsqueeze(0).float() for i in self._rendered_data]

    def __iter__(self)->Iterator[List[torch.Tensor]]:
        for i in range(len(self)):
            yield self[i]


def LocalisationDataSet2D(image_size: int, nm_per_pixel: float, data: List[torch.Tensor], augmentations: int = 1, device: Union[None,torch.device]=None)->PointLocalisationDataSet:
    ''' Localisation DataSet with 2D data (and so one angle) '''
    def renderer(centres: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor, datum_index: int)->List[torch.Tensor]:
        _=datum_index
        return [render.render_batch_weights(centres=centres[:,:,0:2], weights=weights, sigma_nm=sigma, nm_per_pixel=nm_per_pixel, size=image_size)]

    assert data[0].shape[1] == 2, "Data pounts must be 2D"

    data_3d = [ torch.cat([i, torch.zeros(i.shape[0], 1, dtype=i.dtype, device=i.device)], 1) for i in data ]
    return PointLocalisationDataSet(data_3d, augmentations, renderer, device)


def LocalisationDataSet(image_size: int, nm_per_pixel: float, data: List[torch.Tensor], augmentations: int = 1, device: Union[None,torch.device]=None)->PointLocalisationDataSet:
    ''' Localisation DataSet with one angle '''
    def renderer(centres: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor, datum_index: int)->List[torch.Tensor]:
        _=datum_index
        return [render.render_batch_weights(centres=centres[:,:,0:2], weights=weights, sigma_nm=sigma, nm_per_pixel=nm_per_pixel, size=image_size)]
    return PointLocalisationDataSet(data, augmentations, renderer, device)

def LocalisationDataSetMultiple(
                 image_size_xy: int,
                 image_size_z: int,
                 nm_per_pixel_xy: float,
                 z_scale: float,
                 data: List[torch.Tensor],
                 augmentations: int=1,
                 device: Union[None,torch.device]=None)->PointLocalisationDataSet:
    '''
    Localisation dataset with renderings from 3 angles, XY, YZ, XZ
    '''
    def renderer(centres: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor, datum_index: int)->List[torch.Tensor]:
        _=datum_index
        return render.render_multiple_scale(
            centres=centres,
            sigma_xy_nm=sigma,
            weights=weights,
            nm_per_pixel_xy=nm_per_pixel_xy,
            z_scale=z_scale,
            xy_size=image_size_xy,
            z_size=image_size_z)

    return PointLocalisationDataSet(data, augmentations, renderer, device)


def RenderDan6(data: List[torch.Tensor])->Callable[..., List[torch.Tensor]]:
    '''6-plane rendering style'''
    _PERCENTILE=90
    z=[]
    for i in data: 
        z += i[:,2].abs().tolist()
    z.sort()
    z_plane = z[len(z)*_PERCENTILE//100]

    xy=[]
    for i in data: 
        xy.append(i[:,0:2])
    r = sorted((torch.cat(xy, 0)**2).sum(1).sqrt().tolist())

    xy_plane = r[len(r)*_PERCENTILE//100]


    def render_scale(centres: torch.Tensor, sigma_xy_nm: torch.Tensor, weights: torch.Tensor, nm_per_pixel_xy: float, z_scale: float, xy_size: int, z_size: int)->List[torch.Tensor]:
        return render.render_multiple_dan6(
            centres, 
            sigma_xy_z_nm=torch.stack([sigma_xy_nm, sigma_xy_nm*z_scale], -1), 
            weights=weights, 
            nm_per_pixel_xy=nm_per_pixel_xy, 
            nm_per_pixel_z=nm_per_pixel_xy*z_scale, 
            xy_size=xy_size, 
            z_size=z_size,
            max_abs_xy=xy_plane,
            max_abs_z=z_plane
            )
    return render_scale


def LocalisationDataSetMultipleDan6(
                 image_size_xy: int,
                 image_size_z: int,
                 nm_per_pixel_xy: float,
                 z_scale: float,
                 data: List[torch.Tensor],
                 augmentations: int=1,
                 device: Union[None,torch.device]=None)->PointLocalisationDataSet:
    '''Dataset using the 6-plane rendering'''
    dan6r = RenderDan6(data)

    def renderer(centres: torch.Tensor, sigma: torch.Tensor, weights: torch.Tensor, datum_index: int)->List[torch.Tensor]:
        _=datum_index
        return dan6r(
            centres=centres,
            sigma_xy_nm=sigma,
            weights=weights,
            nm_per_pixel_xy=nm_per_pixel_xy,
            z_scale=z_scale,
            xy_size=image_size_xy,
            z_size=image_size_z)

    return PointLocalisationDataSet(data, augmentations, renderer, device)

class LocalisationDataSetMultipleAndPrecision(GeneralLocalisationDataSet):
    '''
    Localisation dataset with renderings from multiple angles

    Incoming data is a collection of (x,y) pairs per image. Assume distances
    in nm, centred about 0.

    The dataset class presents this as reconstructed images, returned in the Torch style as c,w,h with c=1

    '''

    def __init__(self,
                 image_size_xy: int,
                 image_size_z: int,
                 nm_per_pixel_xy: float,
                 z_scale: float,
                 data: List[torch.Tensor],
                 augmentations: int = 1,
                 device: Union[None,torch.device]=None):

        self._image_size_xy = image_size_xy
        self._image_size_z = image_size_z
        self._nm_per_pixel_xy = nm_per_pixel_xy
        self._z_scale = z_scale
        self._data = data
        self._augmentations = augmentations
        self._device = device
        
        # Perform a test rendering, use nm_per_pix*3 as sigma. It's arbitrary
        d= data[0]
        tmp_sigma= torch.ones(d.shape[0], 2, dtype=d.dtype, device=d.device)*nm_per_pixel_xy*3
        tmp_weights = torch.ones(1, d.shape[0], dtype=d.dtype, device=d.device)

        result = render.render_multiple(d[...,0:3].unsqueeze(0), tmp_sigma.unsqueeze(0), tmp_weights, nm_per_pixel_xy, z_scale*nm_per_pixel_xy, image_size_xy, image_size_z)

        
        # Allocate memory based on whatever render returns
        self._rendered_data = [ torch.zeros(len(data)*augmentations, *i.shape[1:], device=device, dtype=d.dtype)  for i in result]

        self._sigma_xy: Optional[float] = None

        self._data_lowest_sigma_xy = float(min(d[:,3].min().item() for d in data))
        self._data_lowest_sigma_z = float(min(d[:,4].min().item() for d in data))
        print("*"*80)
        print(self._data_lowest_sigma_xy)

        assert data[0].shape[1] == 5, "Data pounts must be 3D + 2 precisions"

    def set_sigma(self, sigma_nm: float)->None:
        """Updates the rendering sigma. Idempotent.
        Note this corresponds to the sigma of the most precise point
        """
        if sigma_nm == self._sigma_xy:
            return

        torch.cuda.empty_cache()
        self._sigma_xy = sigma_nm

        # Sigmas add in quaderature: calculate the additional sigma needed
        # to make the loswst sigma meet the spec
        additional_sigma_xy = math.sqrt(sigma_nm**2 - self._data_lowest_sigma_xy**2)
        additional_sigma_z = math.sqrt(max(0, (self._z_scale*sigma_nm)**2 - self._data_lowest_sigma_z**2))

        print("Re-rendering dataset")
        for i in tqdm(range(len(self._data)), unit_scale = self._augmentations):
            pts = self._data[i][:,0:3].to(self._device).unsqueeze(0).expand(self._augmentations, self._data[i].shape[0], 3)
            

            sigmas = self._data[i][:,3:5].to(self._device)
            # Add in the additional sigma
            sigmas = (sigmas**2 + torch.tensor([additional_sigma_xy, additional_sigma_z], device=self._device).expand_as(sigmas)**2).sqrt()
            sigmas = sigmas.unsqueeze(0).expand(self._augmentations, *sigmas.shape)

            if self._augmentations > 1:
                angles = torch.rand(self._augmentations, 1, 1, device=self._device, dtype=self._data[0].dtype) * 2 * torch.pi
                c = torch.cos(angles)
                s = torch.sin(angles)
                O = torch.zeros_like(c) # noqa
                l = torch.ones_like(c) # noqa

                rots = torch.cat((
                   torch.cat(( c, s, O), 2),
                    torch.cat((-s, c, O), 2),
                    torch.cat(( O, O, l), 2)),
                    1)

                pts = trn(rots @ trn(pts))

            start=i*self._augmentations
            end=(i+1)*self._augmentations
            
            weights = torch.ones((pts.shape[0], pts.shape[1]), dtype=pts.dtype, device=pts.device)

            result = render.render_multiple(centres=pts, 
                sigma_xy_z_nm=sigmas, 
                weights=weights, 
                nm_per_pixel_xy=self._nm_per_pixel_xy, 
                nm_per_pixel_z =self._nm_per_pixel_xy * self._z_scale,
                xy_size=self._image_size_xy,
                z_size=self._image_size_z,
            )
            
            for i,v in enumerate(result):
                self._rendered_data[i][start:end:,:] = v


    def  __len__(self)->int:
        return self._rendered_data[0].shape[0]

    def __getitem__(self, idx:int)->List[torch.Tensor]:
        if self._sigma_xy is None:
            raise RuntimeError('Rendering sigma not set')

        # Add singleton channel dimension
        return [ i[idx, :, :].unsqueeze(0).float() for i in self._rendered_data]

    def get_augmentations(self)->int:
        return self._augmentations


def _line(start: torch.Tensor, end: torch.Tensor, n:int)->List[torch.Tensor]:
    return [start + (end-start)*i/n for i in range(n)]

def _adhoc_test()->None:
    # pylint: disable=import-outside-toplevel
    import tifffile 
    
    # Generate one data element to break symmetry
    #
    # Triangle in the y/z plane with it pointing towards +z
    # Small triangle in the y=0/z=0 corner offset slightly in x,y,z also pointing towards +z
    # Entire pattern duplicated and shifted in x

    data3d_ = _line( torch.tensor([0, 0.,  0]), torch.tensor([0, 1, 0]), 20)
    data3d_ += _line(torch.tensor([0, 1.,  0]), torch.tensor([0,.5, .7]), 20)
    data3d_ += _line(torch.tensor([0, .5, .7]), torch.tensor([0, 0., 0]), 20)

    data3d = torch.stack(data3d_, 0)
    data3d = torch.cat((data3d, data3d[:,:]*.5 + torch.tensor([.1 , .1, .1])), 0)
    data3d = torch.cat([data3d, data3d + torch.tensor([.5, 0, 0])], 0)
    data3d *= 200
    data3d -= data3d.mean(0)
    data5d = [torch.cat([data3d, torch.ones_like(data3d)[...,0:2]], -1)]


    dataset = LocalisationDataSetMultipleAndPrecision(data=data5d, image_size_xy=64, image_size_z=32, nm_per_pixel_xy=4, z_scale=2, augmentations=1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    dataset.set_sigma(1.1)

    imgs=[]

    border=2 
    
    for d in loader:
        width = sum(i.shape[-1] for i in d) + border * (len(d))
        height = max(i.shape[-2] for i in d)

        img = torch.zeros(3, height, width)
        #Set bg to blue
        img[2,:,:] = 1
        
        h=border//2
        for i in d:
            i /= i.max()
            i = i.squeeze(0).squeeze(0)
            img[0,0:i.shape[0], h:h+i.shape[1]] = i
            img[1,0:i.shape[0], h:h+i.shape[1]] = i
            img[2,0:i.shape[0], h:h+i.shape[1]] = i
            h += i.shape[1] + border
        
        img *=255
        img = img.permute(1, 2, 0)
        img = img.char()
        imgs.append(img)

    tifffile.imwrite('hax/staq.tiff', torch.stack(imgs, 0).numpy())

if __name__=="__main__":
    _adhoc_test()
