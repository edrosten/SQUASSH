from __future__ import annotations
import math
from typing import Iterable, List, Literal

import tqdm
import torch

import render
import device as dev

def make_stack(data: Iterable[torch.Tensor], sigma: float, nm_per_pixel: float, S: int)->List[torch.Tensor]:
    '''Render a stack of images'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = [d.to(device).half() for d in data]
    sigma_t = torch.tensor([sigma]).to(device).half()
    
    rendered: List[torch.Tensor] = []
    for d in tqdm.tqdm(data):
        rendered.append(render.render_batch_weights(d.unsqueeze(0), weights=torch.ones_like(d)[:,0].unsqueeze(0), sigma_nm=sigma_t, nm_per_pixel=nm_per_pixel, size=S))

    return rendered

def _render_all_multiple(data: Iterable[torch.Tensor], sigma: float, nm_per_pixel: float, S: int, z_scale: int)->List[List[torch.Tensor]]:
    rendered: List[List[torch.Tensor]] = []
    for d in tqdm.tqdm(data):
        if d.shape[1] == 3:
            pts = d
        elif d.shape[1] == 5:
            pts = d[:,0:3]
        else:
            assert False

        rendered.append(
                render.render_multiple_scale(
                    centres=pts.unsqueeze(0),
                    sigma_xy_nm = sigma * torch.ones(1, device=d.device, dtype=d.dtype),
                    weights = torch.ones(1, d.shape[0], device=d.device, dtype=d.dtype),
                    nm_per_pixel_xy=nm_per_pixel ,
                    z_scale=z_scale,
                    xy_size=S,
                    z_size=S//z_scale
                )
            )            
    return rendered

# pylint: disable=too-many-positional-arguments
def make_stack_multiple(data: Iterable[torch.Tensor], sigma: float, nm_per_pixel: float, S: int, z_scale:int=2, device:Literal['auto']|torch.device='auto')->List[torch.Tensor]:
    '''Render a stack of images'''
    
    if device == 'auto':
        actual_device = dev.device
    else:
        actual_device = device

    data = [d.to(actual_device).half() for d in data]
    rendered = _render_all_multiple(data, sigma, nm_per_pixel, S, z_scale) 
    return make_stack_images(rendered)


def make_stack_images(rendered:List[List[torch.Tensor]])->List[torch.Tensor]:
    '''Composite images together into a per item montage'''
    imgs=[]

    border=2 
    for d in rendered:
        width = sum(i.shape[-1] for i in d) + border * (len(d))
        height = max(i.shape[-2] for i in d)

        img = torch.zeros(3, height, width)
        #Set bg to blue
        img[2,:,:] = 1
        
        h=border//2
        for i in d:
            i = i.float()
            i = i/(i.max() + 1e-10)
            i = i.squeeze(0).squeeze(0)
            img[0,0:i.shape[0], h:h+i.shape[1]] = i
            img[1,0:i.shape[0], h:h+i.shape[1]] = i
            img[2,0:i.shape[0], h:h+i.shape[1]] = i
            h += i.shape[1] + border
        
        imgs.append(img)

    return imgs



def montage(rendered: List[torch.Tensor])->torch.Tensor:
    '''Create a montage image'''
    norm = [ d / d.max() for d in rendered ]

    N = int(math.ceil(math.sqrt(len(rendered))))
    # Overall 
    S = max((max(i.shape) for i in rendered))
    m= torch.zeros(S*N, S*N)

    for i,n in enumerate(norm):
        
        r = i // N
        c = i % N
        #n[0,:,0]=-1
        #n[0,0,:]=-1
        m[r*S:(r+1)*S, c*S:(c+1)*S] = n.cpu().squeeze()

    return m



def make_montage(data: Iterable[torch.Tensor], sigma:float=50.0, nm_per_pixel:float=10.0,S:int=128)->torch.Tensor:
    '''Create a montage image'''
    rendered = make_stack(data, sigma, nm_per_pixel, S)
    return montage(rendered)
