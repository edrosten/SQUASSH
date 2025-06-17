from __future__ import annotations
from typing import List
import torch
import tifffile

import montage
import device
import data.mark_bates_2021

load_3d_separate = data.mark_bates_2021.load_all

def load_3d()->List[torch.Tensor]:
    d = load_3d_separate()
    flat =  [x for ds in d for x in ds]

    for f in flat:
        f-= f.mean(0)

    return flat

def load_3d_list_and_means()->tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    d = load_3d_separate()
    m: list[list[torch.Tensor]] = [] 
    for a in d:
        m.append([])
        for f in a:
            mmean = f.mean(0)
            f-= mmean
            m[-1].append(mmean)

    return d, m

def load_3d_list()->list[list[torch.Tensor]]:
    d = load_3d_separate()
    
    for a in d:
        for f in a:
            f-= f.mean(0)

    return d

if __name__ == "__main__":
    def _do()->None:

        data = load_3d()
        data = [d.to(device.device) for d in data]

        m = montage.make_stack_multiple(data, 3, 3, 64)
        tifffile.imwrite('hax/bates_stack.tiff', (torch.stack(m, 0).permute(0,2,3,1)*255).char().numpy())

    _do()

