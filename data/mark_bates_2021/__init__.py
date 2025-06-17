from typing import List
from pathlib import Path

import h5py                                                                                                                 
import numpy as np                                                                                                                                      
import torch                                                                                                                    



def _load_h5_file(filename: Path)->List[List[torch.Tensor]]:
    
    file = h5py.File(filename)
    d = np.array(file['output_data']['molecule_data'])

    xind = d.dtype.names.index('X_POS_PIXELS')  
    yind = d.dtype.names.index('Y_POS_PIXELS')  
    zind = d.dtype.names.index('Z_POS_PIXELS')  
    cind = d.dtype.names.index('CHANNEL')  



    xs = torch.tensor([i[xind] for i in d])
    ys = torch.tensor([i[yind] for i in d])
    zs = torch.tensor([i[zind] for i in d])
    cs = torch.tensor([i[cind] for i in d])

    nm_per_pix= np.array(file['output_data']['pixel_size_um']).item() * 1000

    # Channel as float. A bit of a hack but it's fine
    xyzc = torch.cat([torch.stack([xs, ys, zs], 1) * nm_per_pix, cs.unsqueeze(1)], 1)

    # Group by segment
    segment_start_indices = file['output_data']['particle_index_array']
    counts = file['output_data']['n_molecules_per_particle']
    segments = [ xyzc[start:start+count, :] for start, count in zip(segment_start_indices, counts) ]

    # Now split each segment by channel
     
    # Sanity check:
    if not (torch.unique(cs) == torch.arange(cs.max().item()+1)).all():
        raise RuntimeError(f'Bad channels in {str(filename)}')
    
    n_channels: int = int(cs.max().item()) + 1

    ret: List[List[torch.Tensor]] =  []
    for segment in segments:
        ret.append([segment[segment[:,3] == channel, 0:3] for channel in range(n_channels)])
        
    return ret
        

    

def load_all()->List[List[torch.Tensor]]:
    
    folder = Path(__file__).parent/'2021_02_03_nup96'
    files = list(folder.glob('*.h5'))
    files.sort()
    
    # This is one colour data, so just splat away the colour channel
    return [ [ i[0] for i in _load_h5_file(f) ]  for f in files]

