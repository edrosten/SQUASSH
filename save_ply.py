from pathlib import Path
from typing import Union, List, Tuple
import textwrap

from skimage import measure
import torch


def save(filename: Union[Path, str], points: List[Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int,int,int]]]])->None:
    """Save a set of 3D points as a ply file"""

    default_colour = "128 128 128"
    verts=[]

    for item in points:
       
        pointset = item if isinstance(item, torch.Tensor) else item[0]
        item_color = default_colour if isinstance(item, torch.Tensor) else " ".join([str(x) for x in item[1]])

        assert pointset.ndim == 2
        assert pointset.shape[1] == 3 or pointset.shape[1] == 6
        for vertex in pointset:
            
            if len(vertex) == 6:
                color = f"{int(vertex[3])} {int(vertex[4])} {int(vertex[5])}"
            else:
                color = item_color

            verts.append(" ".join([str(x.item()) for x in vertex[0:3]]) + " " + color)


    header=f"""\
    ply
    format ascii 1.0
    element vertex {len(verts)}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header"""

    with open(filename, 'w', encoding='utf-8') as f:
        print(textwrap.dedent(header), file=f)
        print("\n".join(verts), file=f)



def save_pointcloud_as_mesh(filename: Union[Path, str], data:torch.Tensor, weights: torch.Tensor, sigma: float, threshold:float=0.1, size:int=100, colour:tuple[int,int,int,int]=(127,127,127,255), comments: str="", maxval: None|float=None)->None:
    """Mesh a set of 3D points and save as a ply file"""

    with open(filename, 'w', encoding='utf-8') as f:

        hi = data.abs().max() * 1.2 if maxval is None else torch.tensor(maxval*1.0)

        r = (torch.arange(0, size).to(data) / (size-1)-.5) * 2 * hi
        xs = r.reshape(1,1,size).expand(size,size,size)
        ys = xs.permute(0,2,1)
        zs = xs.permute(2,0,1)
    
        cs = torch.stack((zs, ys, xs), 3).unsqueeze(0)
        coords = data.reshape(data.shape[0], 1, 1, 1, data.shape[1])

        weights = weights.reshape(data.shape[0], 1, 1, 1)

        vol = (weights*torch.exp(-((coords-cs)**2).sum(-1) / (2*sigma**2))).sum(0).float().cpu()

        print("ply", file=f)
        print("format ascii 1.0", file=f)
        print(f"comment sigma {sigma}", file=f)
        f.flush()
        
        threshold = vol.max().item()*threshold
        
        try:
            verts, faces, _, _ = measure.marching_cubes(vol.numpy(), threshold) #type: ignore
        except ValueError:
            return
        

        verts = (verts / (size-1)-.5) * 2 * hi.item()


        header = f"""\
        element vertex {len(verts)}
        comment {comments}
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        element face {len(faces)}
        property list uchar int vertex_index
        end_header"""

        print(textwrap.dedent(header), file=f)

        for v in verts:
            print(" ".join([str(x) for x in v]), " ", " ".join([str(i) for i in colour]), file=f)


        for face in faces:
            print(3, " ".join([str(i) for i in face][::-1]) ,file=f)
    
    # For unknown reasons this causes the GPU usage to spike very high
    # and remain there. It's cace, so repeated calls don't cause memory
    # exhaustion until the do. So, clear the cache "fixes" the problem
    # or hides it or something.
    # And now the cargo will come
    torch.cuda.empty_cache()
