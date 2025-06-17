import os
from pathlib import Path
import textwrap
from typing import Tuple

import tqdm
import open3d
import numpy as np
import torch

import save_ply
import train
import generate_data
import train
import device
import network
from matrix import trn
import localisation_data


logdir=Path('log/1716727671-4e5e064bec2b231d4c6e14a6a234eca3d8267891/')



files = list(logdir.glob('*-model_only/current_model.txt'))
files.sort()
files = files[0::2]

os.mkdir('hax/anim')

sphere = open3d.geometry.TriangleMesh.create_sphere(resolution=8)

sphere_vertices = torch.tensor(np.asarray(sphere.vertices))
sphere_tri_ind =  torch.tensor(np.asarray(sphere.triangles))

def write_sphere_ply(points: torch.Tensor, weights: torch.Tensor, fn: str, radius: float)->None:
    npts = points.shape[0]

    
    nv = sphere_vertices.shape[0]
    nt = sphere_tri_ind.shape[0]

    # Create one sphere per point
    vertices = sphere_vertices.unsqueeze(0).expand(npts, nv, 3)
    
    # Scale by intensity
    vertices = vertices *  weights.reshape(npts,1, 1).expand(npts, nv, 3)* radius

    # Shift to point centres
    vertices = vertices + points.unsqueeze(1).expand(npts, nv, 3)

    # Now create the indices
    indices = sphere_tri_ind.unsqueeze(0).expand(npts, nt, 3) + nv*torch.arange(npts).reshape(npts,1,1).expand(npts, nt, 3)


    vertices = vertices.flatten(0,1)
    indices = indices.flatten(0,1)

    # Add counts
    indices = torch.cat((torch.ones(indices.shape[0],1,dtype=torch.int64)*3, indices), 1)

    with open(fn, "w", encoding='ascii') as ballfile:
        header = f"""\
            ply
            format ascii 1.0
            element vertex {vertices.shape[0]}
            property float x
            property float y
            property float z
            element face {indices.shape[0]}
            property list uchar int vertex_index
            end_header"""

        print(textwrap.dedent(header), file=ballfile)
        np.savetxt(ballfile, vertices)
        np.savetxt(ballfile, indices, fmt='%i')


def loadfile(fn: Path)->Tuple[torch.Tensor, torch.Tensor]:
    with fn.open() as f:
        txtlines = f.readlines()

        txt = [line.split() for line in txtlines]

        datalines = [ [float(d) for d in line ] for line in txt ]

        data = torch.tensor(datalines)

        points = data[:,0:3]

        assert data.shape[1] == 4
        weights = data[:,3]
        return points, weights
 

# Animation of fitting
if False:

    for i, filename in enumerate(tqdm.tqdm(files)):
        pts, wts = loadfile(filename)

        e = torch.tensor([1.0]).half().cuda()
        radius=15
        save_ply.save_pointcloud_as_mesh(f"hax/anim/mesh-{i:04}.ply", pts.to(e), wts.to(e), radius, .5)
        write_sphere_ply(pts, wts, f"hax/anim/balls-{i:04}.ply", radius)


# Animation of rotations
from matplotlib.pyplot import *


t_seed =  int(torch.randint(0xffff_ffff_ffff, []).item())
t_seed = 91444048830155

INDICES=[100, 200, 399]

dataset_vertices = generate_data.load_ply_vertices('data/test_data/bunny.ply')
# Simulated data params
teapot_size = 800
scatter_nm = 0 #0 not 10 to make rendering nicer

bunnies3D = [ t.cuda().half() for t in 
            generate_data.pointcloud_dataset3D(dataset_vertices, 
                                             size=1, 
                                             dropout=0.99,
                                             teapot_size_nm=teapot_size, 
                                             seed=t_seed, 
                                             offset_percent_sigma=0.0,
                                             scatter_xy_nm_sigma=scatter_nm, 
                                             scatter_z_nm_sigma=scatter_nm,
                                             anisotropic_scale_3_sigma=1,
                                             random_rotations=False
                                             ) ]

data_parameters = train.DataParametersXYYZ(
    image_size_xy=64,
    image_size_z=64,
    nm_per_pixel_xy=10.,
    z_scale = 1.0
)

params = train.TrainingParameters()
params.batch_size=75
params.schedule[0].epochs = 400
params.schedule[0].initial_psf = 160
params.schedule[0].final_psf = 40.0
params.schedule[0].psf_step_every= 20
params.schedule[0].initial_lr= 0.0001
params.schedule[0].final_lr= 0.0001


nets = [ network.PredictReconstructionStretchExpandValidDan6(**vars(data_parameters), model_size=400, data=bunnies3D)[0].to(device.device) for _ in INDICES]



sigmas = [ train._exponential_step(
    train.fwhm_to_sigma(params.schedule[0].initial_psf),
    train.fwhm_to_sigma(params.schedule[0].final_psf),
    index, 
    params.schedule[0].epochs,
    params.schedule[0].psf_step_every) for index in INDICES ]

states = [ torch.load(logdir/f'{index:05}/network.zip') for index in INDICES ]


for state, net in zip(states, nets):
    mapped = { k[10:]:v for k,v in state['state_dict'].items()}
    net.load_state_dict(mapped)
    net.eval()




ion()
torch.no_grad()


N = 150
raxis = torch.nn.functional.normalize(torch.tensor([1., 1, 1], device=device.device), dim=0)


def xprodmat(v: torch.Tensor)->torch.Tensor:
    v2 = v[...,2]
    v1 = v[...,1]
    v0 = v[...,0]
    O = torch.zeros_like(v2)

    m = torch.stack((
            torch.stack((  O, -v2,  v1), -1),
            torch.stack(( v2,   O, -v0), -1),
            torch.stack((-v1,  v0,   O), -1),
            ), -2)

    return m

axes = raxis.unsqueeze(0) * torch.arange(N, device=device.device).unsqueeze(1)/N * torch.pi * 2

rotations = torch.linalg.matrix_exp(xprodmat(axes))


rbuns = trn(rotations @ trn(bunnies3D[0].float()).unsqueeze(0).expand(N, 3, -1))


datasets = [ localisation_data.LocalisationDataSetMultipleDan6(data=rbuns, **vars(data_parameters), device=device.device) for _ in INDICES]
for sigma, dataset in zip(sigmas, datasets):
    dataset.set_sigma(sigma)

loaders = [ torch.utils.data.DataLoader(dataset, batch_size=1) for dataset in datasets]

COLS=4
figure(figsize=(10.24, 10.24), dpi=100)
for i, batches in enumerate(zip(*loaders)):

    clf()
    for row in range(len(INDICES)):
        style.use('dark_background')
        subplot(len(INDICES),COLS,2 + COLS*row)
        imshow(batches[row][0].cpu().squeeze(0).squeeze(0).flip(0), cmap='grey')
        if(row == 0):
            title('Input projection')
        axis('off')

        subplot(len(INDICES),COLS,1 + COLS*row)
        gca().scatter(rbuns[i][:,0].cpu(), rbuns[i][:,1].cpu(), s=.3)
        axis('scaled')
        axis((-200,200,-200,200))
        if(row == 0):
            title('Input model')
        axis('off')
        

        subplot(len(INDICES),COLS,3 + COLS*row)
        _, R, _, _, _ = nets[row].process_input(batches[row], sigma)
        R
        pts2 = trn(R @ trn(nets[-1].get_model()[0]).unsqueeze(0))
        gca().scatter(pts2[-1][:,0].cpu().detach(), pts2[-1][:,1].cpu().detach(), s=.3)
        axis('scaled')
        axis((-200,200,-200,200))
        if(row == 0):
            title('Predicted rotation')
        axis('off')
        
        subplot(len(INDICES),COLS,4 + COLS*row)
        _, R, _, _, _ = nets[row].process_input(batches[row], sigma)
        R
        pts2 = trn(R @ trn(nets[row].get_model()[0]).unsqueeze(0))
        gca().scatter(pts2[-1][:,0].cpu().detach(), pts2[-1][:,1].cpu().detach(), s=.3)
        axis('scaled')
        axis((-200,200,-200,200))
        if(row == 0):
            title('Predicted model')
        axis('off')

    pause(.1)

    savefig(f'hax/anim/figure-{i:04}.png', dpi=100)
