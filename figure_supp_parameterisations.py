import random

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot, axis, plot
import torch

import generate_data
import train_spectrin
dataset_vertices = generate_data.load_ply_vertices('data/test_data/bunny.ply')
dataset_vertices /= dataset_vertices.max()
dataset_vertices-=dataset_vertices.mean(0)

dataset_vertices = dataset_vertices[random.sample(range(dataset_vertices.shape[0]), 1000), :]

ROWS=5
COLS=5

cm = 1/2.54
plt.gcf().set_size_inches(18*cm, 18*cm)
ms = .1


plt.ion()
plt.clf()

of_evil = (-0.8, 1., -0.85, 1)
# Repetitions

for i in range(COLS):
    subplot(COLS, ROWS, i+1)
    for j in range(i+1):
        plot(dataset_vertices[:,0]+j*.5, dataset_vertices[:,1], 'C0.', markersize=ms)
        axis('equal')
        axis(of_evil)
        axis('off')



for i in range(COLS):
    subplot(COLS, ROWS, i+1 + ROWS*1)

    n = 2**(2*(i/(COLS-1) -.5))
    print(n)
    plot(dataset_vertices[:,0] *n, dataset_vertices[:,1], '.', markersize=ms)
    axis('equal')
    axis(of_evil)
    axis('off')




for i in range(COLS):
    subplot(COLS, ROWS, i+1 + ROWS*2)

    n = 2**(2*(i/(COLS-1) -.5))
    print(n)
    plot(-n*dataset_vertices[:,2], n* dataset_vertices[:,1], '.', markersize=ms)
    axis('equal')
    axis(of_evil)
    axis('off')



radial = train_spectrin.AxialRepeatRadialExpand(1,1)
for i in radial.parameters():
    i.requires_grad = False


radial._principal_axis[:] = torch.tensor([1., 0, 0])
radial._secondary_axis[:] = torch.tensor([0., 1, 0])
 
for i in range(COLS):
    subplot(COLS, ROWS, i+1+ROWS*3)
    n = 10**(2*(i/(COLS-1) -.5))

    ns = [ -10*(i==j) for j in range(4)]

    pts = radial(dataset_vertices, torch.ones(dataset_vertices.shape[0]), torch.tensor([[0, 0, *ns]]))[0].detach().squeeze(0)
    plot(pts[:,2], pts[:,1], '.', markersize=ms)
    axis('equal')
    axis(of_evil)
    axis('off')

for i in range(COLS):
    subplot(COLS, ROWS, i+1+ROWS*4)
    n = 10**(2*(i/(COLS-1) -.5))

    ns = [ 10*(i==j) for j in range(4)]

    pts = radial(dataset_vertices, torch.ones(dataset_vertices.shape[0]), torch.tensor([[0, 0, *ns]]))[0].detach().squeeze(0)
    plot(pts[:,2], pts[:,1], '.', markersize=ms)
    axis('equal')
    axis(of_evil)
    axis('off')

plt.tight_layout()
plt.savefig('hax/supp_parameterisations.pdf')
