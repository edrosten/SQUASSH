from __future__ import annotations
import itertools
from typing import Iterable
from pathlib import Path

import pyvista as pv
import torch
import torchvision
import train
import device
 
import network
import generate_data
import render
from render import _cap_01

import matplotlib.pyplot as plt

dataset_vertices = generate_data.load_ply_vertices('data/test_data/bunny.ply')

bun_mesh = pv.read('data/test_data/bunny.ply')


centres = dataset_vertices[:,[0,2,1]]
centres[1] *= -1
centres -= centres.mean(0)
centres /= centres.abs().max()
centres[1,2] = 0

bun_mesh.points = centres.numpy()


#plt.ion()

#x = plt.gcf().add_subplot(projection='3d')

x_plane = 0.2


weights_x = _cap_01(centres[:,0]*.5/x_plane + .5)

#x.scatter(centres[:,0], centres[:,1], centres[:,2], c=weights_z, depthshade=False)



xs = torch.linspace(-.4, .4, 2)
y, z = torch.meshgrid(xs, xs)
x = torch.ones_like(y) * -x_plane

#x.plot_surface(x, y, z, alpha=0.5)






#plotter = pv.Plotter()
#
#
#grid = pv.StructuredGrid(x.numpy(), y.numpy(), z.numpy())
#plotter.add_mesh(grid, cmap='viridis', opacity=0.7, show_edges=False) 
#
#grid = pv.StructuredGrid(-x.numpy(), y.numpy(), z.numpy())
#plotter.add_mesh(grid, cmap='viridis', opacity=0.7, show_edges=False) 
#plotter.add_points(centres.numpy(), scalars=weights_x)
#plotter.show_bounds()
#plotter.show(interactive=True)
#




sigmas = 0.01 * torch.ones_like(centres)[:,0:2].unsqueeze(0)
weights = torch.ones_like(centres)[:,0].unsqueeze(0)
nm_per_pix = 0.008
size= 100


# XY, YZ, XZ
ims = render.render_multiple_dan6(centres.unsqueeze(0),sigmas, weights,nm_per_pix, nm_per_pix, size, size, x_plane, x_plane)

ims = [ i/i.max() for i in ims]

# Make a grid in order to display the image. Shirly, there must be a better way!
ps = torch.arange(size)*nm_per_pix
ps -= ps.mean()

y, z = torch.meshgrid(ps, ps)
x = torch.ones_like(y) * -x_plane



plotter = pv.Plotter(window_size=[700,1300])


plotter.add_mesh(bun_mesh, scalars=weights_x, cmap="jet", scalar_bar_args={
    "title_font_size":10,
    "label_font_size":10,
    "shadow":True,
    "n_labels":5,
    "italic":False,
    "fmt":"-",
    "font_family":"arial",
    "vertical":True,
    "interactive":True,
}   
)


pzz = torch.tensor([ps.min(), ps.max()])
pyy = torch.tensor([ps.min(), .2])
yy, zz = torch.meshgrid(pzz, pzz)
xx = torch.ones_like(yy) * -x_plane


grid = pv.StructuredGrid(xx.numpy(), yy.numpy(), zz.numpy())
plotter.add_mesh(grid, cmap='gray', opacity=0.5, show_edges=False, show_scalar_bar=False) 
grid = pv.StructuredGrid(-xx.numpy(), -yy.numpy(), zz.numpy())
plotter.add_mesh(grid, cmap='gray', opacity=0.5, show_edges=False, show_scalar_bar=False) 

grid = pv.StructuredGrid(x.numpy(), (y+0*y.min()).numpy(), (z+2*z.min()).numpy())
plotter.add_mesh(grid, scalars = ims[3].squeeze().numpy(), cmap='gray', opacity=1.0, show_edges=False, show_scalar_bar=False) 

grid = pv.StructuredGrid(-x.numpy(), (y-0*y.min()).numpy(), (z-2*z.min()).numpy())
plotter.add_mesh(grid, scalars = ims[2].squeeze().numpy(), cmap='gray', opacity=1.0, show_edges=False, show_scalar_bar=False) 
#plotter.show_bounds()

plotter.camera.position=(2.54730759761373, -3.3700356762770047, 0.4295907914926287)
plotter.camera.up = (-0.03886823063522411, 0.08796422367661456, 0.9953650365570701)
plotter.camera.focal_point = (0.12502440074922336, 0.07633781641562362, 0.030433280276242522)
plotter.camera.zoom(.88)

plotter.show(interactive=False, auto_close=False)
plotter.screenshot("hax/figure_1_6plane.png")
plotter.close()



