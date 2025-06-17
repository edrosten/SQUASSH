from __future__ import annotations
from pathlib import Path
from typing import List
import random


import tqdm
import scipy
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import scatter, axis, plot, text, colorbar, gcf, sca, imshow, pause, title, clf

import resi_data
import mark_bates_data
from matrix import trn, scale_along_axis_and_expand_matrix
import network
import device
import save_ply
from train import fwhm_to_sigma, DataParametersXYYZ
from localisation_data import LocalisationDataSetMultipleDan6

plot_size = 31397.046857833866
FIGSCALE=2
cm = FIGSCALE/2.54  # centimeters in inches, plus an overall figure scaling
FS=7*FIGSCALE

def process_data_and_plot_some_crap(data3d: List[Tensor], means: List[Tensor], directory: Path)->tuple[Tensor, float]:
    '''lol'''
    nupc3d = [t.to(device.device).half() for t in data3d]
    
    # Set up a model to match the way it was trained. 
    SCALE=1.3
    data_parameters = DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 3*SCALE,
        z_scale = 2
    )
    fwhm = 10.0 * SCALE

    fwhm_t = torch.tensor(fwhm)

    net, parameterisation =network.PredictReconstructionStretchExpandValidDan6(model_size=700, **vars(data_parameters), data=nupc3d)
    parameterisation.max_stretch_factor_axis = 2.0
    parameterisation.max_stretch_factor_expand = 1.3
    
    # Load the learned weights
    loaded = torch.load(directory/"final_net.zip")
    loaded = {k[10:]: v for k,v in loaded.items()} # This is needed because of torch.compile
    net.load_state_dict(loaded)
    
    # Set up a sataset with the final sigma
    dataset = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)
    dataset.set_sigma(fwhm_to_sigma(fwhm))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Now process all of the data with the network
    net.cuda()
    net.eval()
    results: List[torch.Tensor] = []
    recons: List[List[torch.Tensor]] = []
    results_map: List[int] = []
    bad_indices: List[int] = []
    with torch.no_grad():
        for i, (mean, batch) in enumerate(zip(means, tqdm.tqdm(loader))):
            _,R,_,is_valid,parameters = net.process_input(batch, min_sigma_nm=fwhm_to_sigma(fwhm_t))
            recon, _, _, _ = net(batch, min_sigma_nm=fwhm_to_sigma(fwhm_t)) 
            scale = parameterisation.max_stretch_factor_axis**torch.tanh(parameters[:,0]).cpu()
            expand = parameterisation.max_stretch_factor_expand**torch.tanh(parameters[:,1]).cpu()
            
            # model scaled along axis.
            # R multiplies point as a column into camera space

            z_ax = torch.tensor([[[0,0,-1.]]]).to(R)

            angle = (z_ax @ R @ parameterisation.get_axis().unsqueeze(1).unsqueeze(0)).acos().cpu().squeeze(2).squeeze(1)* 180 / torch.pi


            if is_valid.item()> .5:
                if scale.item() > .8:
                    results.append(torch.cat((mean, scale, expand, angle)))
                    results_map.append(i)
                    recons.append(recon)
            else:
                bad_indices.append(i)

    res = torch.stack(results)
    res_orig = res

    #Project points onto the axis, find centre to centre spacing
    ax = parameterisation.get_axis().detach().cpu()
    pts, weights = (i.detach().cpu() for i in net.get_model())
    proj = (pts * ax.unsqueeze(0).expand(pts.shape[0], -1)).sum(1)
    hi = proj.abs().max()
    N = 1000
    xs = hi*(torch.arange(N)/(N-1) *2 -1)
    sigma=1

    ksdensity = ((-(proj.unsqueeze(0).expand(N, -1) - xs.unsqueeze(1).expand(N, len(pts)))**2 / (2*sigma**2)).exp() * weights.unsqueeze(0).expand(N,-1)).sum(1)
    localmax = torch.cat([torch.tensor([False]), (ksdensity[0:-2] < ksdensity[1:-1]).logical_and(ksdensity[2:] < ksdensity[1:-1]), torch.tensor([False])])
    _, max_indices = ksdensity[localmax].sort(descending=True)
    #Distance between 2 largest local maxima

    positions = xs[localmax.nonzero()[max_indices[0:2]]]

    spacing = (positions[0]-positions[1]).abs()

    #0, 9, 
    plt.subplots(figsize=(8*cm, 4*cm))
    clf()
    NIMS=3 # Number of sample images to select

    plotax = gcf().subplot_mosaic("""
                0AAAAA
                0AAAAA
                1AAAAA
                1AAAAA
                2AAAAA
                2AAAAA
            """)

    sca(plotax['A'])

    # Now select N points far from each other
    pts_list = [random.randint(0,len(results_map)-1)]
    xy = res[:,0:2]

    for _ in range(NIMS-1):
        inds = torch.tensor(pts_list)
        selected = xy[inds, :]
        N = len(pts_list)
        M = len(xy)
        furthest = ((xy.unsqueeze(0).expand(N, M, 2) - selected.unsqueeze(1).expand(N, M, 2))**2).sum(2).min(0).values.max(0).indices.item()
        pts_list.append(furthest)


    _, ind = xy[pts_list,0].sort()
    pts_list = list(torch.tensor(pts_list)[ind])

    #copy those points to the end of the list, so they appear on top

    res_end = res[pts_list, :]
    res = torch.cat((res, res_end), 0)


    scatter(res[:,0], res[:,1], c=res[:,3]*spacing, cmap='inferno', vmin=35, vmax=80, s=5)
    axis('square')
    limits = plt.axis()
    ax_cx = (limits[1]+limits[0])/2
    ax_cy = (limits[3]+limits[2])/2
    axis((ax_cx - plot_size/2, ax_cx + plot_size/2, ax_cy - plot_size/2, ax_cy + plot_size/2))

    scale_x0 = plt.axis()[1] - 10000
    scale_x1 = plt.axis()[1]
    scale_y = plt.axis()[3] - 5000


    plot([scale_x0, scale_x1], [scale_y, scale_y], 'k', linewidth=5)
    text(scale_x0 + 500, scale_y+1000, '10$\\mu$m', fontsize=FS)
    axis('off')
    cb = colorbar()
    cb.set_label(label='Spacing (nm)', size=FS)
    cb.ax.tick_params(labelsize=FS)
    #cb.vmin = 3 #  35.857139913762694
    #cb.vmax = 100# 79.53197165614256

    #embed()
    scatter(xy[pts_list, 0], xy[pts_list, 1], facecolors='none', edgecolors='g', s=9, linewidth=0.5)

    for n, i in enumerate(pts_list):
        sca(plotax[str(n)])
        
        input_img = (dataset[results_map[i]][2] + dataset[results_map[i]][3]).squeeze().cpu()
        output_img = (recons[i][2] + recons[i][3]).squeeze().cpu()

        input_img /= input_img.sum()
        output_img /= output_img.sum()

        imshow(torch.cat((input_img, output_img), 0), cmap='gray')
        title(f"{res[i,3].item()*spacing.item():.3} nm", pad=-10, fontsize=FS)
        axis('off')

    gcf().tight_layout()
    pause(.1)

    transFigure = gcf().transFigure.inverted()
    lines = []
    for n, i in enumerate(pts_list):

        coord1 = transFigure.transform(plotax['A'].transData.transform(xy[i,:]))
        coord2 = transFigure.transform(plotax[str(n)].transData.transform([64,32]))
        lines.append(matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]), transform=gcf().transFigure))

    gcf().lines = lines


    pause(.1)

    return res_orig, spacing.item(), net

resi_run_dir = Path("sample_logs/1711985336-4d7cc96effb6e4740278bd39261837986110b4a2/run-000-phase_1")
resi_data3d, resi_means = resi_data.load_3d_with_means()
random.seed(29)
res_resi, spacing_resi, net_resi = process_data_and_plot_some_crap(resi_data3d, resi_means, resi_run_dir)

pause(.1)
pause(.1)
plt.savefig('hax/figure2_resi.svg', format='svg')
plt.close('all')
pause(.1)


bates_run_dir = Path("sample_logs/1724184869-5e1a100264c656834ccb69bf44e0f261b99a4612/run-000-phase_1/")
stuff = mark_bates_data.load_3d_list_and_means()
bates_data3d, bates_means = stuff[0][0], stuff[1][0]

random.seed(4)
res_bates, spacing_bates, net_bates = process_data_and_plot_some_crap(bates_data3d, bates_means, bates_run_dir)

pause(.1)
pause(.1)
plt.savefig('hax/figure2_bates.svg', format='svg')
plt.close('all')
pause(.1)

plt.close('all')

resi_col=[.5, .5, 1]
bates_col=[1, .5, 0]
plt.subplots(figsize=(8.0*cm, 4.0*cm))
clf()
scatter(res_resi[:,2] - res_resi[:,2].mean(), res_resi[:,3]*spacing_resi, color=[*resi_col, .5], edgecolor=[0,0,0,0.])
scatter(res_bates[:,2] - res_bates[:,2].mean(), res_bates[:,3]*spacing_bates, color=[*bates_col, .5], edgecolor=[0,0,0,0.])
plt.xlabel('Z (nm)', fontsize=FS)
plt.ylabel('Spacing (nm)', fontsize=FS)
plt.gca().tick_params(labelsize=FS)
plt.gca().legend(['RESI', '4Pi STORM'], loc='lower right', fontsize=FS)
leg = plt.gca().get_legend()
leg.legend_handles[0].set_facecolor(resi_col)
leg.legend_handles[1].set_facecolor(bates_col)
plt.tight_layout()
plt.savefig('hax/figure2_z_correlation.svg', format='svg')


#of_evil = plt.gca().axis()
cr, pr = scipy.stats.pearsonr(res_resi[:,2] - res_resi[:,2].mean(), res_resi[:,3]*spacing_resi)
#mrs =  (res_resi[:,3]*spacing_resi).mean()
#plot([-1000, 1000], [cr*-1000+mrs, cr*1000+mrs], color=resi_col)
cb, pb = scipy.stats.pearsonr(res_bates[:,2] - res_bates[:,2].mean(), res_bates[:,3]*spacing_bates)
#mbs =  (res_resi[:,3]*spacing_resi).mean()
#plot([-1000, 1000], [cb*-1000+mbs, cb*1000+mbs], color=bates_col)
#plt.gca().axis(of_evil)

plt.close('all')

plt.subplots(figsize=(4.0*cm, 4.0*cm))

clf()
r_count, r_bins = np.histogram(res_resi[:,3]*spacing_resi, 20)
b_count, b_bins = np.histogram(res_bates[:,3]*spacing_bates, r_bins-.0)
plt.stairs(r_count/r_count.sum(), r_bins, color=[*resi_col, .5], fill=True)
plt.stairs(b_count/b_count.sum(), b_bins, color=[*bates_col, .5], fill=True)
plt.stairs(r_count/r_count.sum(), r_bins, color=[*resi_col], fill=False, linewidth=3)
plt.stairs(b_count/b_count.sum(), b_bins, color=[*bates_col], fill=False, linewidth=3)
plt.xlabel('Spacing (nm)', fontsize=FS)
plt.gca().tick_params(labelsize=FS)
plt.yticks([])
plt.ylabel('Relative count (au)      ', fontsize=FS)
plt.tight_layout()
plt.savefig('hax/figure2_historgram.svg', format='svg')
plt.close('all')


# Now scale the models by their size and save as ply files
pts, weights = (i.detach().cpu() for i in net_resi.get_model())
pts.requires_grad = False
weights.requires_grad = False
spacing = res_resi[:,3].median().unsqueeze(0).detach().cpu().float()
expand = res_resi[:,4].median().unsqueeze(0).detach().cpu().float()
S = scale_along_axis_and_expand_matrix(net_resi._parameterisation.get_axis().cpu(), spacing, expand).squeeze(0)
S=S.detach()
S.requires_grad = False
pts = trn(S@trn(pts))
save_ply.save_pointcloud_as_mesh("hax/figure2_resi_3d.ply", pts, weights, 2.0, 0.1, 100)



pts, weights = (i.detach().cpu() for i in net_bates.get_model())
pts.requires_grad = False
weights.requires_grad = False
spacing = res_bates[:,3].median().unsqueeze(0).detach().cpu().float()
expand = res_bates[:,4].median().unsqueeze(0).detach().cpu().float()
S = scale_along_axis_and_expand_matrix(net_bates._parameterisation.get_axis().cpu(), spacing, expand).squeeze(0)
S=S.detach()
S.requires_grad = False
pts = trn(S@trn(pts))
save_ply.save_pointcloud_as_mesh("hax/figure2_bates_3d.ply", pts, weights, 3.5, 0.2, 100)

