from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import train
import device
from matrix import trn
from network import GeneralPredictReconstruction
from localisation_data import LocalisationDataSetMultipleDan6
from train_nupc import PredictReconstruction, AxialStretchRadialExpandWithGeneralShift


## Standard parameters for the end of an NUPC run
data_parameters = train.DataParametersXYYZ(
    image_size_xy = 64,
    image_size_z = 32,
    nm_per_pixel_xy = 3.9,
    z_scale = 2
)
final_fwhm = 13.0


# General parameters for figure plotting
# Plotting with everything twice the size improves the figure a bit
# because the margins don't get scaled up, so the final figure looks
# a bit tighter and wastes less space 
FIGSCALE=2
cm = FIGSCALE/2.54  # centimeters in inches, plus an overall figure scaling
FS=7*FIGSCALE


def _load_net(nupc3d: list[Tensor], trained_weights: dict, pts:int)->tuple[GeneralPredictReconstruction, AxialStretchRadialExpandWithGeneralShift]:
    net, parameterisation = PredictReconstruction(initial_model_size=pts, final_model_size=pts, **vars(data_parameters), data=nupc3d)
    parameterisation.per_point_shift=True
    
    # _orig is an artifact of how torch.compile() works. The emitted network references
    # the underlying parameters by prepending 10 characters of stuff. Strip them here since
    # sometimes I forgot to re-enable torch.compile so not every saved net has these.
    if "_orig" in next(iter(trained_weights.keys())):
        trained_weights = { k[10:]:v for k,v in trained_weights.items()}

    # During the development, shift_network went from private to public, so this 
    # code allows loading old runs with the private shift network
    trained_weights = { k.replace("_shift_network", "shift_network"):v for k,v in trained_weights.items()}

    net.load_state_dict(trained_weights)
    
    # Switch off everything related to training here
    net.eval()
    for i in net.parameters():
        i.requires_grad=False
        
    return net, parameterisation

def _cov(points: Tensor)->Tensor:
    assert len(points.shape)==3
    pts_centred = points - points.mean(1).unsqueeze(1).expand_as(points)
    return torch.einsum('hij,hik->hjk', pts_centred, pts_centred)/pts_centred.shape[1]

def _primary_axis_angle(cov: Tensor)->Tensor:
    assert len(cov.shape)==3
    assert cov.shape[1] == 2
    assert cov.shape[2] == 2
    val, vec =torch.linalg.eigh(cov)  # pylint: disable=not-callable

    assert (val[:,1] >= val[:,0]).all()
    
    major_axis = vec[:,:,1]
    major_axis *= major_axis[:,1].sign().unsqueeze(1).expand(-1, 2)
    return torch.atan2(major_axis[:,1], major_axis[:,0])


@dataclass
class _NetRes:
    points: Tensor
    indices: list[int]
    images: list[list[Tensor]]
    data: list[list[Tensor]]




def _apply_net(nupc3d: list[Tensor], trained_weights: dict, pts:int=700)->_NetRes:

    final_sigma = train.fwhm_to_sigma(final_fwhm)
    final_sigma_t = torch.tensor(final_sigma)

    dataset = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)

    net, parameterisation = _load_net(nupc3d, trained_weights, pts)
    net.to(device.device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataset.set_sigma(final_sigma)

    R=4
    C=6
    #plt.ion()

    pts_list = []
    ind_list = []
    img_list = []
    data_list = []

    plot=False

    with torch.no_grad():
        for index,datum in enumerate(tqdm(loader)):

            imgs, _, _, _ = net.forward(datum, final_sigma_t)
            t, r, _, is_valid, parameters = net.process_input(datum, final_sigma_t)
            batch_size = t.shape[0]

            # Apply the parameterisation
            points, _, _ = parameterisation(*net.get_model(), parameters)
            
            # Note that the parameterisation can change the number of points.
            Nv = points.shape[1]
            t_per_point = t.unsqueeze(1).expand(batch_size, Nv, 3)
            
            # Rotate and shift resulting aggregate
            points = trn(r @ trn(points)) + t_per_point

            points = points.squeeze().cpu()

            if is_valid > 0.5:
                pts_list.append(points.cpu())
                ind_list.append(index)
                img_list.append(imgs)
                data_list.append(datum)

                # cov = vec @ val.diag() @ trn(vec)

                if plot:
                    plt.clf()
                    plt.suptitle(f'Validity = {is_valid.item():0.3}')
                    for n, i in enumerate(datum):
                        plt.subplot(R, C, n+1)
                        plt.imshow(i[0,0].cpu(), cmap='grey')
                        plt.axis('off')
                
                    reconstruction, _, _, _ = net(datum, final_sigma_t) 
                    for n, i in enumerate(reconstruction):
                        plt.subplot(R, C, C + n+1)
                        plt.imshow(i[0,0].cpu(), cmap='grey')
                        plt.axis('off')
                    
                    for n, i in enumerate(train._normalized_difference(datum,reconstruction)): # pylint: disable=protected-access
                        plt.subplot(R, C, 2*C + n+1)
                        plt.imshow(i[0,0].cpu(), cmap='grey')
                        plt.axis('off')

                    plt.subplot(R, C, 3*C + 1)
                    plt.imshow(datum[0][0,0].cpu(), cmap='grey')
                    pts_px = points[:,0:2] / data_parameters.nm_per_pixel_xy + data_parameters.image_size_xy/2
                    plt.scatter(pts_px[:,0], pts_px[:,1], c=[.5, .5, 1, .02])

                    #center_2d_px = pts_px.mean(0)
                    #
                    #for i in [0,1]:
                    #    v = vec[:,i] * val[i].sqrt() / data_parameters.nm_per_pixel_xy
                    #    v = torch.stack([-v, v], 0) + center_2d_px.unsqueeze(0).expand(2,2)
                    #    plt.plot(*v.permute(1,0))


                    #plt.axis('square')

                    plt.subplot(R, C, 3*C + 3, projection='3d')
                    plt.gca().scatter(*trn(points.cpu()))
                    plt.axis('square')
                    plt.tight_layout()
                    plt.pause(.1)
                    #plt.waitforbuttonpress()

    return _NetRes(
        points=torch.stack(pts_list, 0), 
        indices=ind_list, 
        images=img_list,
        data=data_list,
    )




def _plot_angular(good_means: Tensor, inp: _NetRes)->None:

    # This is a figure size that's big enough for the RESI data.
    # Use it for both to they are a consistent size
    plot_size = 31397.046857833866
    
    angs = _primary_axis_angle(_cov(inp.points[:,:,0:2]))
    eigs, _ = torch.linalg.eigh(_cov(inp.points[:,:,0:2])) # pylint: disable=not-callable
    
    plt.subplots(figsize=(8*cm, 4*cm))
    plt.clf()
    plotax = plt.gcf().subplot_mosaic("""
                0AAAAA
                0AAAAA
                1AAAAA
                1AAAAA
                2AAAAA
                2AAAAA
            """)


    # Main scatter plot
    plt.sca(plotax['A'])
    plt.scatter(*good_means[:,0:2].permute(1,0), c=angs, cmap='twilight', s=5) # type: ignore[misc]
    plt.axis('equal')
    limits = plt.axis()
    ax_cx = (limits[1]+limits[0])/2 + 12000 # shift left a bit
    ax_cy = (limits[3]+limits[2])/2
    plt.axis((ax_cx - plot_size/2, ax_cx + plot_size/2, ax_cy - plot_size/2, ax_cy + plot_size/2))

    plt.axis('off')
    
    # Draw a scalebar
    scale_x0 = plt.axis()[1] - 10000
    scale_x1 = plt.axis()[1]
    scale_y = plt.axis()[3] - 5000
    plt.plot([scale_x0, scale_x1], [scale_y, scale_y], 'k', linewidth=5)
    plt.text(scale_x0 + 500, scale_y+1000, '10$\\mu$m', fontsize=FS)

    plt.tight_layout()


    #Create angulat histogram as an inset plot
    plt.subplot(3,3,1, projection='polar')
    # x, y, w, h
    plt.gca().set_position(matplotlib.transforms.Bbox.from_bounds(.66, .3, .25, .25))

    # Create angular histogram
    counts, edges = np.histogram(angs, 20)
    for i, count in enumerate(counts):
        midangle = sum(edges[i:i+2])/2
        color = matplotlib.cm.twilight(midangle/torch.pi) # type: ignore[attr-defined] # pylint: disable=no-member
        plt.fill_between([edges[i], edges[i], edges[i+1], edges[i+1]], [0, count, count, 0], 0, color=color)
        plt.fill_between([edges[i]+torch.pi, edges[i]+torch.pi, edges[i+1]+torch.pi, edges[i+1]+torch.pi], [0, count, count, 0], 0, color=color)

    plt.xticks(fontsize=FS)
    plt.yticks(range(0,70,20), [])

    # Show some image clips
    # Sort by aspect ratio to find some extreme examples
    _, inds = (eigs[:,1] / eigs[:,0]).sort(descending=True)
    
    # Pick the top 20
    best: list[int] = list(inds[0:20])
    
    # Pick 3 from them
    selection = torch.tensor(random.sample(best, 3))

    # Now sort by y
    _, y_inds = good_means[selection,1].sort(descending=True)
    selection = selection[y_inds]

    for i in [0,1,2]:
        plt.sca(plotax[str(i)])
        plt.imshow((inp.data[selection[i]][0]+inp.data[selection[i]][1]).cpu().squeeze(), cmap='gray', origin='lower')
        plt.title(f'{angs[selection[i]]*180/torch.pi:.0f}°')
        plt.axis('off')
        

    plt.pause(.1)
    # This hacking here draws lines from the image clip to the point that the clip corresponds to.
    transFigure = plt.gcf().transFigure.inverted()
    lines = []
    for n, i in enumerate(selection):

        coord1 = transFigure.transform(plotax['A'].transData.transform(good_means[i,0:2]))
        coord2 = transFigure.transform(plotax[str(n)].transData.transform([64,32])) # 64,32 is midway down the right hand edge, since the image is 64x64
        lines.append(matplotlib.lines.Line2D((coord1[0],coord2[0]),(coord1[1],coord2[1]), transform=plt.gcf().transFigure))

    plt.gcf().lines = lines



    #edge1_x = math.cos(edges[i]) * count
    #edge1_y = math.sin(edges[i]) * count
    #edge2_x = math.cos(edges[i+1]) * count
    #edge2_y = math.sin(edges[i+1]) * count
#
#    plt.plot([0, edge1_x, edge2_x, 0], [0, edge1_y, edge2_y, 0])
#    plt.plot([0, -edge1_x, -edge2_x, 0], [0, -edge1_y, -edge2_y, 0])



nupc3d_resi, nupc3d_resi_means = resi_data.load_3d_with_means()
nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]
trained_weights_resi = torch.load('log/1766516868-66b60604c41adb3c784b829cbd0205da1b12c1cd/phase_2/final_net.zip', map_location=torch.device('cpu'))
res_resi = _apply_net(nupc3d_resi, trained_weights_resi)
good_means_resi = torch.stack(nupc3d_resi_means)[res_resi.indices,:]

plt.close('all')
random.seed(11)
_plot_angular(good_means_resi, res_resi)
plt.savefig('tmp/resi_angular.svg')


I=0

nupc3d_bates, nupc3d_bates_means = mark_bates_data.load_3d_list_and_means()
trained_weights_bates = torch.load('log/1766605809-a396351dc2c407f97a32efa35b421a9aa8d2de55/phase_2/final_net.zip', map_location=torch.device('cpu'))
res_bates = _apply_net(nupc3d_bates[I], trained_weights_bates)
good_means_bates = torch.stack(nupc3d_bates_means[I])[res_bates.indices,:]

plt.close('all')
random.seed(3)
_plot_angular(good_means_bates, res_bates)
plt.savefig('tmp/bates_angular.svg')

#plt.clf()
#plt.subplot(1,2,1)
#plt.scatter(*good_means_bates[:,0:2].permute(1,0), c=angs_bates, cmap='twilight')
#plt.subplot(1,2,2)
#plt.hist(angs_bates * 180 / torch.pi, 20)
