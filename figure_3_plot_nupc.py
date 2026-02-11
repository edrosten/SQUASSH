from dataclasses import dataclass
from pathlib import Path
import math
import random

import cv2
import numpy as np
import scipy
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


import resi_data
import mark_bates_data
import train
import device
import save_ply
import render
from matrix import trn, euler
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

# Batched covariance matrix
def _cov(points: Tensor)->Tensor:
    assert len(points.shape)==3
    pts_centred = points - points.mean(1).unsqueeze(1).expand_as(points)
    return torch.einsum('hij,hik->hjk', pts_centred, pts_centred)/pts_centred.shape[1]

# Assuming a 2D cov matrix, the angle (normalized to 0-180 of the
# vector corresponding to the largest eigenvalue
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
class _PCAResult:
    S: Tensor
    Vh: Tensor
    stddev: Tensor
    centre: Tensor


def _PCA(points:Tensor)->_PCAResult:
    n_data = points.shape[0]
    flat_pts =points.reshape(n_data, -1)

    flat_pts_centred = flat_pts - flat_pts.mean(0).unsqueeze(0).expand(n_data, -1)
    (_, S, Vh_vectors) = torch.linalg.svd(flat_pts_centred, full_matrices=False) # pylint: disable=not-callable

    # Covariances are S^2 / (n-1)
    # standard devs are S/sqrt(n-1)
    stddev = S / (math.sqrt(n_data)-1)
    centre = flat_pts.mean(0).reshape(-1, 3)
    Vh_vectors = Vh_vectors.reshape(Vh_vectors.shape[0], *centre.shape)

    return _PCAResult(S=S,Vh=Vh_vectors,stddev=stddev,centre=centre)



@dataclass
class _NetRes:
    points_img: Tensor
    points_model: Tensor
    indices: list[int]
    images: list[list[Tensor]]
    data: list[list[Tensor]]
    net: GeneralPredictReconstruction
    parameterisation: AxialStretchRadialExpandWithGeneralShift


def _apply_net(nupc3d: list[Tensor], trained_weights: dict, pts:int=700)->_NetRes:

    final_sigma = train.fwhm_to_sigma(final_fwhm)
    final_sigma_t = torch.tensor(final_sigma)

    dataset = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)

    net, parameterisation = _load_net(nupc3d, trained_weights, pts)
    net.to(device.device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataset.set_sigma(final_sigma)

    pts_img_list = []
    pts_model_list = []
    ind_list = []
    img_list = []
    data_list = []

    with torch.no_grad():
        for index,datum in enumerate(tqdm(loader)):

            imgs, _, _, _ = net.forward(datum, final_sigma_t)
            t, r, _, is_valid, parameters = net.process_input(datum, final_sigma_t)
            batch_size = t.shape[0]

            # Apply the parameterisation
            points_orig, _, _ = parameterisation(*net.get_model(), parameters)
            
            # Note that the parameterisation can change the number of points.
            Nv = points_orig.shape[1]
            t_per_point = t.unsqueeze(1).expand(batch_size, Nv, 3)
            
            # Rotate and shift resulting aggregate
            points_img = trn(r @ trn(points_orig)) + t_per_point

            points_img = points_img.squeeze().cpu()

            if is_valid > 0.5:
                pts_img_list.append(points_img.cpu())
                pts_model_list.append(points_orig.squeeze().cpu())
                ind_list.append(index)
                img_list.append(imgs)
                data_list.append(datum)

    return _NetRes(
        points_img=torch.stack(pts_img_list, 0), 
        points_model=torch.stack(pts_model_list, 0), 
        indices=ind_list, 
        images=img_list,
        data=data_list,
        net=net,
        parameterisation=parameterisation
    )

def _rotation_ring_to_xy(res: _NetRes)->Tensor:
    # Model has the main axis aligned with the stretch, rotate so it's the Z axis
    return euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ res.parameterisation.get_R().cpu()

def _segment_top_ring(res: _NetRes)->Tensor:
    baseline_model = res.net.get_model()[0].cpu()
    R = _rotation_ring_to_xy(res)
    return (R @ baseline_model.permute(1,0)).permute(1,0)[:,2] > 0



def _print_ring_size_ratio_top_bottom(res: _NetRes)->None:
    top_mask = _segment_top_ring(res)
    pts_xy = (res.points_model @ trn(_rotation_ring_to_xy(res)))[:,:,0:2]


    cov_top = _cov(pts_xy[:,top_mask,:])
    cov_bot = _cov(pts_xy[:,top_mask.logical_not(),:])

    covs = torch.stack([cov_top, cov_bot], 1)

    # Std dev (variance) as trace of covariance matrix, equivlaent to RMS radius
    stds = covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).sqrt()

    principal_axes=torch.linalg.eigvalsh(covs).sqrt() # pylint: disable=not-callable
    ratio = principal_axes[...,0]/principal_axes[...,1]


    print("Size top / bottom ± at 1σ")
    for i in [0,1]:
        print(f"{stds[:,i].mean().item():0.3} ± {(stds[:,i].var()/stds.shape[0]).sqrt().item():0.1}   ", end="")
    print("\n")

    print("Aspect ratio top/bottom")
    for i in [0,1]:
        print(f"{ratio[:,i].mean().item():0.3} ± {(ratio[:,i].var()/ratio.shape[0]).sqrt().item():0.1}   ", end="")

    print("")

def _nn_graph(pts: Tensor)->tuple[Tensor, Tensor]:
    # Calculate the nearest neighbour graph.
    # where all 32 points were fitted correctly.

    # Simple quadratical method. Only 32 points and we have a GPU
    npts = pts.shape[0]
    
    pairwise_distance = (pts.unsqueeze(0).expand(npts, *pts.shape) - pts.unsqueeze(1).expand(npts, *pts.shape)).pow(2).sum(-1).sqrt() + torch.eye(npts)*1e10
    
    min_dist, index_of_closest = pairwise_distance.min(1)
    
    # Only keep neighbours closer than 25nm. This should capture all doublets
    good_points_mask = min_dist < 25.0
    
    plt.subplot(1,1,1,projection='3d')
    plt.gca().scatter(*pts.permute(1,0))
    for i in torch.arange(npts)[good_points_mask]:
        p1 = pts[i] 
        p2 = pts[index_of_closest[i]]
        ps = torch.stack([p1, (p1+p2)/2, p2], 0)
        plt.gca().plot(*ps.permute(1,0)[:,0:2], c=plt.cm.winter(0.0)) #type: ignore[attr-defined] # pylint: disable=no-member
        plt.gca().plot(*ps.permute(1,0)[:,1:3], c=plt.cm.winter(1.0)) #type: ignore[attr-defined] # pylint: disable=no-member

    plt.title('Check NN assignment')

    return index_of_closest, good_points_mask

def _std_and_ratio(covs: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    # Std dev (variance) as trace of covariance matrix, equivlaent to RMS radius
    assert covs.shape[-1] == 2
    assert covs.shape[-2] == 2
    stds = covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).sqrt()

    principal_axes=torch.linalg.eigvalsh(covs).sqrt() # pylint: disable=not-callable
    ratios = principal_axes[...,0]/principal_axes[...,1]
    return stds, ratios

def _plot_distance_vs_eccentricity(res: _NetRes)->None:

    assert res.points_model.shape[1] == 32, "This only works with 32 point models"

    baseline_model = res.net.get_model()[0].cpu()

    plt.figure()
    index_closest, good_mask = _nn_graph(baseline_model)
    assert good_mask.all(), "Honestly this has not been tested with slightly incomplete models"

    R = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ res.parameterisation.get_R().cpu()

    pts = res.points_model @ R.permute(1,0).unsqueeze(0).expand(res.points_model.shape[0], 3, 3)
    closest = pts[:, index_closest, :]
    distances = (pts-closest).pow(2).sum(-1).sqrt()
    

    top_mask = (R @ baseline_model.permute(1,0)).permute(1,0)[:,2] > 0
    bot_mask = top_mask.logical_not()

    plt.figure()
    plt.subplot(1,1,1,projection='3d')
    plt.gca().scatter(*baseline_model[top_mask].permute(1,0))
    plt.gca().scatter(*baseline_model[bot_mask].permute(1,0))
    plt.title('Check top/bottom segmentation')

    std_dist_top = distances[:,top_mask].std(1)
    std_dist_bot = distances[:,bot_mask].std(1)

    cov_top = _cov(pts[:,top_mask,0:2])
    cov_bot = _cov(pts[:,bot_mask,0:2])

    _, ratio_top = _std_and_ratio(cov_top)
    _, ratio_bot = _std_and_ratio(cov_bot)
       
    
    for ind, fit in enumerate(pts):
        el_top = torch.tensor(cv2.fitEllipse(fit[top_mask,0:2].numpy())[1])
        el_bot = torch.tensor(cv2.fitEllipse(fit[bot_mask,0:2].numpy())[1])
        ratio_top[ind] = el_top.min()/el_top.max()
        ratio_bot[ind] = el_bot.min()/el_bot.max()



    eccentricity_top = (1-ratio_top**2).sqrt()
    eccentricity_bot = (1-ratio_bot**2).sqrt()
       
    #mean_dist_top = distances[:,top_mask].mean(1)
    #mean_dist_bot = distances[:,bot_mask].mean(1)

    def _to_pretty_sci(x: float)->str:
        superscripts = "⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹"
        normal       = "+-0123456789"
        mapping=dict(zip(normal, superscripts))
        x_str = ("%.1E"%x).split("E") # pylint: disable=consider-using-f-string
        print(x_str)
        return x_str[0] + "×10" + "".join([mapping[i] for i in x_str[1]])



    top_stats = scipy.stats.pearsonr(eccentricity_top, std_dist_top)
    bot_stats = scipy.stats.pearsonr(eccentricity_bot, std_dist_bot)
    plt.subplots(figsize=(8*cm, 4*cm))
    plt.clf()
    plt.scatter(eccentricity_top, std_dist_top, label=f"NR r={top_stats.statistic:0.2}, p={_to_pretty_sci(top_stats.pvalue)}")
    plt.scatter(eccentricity_bot, std_dist_bot, label=f"CR r={bot_stats.statistic:0.2}, p={_to_pretty_sci(bot_stats.pvalue)}")
    plt.xlabel('Eccentricity', fontsize=FS)
    plt.ylabel('Standard deviation\nof doublet spacing', fontsize=FS)
    plt.xticks(fontsize=FS)
    plt.yticks(fontsize=FS)
    plt.legend(fontsize=FS)
    plt.tight_layout()
    plt.pause(.1)
    plt.savefig('tmp/fig3_doublet_spacing_variance_vs_eccentricity.svg')


def _primary_axis_angle_elfit(pts: Tensor)->Tensor:
    v=[]
    a=[]
    for fit in pts:
        el = cv2.fitEllipse(fit.clone().numpy())
        v.append(el[1])
        a.append(el[2])
        # centre [xy]
        # radii [a,b] (usually a<b)
        # angle of a, want angle of b
    vt = torch.tensor(v)
    assert (vt[:,0]<=vt[:,1]).all()
    
    ang_b = (torch.tensor(a) + 90.0)%180
    return ang_b * torch.pi / 180


def _plot_angular(good_means: Tensor, inp: _NetRes)->None:

    # This is a figure size that's big enough for the RESI data.
    # Use it for both to they are a consistent size
    plot_size = 31397.046857833866
    
    angs = _primary_axis_angle(_cov(inp.points_img[:,:,0:2]))
    #angs = _primary_axis_angle_elfit(inp.points_img[:,:,0:2])
    eigs, _ = torch.linalg.eigh(_cov(inp.points_img[:,:,0:2])) # pylint: disable=not-callable
    
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


    #Create angular histogram as an inset plot
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



def _pca_figure(res: _NetRes)->None:
    # Flip X and Z axes
    R = _rotation_ring_to_xy(res)

    intensities = res.net.get_model()[1].cpu().detach()


    pca = _PCA(res.points_model)
    centre = pca.centre
    Vh = pca.Vh
    centre = pca.centre
    stddev = pca.stddev


    _, darkest_first = intensities.sort()
    intensities = intensities[darkest_first]
    centre = centre[darkest_first,:]
    Vh = Vh[:, darkest_first, :]


    top_mask = (R @ centre.permute(1,0)).permute(1,0)[:,2] > 0

    plt.subplots(figsize=(8*cm, 4*cm))
    
    N=3 
    alpha=0.1
    plt.clf()
    for I in range(3):
        component = Vh[I]*stddev[I]*5

        plt.subplot(2,N,I+1)
        plt.scatter(*(R @ (centre          )[top_mask,:].permute(1,0))[0:2,:], c=intensities[top_mask], alpha=alpha, cmap='Greys', edgecolors='none', clip_on=False)  # type: ignore[misc]
        plt.scatter(*(R @ (centre+component)[top_mask,:].permute(1,0))[0:2,:], c=intensities[top_mask], alpha=alpha, cmap='Oranges', edgecolors='none', clip_on=False)  # type: ignore[misc]
        plt.xlabel(f'{"" if I > 0 else "PCA Component "}{I+1}', fontsize=FS)
        plt.axis('square')
        plt.axis((-65,65,-65,65))
        for line in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[line].set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().xaxis.set_label_position('top')
        if I == 0:
            plt.ylabel('NR', fontsize=FS)

        plt.subplot(2,N,I+1+N)
        plt.scatter(*(R @ (centre          )[top_mask.logical_not(),:].permute(1,0))[0:2,:], c=intensities[top_mask.logical_not()], alpha=alpha, cmap='Greys', edgecolors='none', clip_on=False)  # type: ignore[misc]
        plt.scatter(*(R @ (centre+component)[top_mask.logical_not(),:].permute(1,0))[0:2,:], c=intensities[top_mask.logical_not()], alpha=alpha, cmap='Oranges', edgecolors='none', clip_on=False)  # type: ignore[misc]
        plt.axis('square')
        plt.axis((-65,65,-65,65))
        for line in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[line].set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        if I == 0:
            plt.ylabel('CR', fontsize=FS)
    
    plt.tight_layout()
    plt.pause(.1)




def _pca_video(res: _NetRes, name:str)->None:
    # Flip X and Z axes
    R = _rotation_ring_to_xy(res).to(device.device)

    intensities = res.net.get_model()[1].detach().to(device.device)


    pca = _PCA(res.points_model)
    Vh = pca.Vh.to(device.device)
    centre = pca.centre.to(device.device)
    stddev = pca.stddev.to(device.device)


    _, darkest_first = intensities.sort()
    intensities = intensities[darkest_first]
    centre = centre[darkest_first,:]
    Vh = Vh[:, darkest_first, :]

    top_mask = (R @ centre.permute(1,0)).permute(1,0)[:,2] > 0
    bot_mask = top_mask.logical_not()

    vid = Path(f'tmp/pca_video_{name}_advanced_3s')
    vid.mkdir()

    for j, i in enumerate(tqdm(torch.arange(0, 1, 1/240))):
        pos = torch.sin(i*torch.pi*2).item()
        
        plt.clf()
        for c in range(3):

            nm_per_pix = 0.25*4
            pixels = 600//4
            sigma=torch.tensor([2.0], device=device.device)
            pts_aligned = (centre + pos * 3 * stddev[c] * Vh[c])@trn(R)

            top = render.render_batch_weights(pts_aligned[top_mask].unsqueeze(0)[...,0:2], sigma, intensities[top_mask].unsqueeze(0), nm_per_pix, pixels)[0] 
            bot = render.render_batch_weights(pts_aligned[bot_mask].unsqueeze(0)[...,0:2], sigma, intensities[bot_mask].unsqueeze(0), nm_per_pix, pixels)[0]

            plt.subplot(2,3,c+1)
            plt.imshow(top.cpu(), cmap='hot')
            plt.axis('off')
            plt.subplot(2,3,c+1+3)
            plt.imshow(bot.cpu(), cmap='hot')
            plt.axis('off')
        plt.tight_layout()
        plt.pause(.1)
        plt.pause(.1)
        plt.pause(.1)
        plt.savefig(vid/f'{j:05}.png')



nupc3d_resi, nupc3d_resi_means = resi_data.load_3d_with_means()
nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]

trained_weights_resi_32 = torch.load('sample_logs/1767449867-0b3ce320f9213553e0b7d942d407268e8c3db4a4/phase_2/final_net.zip', map_location=torch.device('cpu'))
res_resi_32 = _apply_net(nupc3d_resi, trained_weights_resi_32, 32)
good_means_resi_32 = torch.stack(nupc3d_resi_means)[res_resi_32.indices,:]
_plot_distance_vs_eccentricity(res_resi_32)
plt.savefig('tmp/fig3_res32_spacing_v_eccentricity.svg')


trained_weights_resi = torch.load('sample_logs/1766516868-66b60604c41adb3c784b829cbd0205da1b12c1cd/phase_2/final_net.zip', map_location=torch.device('cpu'))
res_resi = _apply_net(nupc3d_resi, trained_weights_resi)
good_means_resi = torch.stack(nupc3d_resi_means)[res_resi.indices,:]

plt.close('all')
random.seed(11)
_plot_angular(good_means_resi, res_resi)
plt.savefig('tmp/fig3_resi_angular.svg')


plt.close('all')
_pca_figure(res_resi)
plt.savefig('tmp/fig3_resi_pca.svg')


CellNo=0

nupc3d_bates, nupc3d_bates_means = mark_bates_data.load_3d_list_and_means()
trained_weights_bates = torch.load('sample_logs/1766605809-a396351dc2c407f97a32efa35b421a9aa8d2de55/phase_2/final_net.zip', map_location=torch.device('cpu'))
res_bates = _apply_net(nupc3d_bates[CellNo], trained_weights_bates)
good_means_bates = torch.stack(nupc3d_bates_means[CellNo])[res_bates.indices,:]

plt.close('all')
random.seed(3)
_plot_angular(good_means_bates, res_bates)
plt.savefig('tmp/fig3_bates_angular.svg')

#_pca_video(res_bates, 'bates')
#_pca_video(res_resi, 'resi')


print("RESI data")
print("---------")
_print_ring_size_ratio_top_bottom(res_resi)
print("")


print("Bates data")
print("----------")
print("")
_print_ring_size_ratio_top_bottom(res_bates)


def _render(pts: torch.Tensor, weights: None|Tensor)->tuple[Tensor, Tensor]:
    top_mask = pts[:,2] > 0
    bot_mask = top_mask.logical_not()
    if weights is None:
        weights = torch.ones(top_mask.shape[0])
    weights = weights.to(pts)
    sigma = torch.tensor([2.0]).to(pts)
    nm_per_pix = 0.25

    top = render.render_batch_weights(pts[top_mask].unsqueeze(0)[...,0:2], sigma, weights[top_mask].unsqueeze(0), nm_per_pix, 600)[0] 
    bot = render.render_batch_weights(pts[bot_mask].unsqueeze(0)[...,0:2], sigma, weights[bot_mask].unsqueeze(0), nm_per_pix, 600)[0] 

    return top, bot

def _renderings(ind: int, data: list[Tensor], res: _NetRes)->None:

    pts_img = res.points_img[ind]
    pts_data = data[res.indices[ind]]
    
    
    dat_t, dat_b = [ i.cpu(). numpy() for i in _render(pts_data.to(device.device), None)]
    mod_t, mod_b = [ i.cpu().numpy() for i in  _render(pts_img.to(device.device), res.net.get_model()[1].to(device.device)) ]

    plt.imsave(f'tmp/figure3_{ind}_data_top.png', (matplotlib.cm.hot(dat_t/dat_t.max())*255.9).astype(np.uint8)[...,0:3])  # type: ignore[attr-defined]  # pylint: disable=no-member
    plt.imsave(f'tmp/figure3_{ind}_data_bot.png', (matplotlib.cm.hot(dat_b/dat_b.max())*255.9).astype(np.uint8)[...,0:3])  # type: ignore[attr-defined]  # pylint: disable=no-member
    plt.imsave(f'tmp/figure3_{ind}_modl_top.png', (matplotlib.cm.hot(mod_t/mod_t.max())*255.9).astype(np.uint8)[...,0:3])  # type: ignore[attr-defined]  # pylint: disable=no-member
    plt.imsave(f'tmp/figure3_{ind}_modl_bot.png', (matplotlib.cm.hot(mod_b/mod_b.max())*255.9).astype(np.uint8)[...,0:3])  # type: ignore[attr-defined]  # pylint: disable=no-member

    save_ply.save_pointcloud_as_mesh(f'tmp/figure3_{ind}_model.ply', pts_img.cpu(),  res.net.get_model()[1].cpu(), 2.0, 0.1)

# some examples with reasonably complete rings and enough 
# distortion to illustrate the distortion model
_renderings(635,  nupc3d_resi, res_resi)
_renderings(752,  nupc3d_resi, res_resi)



def _render_very_many(res: _NetRes)->None:
    R = _rotation_ring_to_xy(res)

    pts = res.points_model @ trn(R)

    out = Path("tmp/fig3_meshes/")
    out.mkdir()

    for i,p in enumerate(tqdm(pts)):
        save_ply.save_pointcloud_as_mesh(out/f'{i:05}.ply', p, res.net.get_model()[1].cpu().detach(), 2.0, 0.1, 50)


