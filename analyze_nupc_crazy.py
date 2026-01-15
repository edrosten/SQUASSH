import math
from pathlib import Path
from typing import Any

import scipy
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import train
import device
from network import GeneralPredictReconstruction
from matrix import trn, euler
from localisation_data import LocalisationDataSetMultipleDan6
from train_nupc import PredictReconstruction, AxialStretchRadialExpandWithGeneralShift
import save_ply


COMPONENTS=5
NUM_STEPS=120
sigmas=2
SCALE=1.3

data_parameters = train.DataParametersXYYZ(
    image_size_xy = 64,
    image_size_z = 32,
    nm_per_pixel_xy = 3*SCALE,
    z_scale = 2
)

final_fwhm = SCALE * 10.0



def _load_net(nupc3d: list[Tensor], trained_weights: dict, pts:int)->tuple[GeneralPredictReconstruction, AxialStretchRadialExpandWithGeneralShift]:
    net, parameterisation = PredictReconstruction(initial_model_size=pts, final_model_size=pts, **vars(data_parameters), data=nupc3d)
    parameterisation.per_point_shift=True
    
    if "_orig" in next(iter(trained_weights.keys())):
        trained_weights = { k[10:]:v for k,v in trained_weights.items()}
    trained_weights = { k.replace("_shift_network", "shift_network"):v for k,v in trained_weights.items()}
    net.load_state_dict(trained_weights)


    net.eval()
    for i in net.parameters():
        i.requires_grad=False
        
    return net, parameterisation


def _analyze(nupc3d: list[Tensor], trained_weights: dict, pts:int=700)->tuple[Tensor, Tensor, Tensor, list[int], Tensor]:

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
    r_list = []

    plot=False

    with torch.no_grad():
        for index,datum in enumerate(tqdm(loader)):

            t,r,_,is_valid,parameters = net.process_input(datum, min_sigma_nm=final_sigma_t)

            points, intensities, _ = net._parameterisation(*net.get_model(), parameters) # pylint: disable=protected-access

            t = t.squeeze(0)
            r = r.squeeze(0)
            points = points.squeeze(0)

            if is_valid > 0.5:
                pts_list.append(points.cpu())
                ind_list.append(index)
                r_list.append(r)

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
                
                plt.subplot(R, C, 3*C + 3, projection='3d')
                plt.gca().scatter(*trn(points.cpu()))
                plt.axis('square')
                plt.tight_layout()
                plt.show()

    return torch.stack(pts_list, 0), intensities.squeeze(0), parameterisation.get_R().cpu().detach(), ind_list, torch.stack(r_list, 0)



def _get_stuff(nupc3d: list[Tensor], trained_weights: dict, components:int, pts:int=700)->tuple[tuple[Tensor,Tensor,Tensor,Tensor,Tensor], Tensor, Tensor, list[int], Tensor]:
    results_pts, results_intensities, results_R, good_inds, rotations = _analyze(nupc3d, trained_weights, pts)

    #nupc3d_bates = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]
    #trained_weights_bates = torch.load('log/1766605809-a396351dc2c407f97a32efa35b421a9aa8d2de55/phase_2/final_net.zip', map_location=torch.device('cpu'))
    #results_pts, results_intensities, results_resi_R = _analyze(nupc3d_bates, trained_weights_bates)
   

    # Find ring sizes
    # swap x and z axes, so align stretch axis to z
    R = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ results_R
    top_mask = (R @ results_pts.mean(0).permute(1,0)).permute(1,0)[:,2] > 0
    bot_mask = top_mask.logical_not()
    
    covs_list = []

    for res in results_pts:
        pts_aligned = trn(R @ trn(res))

        top_2d = pts_aligned[top_mask][:,0:2]
        top_2d = top_2d - top_2d.mean(0).unsqueeze(0).expand_as(top_2d)
        cov_top = torch.einsum('ij,ik->jk', top_2d, top_2d) / top_2d.shape[0]


        bot_2d = pts_aligned[bot_mask][:,0:2]
        bot_2d = bot_2d - bot_2d.mean(0).unsqueeze(0).expand_as(bot_2d)
        cov_bot = torch.einsum('ij,ik->jk', bot_2d, bot_2d) / bot_2d.shape[0]


        #covs_list.append([cov_top.trace().sqrt().item(), cov_bot.trace().sqrt().item()]) 
        covs_list.append(torch.stack([cov_top, cov_bot], 0))






    n_data = results_pts.shape[0]
    flat_pts = results_pts.reshape(n_data, -1)

    flat_pts_centred = flat_pts - flat_pts.mean(0).unsqueeze(0).expand(n_data, -1)
    (_, S, Vh_vectors) = torch.linalg.svd(flat_pts_centred, full_matrices=False) # pylint: disable=not-callable

    # Covariances are S^2 / (n-1)
    # standard devs are S/sqrt(n-1)
    stddev = S / (math.sqrt(n_data)-1)
    centre = flat_pts.mean(0).reshape(-1, 3)


    #Rot = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ results_resi_R
    Rot = results_R
    
    print(rotations.shape)
    print(R.shape)
    print(R)
    print(rotations)

    cov_rot_to_image_space = rotations.cpu() @ R.permute(1,0).unsqueeze(0).expand(rotations.shape[0], 3, 3).cpu()

    return (centre.cpu(), stddev.cpu(), Vh_vectors[0:components, :].cpu(), results_intensities.cpu(), Rot.cpu()), torch.stack(covs_list,0), results_pts, good_inds, cov_rot_to_image_space
    



def _matplotlib_animation(centre: Tensor, stddev: Tensor, Vh: Tensor, _: Any, Rot: Tensor)->None:
    I=0
    R = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ Rot

    top_mask = (R @ centre.permute(1,0)).permute(1,0)[:,2] > 0
    
    for _ in range(100):
        for frame_no in range(NUM_STEPS):
            position = math.sin(frame_no/NUM_STEPS * 2 * math.pi)
            
            component = Vh[I].reshape_as(centre)*stddev[I]*sigmas*position

            plt.clf()
            plt.subplot(1,2,1)
            #plt.gca().scatter(*(R @ centre[top_mask,:].permute(1,0))[0:2,:])
            plt.gca().scatter(*(R @ (centre+component)[top_mask,:].permute(1,0))[0:2,:], alpha=0.2) # type: ignore[misc]
            #plt.gca().scatter(*(R @ (centre-component)[top_mask,:].permute(1,0))[0:2,:], alpha=0.2)
            plt.axis('equal')
            plt.axis((-60,60,-60,60))

            plt.subplot(1,2,2)
            #plt.gca().scatter(*(R @ centre[top_mask.logical_not(),:].permute(1,0))[0:2,:])
            plt.gca().scatter(*(R @ (centre+component)[top_mask.logical_not(),:].permute(1,0))[0:2,:], alpha=0.2) # type: ignore[misc]
            #plt.gca().scatter(*(R @ (centre-component)[top_mask.logical_not(),:].permute(1,0))[0:2,:], alpha=0.2)
            plt.axis('equal')
            plt.axis((-60,60,-60,60))

            plt.suptitle(f'{position:0.3}')

            plt.pause(0.03)

def _mesh_animation(centre: Tensor, stddev: Tensor, Vh: Tensor, intensities: Tensor, Rot: Tensor)->None:
    maxval = centre.max().item() * 1.5
    Path('hax/nupc_component_animation').mkdir()
    for frame_no in tqdm(range(NUM_STEPS)):
        position = math.sin(frame_no/NUM_STEPS * 2 * math.pi)
        for component in range(COMPONENTS):
            xyz = (centre + Vh[component].reshape_as(centre)*stddev[component]*sigmas * position) @ Rot.permute(1,0)
            save_ply.save_pointcloud_as_mesh(f"hax/nupc_component_animation/mesh-{component:02}-{frame_no:05}.ply", xyz.cuda(), intensities.squeeze(0).cuda(), 2.0, .10, 100, maxval=maxval, chunksize=100)




def _pca_figure(centre: Tensor, stddev: Tensor, Vh: Tensor, intensities: Tensor, Rot: Tensor)->None:
    # Flip X and Z axes
    R = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ Rot

    _, darkest_first = intensities.sort()
    intensities = intensities[darkest_first]
    centre = centre[darkest_first,:]
    Vh = Vh.reshape(Vh.shape[0], *centre.shape)[:, darkest_first, :]


    top_mask = (R @ centre.permute(1,0)).permute(1,0)[:,2] > 0
    
    N=3 
    alpha=1.0
    plt.clf()
    for I in range(3):
        component = Vh[I]*stddev[I]*3

        plt.subplot(2,N,I+1)
        plt.scatter(*(R @ (centre          )[top_mask,:].permute(1,0))[0:2,:], c=intensities[top_mask], alpha=alpha, cmap='Greys', edgecolors='none')  # type: ignore[misc]
        plt.scatter(*(R @ (centre+component)[top_mask,:].permute(1,0))[0:2,:], c=intensities[top_mask], alpha=alpha, cmap='Oranges', edgecolors='none')  # type: ignore[misc]
        plt.xlabel(f'Component {I+1}')
        plt.axis('square')
        plt.axis((-65,65,-65,65))
        for line in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[line].set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().xaxis.set_label_position('top')
        if I == 0:
            plt.ylabel('Upper ring')

        plt.subplot(2,N,I+1+N)
        plt.scatter(*(R @ (centre          )[top_mask.logical_not(),:].permute(1,0))[0:2,:], c=intensities[top_mask.logical_not()], alpha=alpha, cmap='Greys', edgecolors='none')  # type: ignore[misc]
        plt.scatter(*(R @ (centre+component)[top_mask.logical_not(),:].permute(1,0))[0:2,:], c=intensities[top_mask.logical_not()], alpha=alpha, cmap='Oranges', edgecolors='none')  # type: ignore[misc]
        plt.axis('square')
        plt.axis((-65,65,-65,65))
        for line in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[line].set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        if I == 0:
            plt.ylabel('Lower ring')
    
    plt.tight_layout()
    plt.pause(.1)
    plt.savefig('hax/supp_resi_pca.svg')



def _print_stats(stds:torch.Tensor, ratio: torch.Tensor)->None:
    print("Size top / bottom ± at 1σ")
    for i in [0,1]:
        print(f"{stds[:,i].mean().item():0.4} ± {(stds[:,i].var()/stds.shape[0]).sqrt().item():0.2}   ", end="")
    print("\n")

    print("Aspect ratio top/bottom")
    for i in [0,1]:
        print(f"{ratio[:,i].mean().item():0.4} ± {(ratio[:,i].var()/ratio.shape[0]).sqrt().item():0.2}   ", end="")

def _std_and_ratio(covs: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    # Std dev (variance) as trace of covariance matrix, equivlaent to RMS radius
    stds = covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).sqrt()

    principal_axes=torch.linalg.eigvalsh(covs).sqrt() # pylint: disable=not-callable
    ratios = principal_axes[...,0]/principal_axes[...,1]
    return stds, ratios




torch.no_grad()


nupc3d_resi, nupc3d_resi_means = resi_data.load_3d_with_means()
nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]

#trained_weights_resi = torch.load('log/1766516868-66b60604c41adb3c784b829cbd0205da1b12c1cd/phase_2/final_net.zip', map_location=torch.device('cpu'))
#results_resi, covs_resi = _get_stuff(nupc3d_resi, trained_weights_resi, COMPONENTS)



# 32 point model!
trained_weights_resi = torch.load('log/1767449867-0b3ce320f9213553e0b7d942d407268e8c3db4a4/phase_2/final_net.zip', map_location=torch.device('cpu'))
results_resi, covs_resi, pts_resi, good_inds_resi, cov_rot_resi = _get_stuff(nupc3d_resi, trained_weights_resi, COMPONENTS, 32)
means_resi = torch.stack(nupc3d_resi_means, 0)[good_inds_resi]


std_ratio_resi = _std_and_ratio(covs_resi)
print("RESI data")
print("---------")
_print_stats(*std_ratio_resi)

_pca_figure(*results_resi)



#nupc3d_bates = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l] # type: ignore[unreachable]
#trained_weights_bates = torch.load('log/1766605809-a396351dc2c407f97a32efa35b421a9aa8d2de55/phase_2/final_net.zip', map_location=torch.device('cpu'))
#results_bates, covs_bates = _get_stuff(nupc3d_bates, trained_weights_bates, COMPONENTS)
#
#
#std_ratio_bates = _std_and_ratio(covs_bates)
#print("Bates data")
#print("----------")
#_print_stats(*std_ratio_bates)
#


net_resi = _load_net(nupc3d_resi, trained_weights_resi, 32)



def _nn_graph(pts: Tensor)->tuple[Tensor, Tensor]:
    # Simple quadratical method
    npts = pts.shape[0]
    
    pairwise_distance = (pts.unsqueeze(0).expand(npts, *pts.shape) - pts.unsqueeze(1).expand(npts, *pts.shape)).pow(2).sum(-1).sqrt() + torch.eye(npts)*1e10
    
    min_dist, index_of_closest = pairwise_distance.min(1)

    good_points = min_dist < 25.0
    
    plt.clf() 
    plt.subplot(1,1,1,projection='3d')
    plt.gca().scatter(*pts.permute(1,0))
    for i in torch.arange(npts)[good_points]:
        p1 = pts[i] 
        p2 = pts[index_of_closest[i]]
        ps = torch.stack([p1, (p1+p2)/2, p2], 0)
        plt.gca().plot(*ps.permute(1,0)[:,0:2], c=plt.cm.winter(0.0)) #type: ignore[attr-defined] # pylint: disable=no-member
        plt.gca().plot(*ps.permute(1,0)[:,1:3], c=plt.cm.winter(1.0)) #type: ignore[attr-defined] # pylint: disable=no-member


    return index_of_closest, good_points





def _plot_distance_vs_eccentricity()->None:

    index_closest, good_mask = _nn_graph(net_resi[0].get_model()[0])

    assert good_mask.all(), "Honestly this has not been tested with slightly incomplete models"

    pts = pts_resi
    closest = pts[:, index_closest, :]


    distances = (pts-closest).pow(2).sum(-1).sqrt()

    R = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ results_resi[4]
    top_mask = (R @ net_resi[0].get_model()[0].permute(1,0)).permute(1,0)[:,2] > 0
    bot_mask = top_mask.logical_not()

    std_dist_top = distances[:,top_mask].std(1)
    std_dist_bot = distances[:,bot_mask].std(1)
    #mean_dist_top = distances[:,top_mask].mean(1)
    #mean_dist_bot = distances[:,bot_mask].mean(1)

    def _to_pretty_sci(x: float)->str:
        superscripts = "⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹"
        normal       = "+-0123456789"
        mapping=dict(zip(normal, superscripts))
        x_str = ("%.1E"%x).split("E") # pylint: disable=consider-using-f-string
        print(x_str)
        return x_str[0] + "×10" + "".join([mapping[i] for i in x_str[1]])



    eccentricity = (1-std_ratio_resi[1]**2).sqrt()
    top_stats = scipy.stats.pearsonr(eccentricity[:,0], std_dist_top)
    bot_stats = scipy.stats.pearsonr(eccentricity[:,1], std_dist_bot)
    plt.close('all')
    plt.clf()
    plt.scatter(eccentricity[:,0], std_dist_top, label=f"NR r={top_stats.statistic:0.2}, p={_to_pretty_sci(top_stats.pvalue)}")
    plt.scatter(eccentricity[:,1], std_dist_bot, label=f"CR r={bot_stats.statistic:0.2}, p={_to_pretty_sci(bot_stats.pvalue)}")
    plt.xlabel('Eccentricity')
    plt.ylabel('Standard deviation of doublet spacing')
    plt.legend()
    plt.pause(.1)
    plt.savefig('tmp/doublet_spacing_variance_vs_eccentricity.svg')

_plot_distance_vs_eccentricity()



# eigenvectors are the columns
covs_evalues, covs_evectors = torch.linalg.eigh(covs_resi) #pylint: disable=not-callable

# Not guaranteed but the max is always the last
assert covs_evalues.max(-1).indices.min()==1

covs_evectors = cov_rot_resi.unsqueeze(1).expand(-1, 2, 3, 3)[:,:,0:2,0:2] @ covs_evectors

cov_biggest_vector = covs_evectors[...,:,1]
cov_biggest_vector *= cov_biggest_vector[...,1].sign().unsqueeze(-1).expand(-1, 2, 2)
angles = torch.atan2(cov_biggest_vector[...,1], cov_biggest_vector[...,0])


#for i in range(100):
#
#    pts1 = net_resi[0].get_model()[0].cpu()
#    pts2 = pts_resi[i]
#
#    plt.clf()index_closest, good_mask = _nn_graph(net_resi[0].get_model()[0])
#    plt.subplot(1,1,1,projection='3d')
#    plt.gca().scatter(*pts1.permute(1,0))
#    plt.gca().scatter(*pts2.permute(1,0))
#
#    for p1, p2 in zip(pts1, pts2):
#        ps = torch.stack([p1, (p1+p2)/2, p2], 0)
#        ps = torch.stack([p1, (p1+p2)/2, p2], 0)
#        plt.gca().plot(*ps.permute(1,0)[:,0:2], c=plt.cm.winter(0.0)) # pylint: disable = no-member
#        plt.gca().plot(*ps.permute(1,0)[:,1:3], c=plt.cm.winter(1.0)) # pylint: disable = no-member
#    
#:plt.show()

