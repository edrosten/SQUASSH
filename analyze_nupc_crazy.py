import math
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import train
import device
from matrix import trn, euler
from localisation_data import LocalisationDataSetMultipleDan6
from train_nupc import PredictReconstructionCrazy
import save_ply


def _analyze(nupc3d: list[Tensor], trained_weights: dict)->tuple[Tensor, Tensor, Tensor]:
    SCALE=1.3

    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 3*SCALE,
        z_scale = 2
    )

    final_fwhm = SCALE * 10.0

    final_sigma = train.fwhm_to_sigma(final_fwhm)
    final_sigma_t = torch.tensor(final_sigma)

    dataset = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)

    net, parameterisation = PredictReconstructionCrazy(model_size=700, **vars(data_parameters), data=nupc3d)
    parameterisation.crazy=True
    
    if "_orig" in next(iter(trained_weights.keys())):
        trained_weights = { k[10:]:v for k,v in trained_weights.items()}
    net.load_state_dict(trained_weights)

    net.to(device.device)
    net.eval()
            
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataset.set_sigma(final_sigma)

    R=4
    C=6
    #plt.ion()

    pts_list = []

    plot=False

    with torch.no_grad():
        for datum in tqdm(loader):

            t,r,_,is_valid,parameters = net.process_input(datum, min_sigma_nm=final_sigma_t)

            points, intensities, _ = net._parameterisation(*net.get_model(), parameters) # pylint: disable=protected-access

            t = t.squeeze(0)
            r = r.squeeze(0)
            points = points.squeeze(0)

            if is_valid > 0.5:
                pts_list.append(points.cpu())

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

    return torch.stack(pts_list, 0), intensities.squeeze(0), parameterisation.get_R().cpu().detach()



def _get_stuff(nupc3d: list[Tensor], trained_weights: dict, components:int)->tuple[Tensor,Tensor,Tensor,Tensor,Tensor]:
    results_pts, results_intensities, results_R = _analyze(nupc3d, trained_weights)

    #nupc3d_bates = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]
    #trained_weights_bates = torch.load('log/1766605809-a396351dc2c407f97a32efa35b421a9aa8d2de55/phase_2/final_net.zip', map_location=torch.device('cpu'))
    #results_pts, results_intensities, results_resi_R = _analyze(nupc3d_bates, trained_weights_bates)
    

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

    return centre, stddev, Vh_vectors[0:components, :], results_intensities, Rot
    

nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]
trained_weights_resi = torch.load('log/1766516868-66b60604c41adb3c784b829cbd0205da1b12c1cd/phase_2/final_net.zip', map_location=torch.device('cpu'))

COMPONENTS=5
NUM_STEPS=120



sigmas=4

#centre_resi, stddev_resi, Vh_resi, intensities_resi, Rot_resi = 
results_resi = tuple(i.cpu() for i in _get_stuff(nupc3d_resi, trained_weights_resi, COMPONENTS))


def _matplotlib_animation(centre: Tensor, stddev: Tensor, Vh: Tensor, _: Tensor, Rot: Tensor)->None:
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
    R = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ Rot

    _, darkest_first = intensities.sort()
    intensities = intensities[darkest_first]
    centre = centre[darkest_first,:]
    Vh = Vh.reshape(Vh.shape[0], *centre.shape)[:, darkest_first, :]


    top_mask = (R @ centre.permute(1,0)).permute(1,0)[:,2] > 0
    
    N=3 
    plt.clf()
    for I in range(3):
        component = Vh[I]*stddev[I]*3

        plt.subplot(2,N,I+1)
        plt.scatter(*(R @ (centre          )[top_mask,:].permute(1,0))[0:2,:], c=intensities[top_mask], alpha=0.2, cmap='Greys', edgecolors='none')  # type: ignore[misc]
        plt.scatter(*(R @ (centre+component)[top_mask,:].permute(1,0))[0:2,:], c=intensities[top_mask], alpha=0.2, cmap='Oranges', edgecolors='none')  # type: ignore[misc]
        plt.xlabel(f'Component {I+1}')
        plt.axis('equal')
        plt.axis((-65,65,-65,65))
        for line in ['top', 'bottom', 'left', 'right']:
            plt.gca().spines[line].set_visible(False)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().xaxis.set_label_position('top')
        if I == 0:
            plt.ylabel('Upper ring')

        plt.subplot(2,N,I+1+N)
        plt.scatter(*(R @ (centre          )[top_mask.logical_not(),:].permute(1,0))[0:2,:], c=intensities[top_mask.logical_not()], alpha=0.2, cmap='Greys', edgecolors='none')  # type: ignore[misc]
        plt.scatter(*(R @ (centre+component)[top_mask.logical_not(),:].permute(1,0))[0:2,:], c=intensities[top_mask.logical_not()], alpha=0.2, cmap='Oranges', edgecolors='none')  # type: ignore[misc]
        plt.axis('equal')
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
_pca_figure(*results_resi)



