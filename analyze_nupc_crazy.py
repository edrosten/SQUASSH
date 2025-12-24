import math
from pathlib import Path

import torch
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


def _analyze(nupc3d: list[torch.Tensor], trained_weights: dict)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    #trained_weights = { k[10:]:v for k,v in trained_weights.items()}
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

    return torch.stack(pts_list, 0), intensities, parameterisation.get_R().cpu().detach()



def _get_stuff(components:int)->tuple[*[torch.Tensor]*5]:
    nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]
    trained_weights_resi = torch.load('log/1766516868-66b60604c41adb3c784b829cbd0205da1b12c1cd/phase_2/final_net.zip', map_location=torch.device('cpu'))
    results_resi_pts, results_resi_intensities, results_resi_R = _analyze(nupc3d_resi, trained_weights_resi)

    n_data = results_resi_pts.shape[0]
    flat_pts = results_resi_pts.reshape(n_data, -1)

    flat_pts_centred = flat_pts - flat_pts.mean(0).unsqueeze(0).expand(n_data, -1)
    (_, S, Vh) = torch.linalg.svd(flat_pts_centred, full_matrices=False) # pylint: disable=not-callable

    # Covariances are S^2 / (n-1)
    # standard devs are S/sqrt(n-1)
    stddev = S / (math.sqrt(n_data)-1)
    centre = flat_pts.mean(0).reshape(-1, 3)


    #Rot = euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ results_resi_R
    Rot = results_resi_R

    return centre, stddev, Vh[0:components, :], results_resi_intensities, Rot
    


# 
# plt.clf()
# plt.subplot(1,1,1, projection="3d")
# plt.gca().scatter(*(R @ centre.permute(1,0)))
# plt.axis('square')
# 
# 
# I=4
# component = Vh[I].reshape_as(centre)*stddev[I]*5
# 
# plt.gca().scatter(*(R @ (centre+component).permute(1,0)), alpha=0.2)
# plt.gca().scatter(*(R @ (centre-component).permute(1,0)), alpha=0.2)
# 

COMPONENTS=5
NUM_STEPS=120

sigmas=2

centre, stddev, Vh, results_resi_intensities, Rot = _get_stuff(COMPONENTS)


maxval = centre.max().item() * 1.5

Path('hax/nupc_component_animation').mkdir()

for frame_no in tqdm(range(NUM_STEPS)):
    position = math.sin(frame_no/NUM_STEPS * 2 * math.pi)
    
    for component in range(COMPONENTS):

        

        xyz = (centre + Vh[component].reshape_as(centre)*stddev[component]*sigmas * position) @ Rot.permute(1,0)
        save_ply.save_pointcloud_as_mesh(f"hax/nupc_component_animation/mesh-{component:02}-{frame_no:05}.ply", xyz.cuda(), results_resi_intensities.squeeze(0).cuda(), 2.0, .10, 100, maxval=maxval, chunksize=100)



