import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm


import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import train
import device
from matrix import trn
from localisation_data import LocalisationDataSetMultipleDan6
from train_nupc import PredictReconstruction


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

    net, parameterisation = PredictReconstruction(model_size=700, **vars(data_parameters), data=nupc3d)
    parameterisation.max_stretch_factor_axis = torch.tensor(2.0)
    parameterisation.max_stretch_factor_expand = torch.tensor(1.0)


    trained_weights = { k[10:]:v for k,v in trained_weights.items()}
    net.load_state_dict(trained_weights)

    net.to(device.device)
    net.eval()
            
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    dataset.set_sigma(final_sigma)

    R=4
    C=6
    #plt.ion()

    scale_list = []
    scale_list2 = []
    aspect_list = []

    plot=False

    with torch.no_grad():
        for datum in tqdm(loader):

            t,r,_,is_valid,parameters = net.process_input(datum, min_sigma_nm=final_sigma_t)

            points, intensities, _ = net._parameterisation(*net.get_model(), parameters) # pylint: disable=protected-access

            t = t.squeeze(0)
            r = r.squeeze(0)
            points = points.squeeze(0)
            intensities = intensities.squeeze(0)

            pts2d = trn(parameterisation.get_R() @ trn(points))[:,1:3]
            
            scale1 = parameterisation.max_stretch_factor_axis**torch.tanh(parameters[:,0])
            scale2 = parameterisation.max_stretch_factor_expand**torch.tanh(parameters[:,1])
            scale3 = parameterisation.max_stretch_factor_expand**torch.tanh(parameters[:,2])


            

            shft = pts2d - pts2d.mean(0).unsqueeze(0).expand_as(pts2d)
            cov = torch.einsum('ij,ik->jk', shft, shft) / shft.shape[0]
            val, vec =torch.linalg.eigh(cov)  #pylint: disable=not-callable
            #print(val.sqrt())
            scale=val.mean().sqrt()
            aspect=val[0]/val[1]
            
            if is_valid > 0.5:
                scale_list.append(scale.item())
                aspect_list.append(aspect.item())
                scale_list2.append([scale1.item(), scale2.item(), scale3.item()])

            # cov = vec @ val.diag() @ trn(vec)

            if plot:
                plt.clf()
                plt.suptitle(f'Validity = {is_valid.item():0.3}, {scale:0.3}, {aspect:0.3}')
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

                
                # Make a ring
                theta = torch.arange(0, 2*torch.pi, .01, device=pts2d.device)
                xy = torch.stack([theta.cos(), theta.sin()], -1)

                # Warp it to match the covariance
                xy = trn(vec @ val.diag().sqrt() @ trn(xy))

                # Make it 3D
                xyz = torch.cat([torch.zeros(xy.shape[0], 1, device=xy.device), xy], 1)
                
                # Transform it to match the model
                xyz = trn(trn(parameterisation.get_R())@trn(xyz))

                plt.subplot(R, C, 3*C + 3, projection='3d')
                plt.gca().scatter(*trn(points.cpu()))
                plt.gca().scatter(*trn(xyz.cpu()))
                plt.axis('square')
                #breakpoint()
                #print(r)
                

                xyz_render = trn(r@trn(xyz)) + t.unsqueeze(0).expand_as(xyz)
                xy_px = xyz_render[:,0:2]/data_parameters.nm_per_pixel_xy + torch.ones_like(xy)*data_parameters.image_size_xy/2
                xy_px = xy_px.cpu()


                plt.subplot(R, C, 3*C + 1)
                plt.imshow(datum[0][0,0].cpu(), cmap='grey')
                plt.plot(xy_px[:,0], xy_px[:,1], 'r.', markersize=.1)

                plt.subplot(R, C, 3*C + 2)
                plt.imshow(reconstruction[0][0,0].cpu(), cmap='grey')
                plt.plot(xy_px[:,0], xy_px[:,1], 'r.', markersize=.1)
#                plt.plot(*(pts2d.cpu().permute(1,0)/data_parameters.nm_per_pixel_xy + torch.ones(pts2d.shape[::-1])*data_parameters.image_size_xy/2), 'r.')

                plt.show()


    return torch.tensor(scale_list), torch.tensor(aspect_list).sqrt(), torch.tensor(scale_list2)



nupc3d_bates = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]
#trained_weights_bates = torch.load('log/1765837449-5303f0399ec89b6b1b1a71436f5b6a9bbe78ca85/phase_1/final_net.zip', map_location=torch.device('cpu'))
#trained_weights_bates = torch.load('log/1765730982-66cba4c56ab77c0d62f9e60692b96f64de4f8cae//phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_bates = torch.load('log/1765923389-aa9ec76d4d2359d653ea4c6a584c3a324ead0607/phase_1/final_net.zip', map_location=torch.device('cpu'))
results_bates = _analyze(nupc3d_bates, trained_weights_bates)[0:2]

#trained_weights_bates = torch.load('log/1765837449-5303f0399ec89b6b1b1a71436f5b6a9bbe78ca85/phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_bates = torch.load('log/1765730982-66cba4c56ab77c0d62f9e60692b96f64de4f8cae//phase_1/final_net.zip', map_location=torch.device('cpu'))
#trained_weights_bates = torch.load('log/1765923389-aa9ec76d4d2359d653ea4c6a584c3a324ead0607/phase_1/final_net.zip', map_location=torch.device('cpu'))
results_bates_old = _analyze(nupc3d_bates, trained_weights_bates)[0:2]


nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]
trained_weights_resi_2 = torch.load('log/1765997082-c67688f4c08fbb970a939d2d3c092665acc6ba76/phase_1/final_net.zip', map_location=torch.device('cpu'))
results_resi_2 = _analyze(nupc3d_resi, trained_weights_resi_2)[0:2]
trained_weights_resi = torch.load('log/1765490795-dea2d3dfafefca44b9fa5e1ea06ba6b8148439c1/phase_1/final_net.zip', map_location=torch.device('cpu'))
results_resi = _analyze(nupc3d_resi, trained_weights_resi)[0:2]


plt.clf()
corr_resi=scipy.stats.pearsonr(*results_resi)
corr_resi_2=scipy.stats.pearsonr(*results_resi_2)
corr_bates=scipy.stats.pearsonr(*results_bates)
corr_bates_old=scipy.stats.pearsonr(*results_bates_old)

plt.scatter(*results_resi, label=f'RESI {corr_resi}', c='#1f77b480')
plt.scatter(*results_resi_2, label=f'RESI 2 {corr_resi_2}', c='#1f00ff80')
plt.scatter(*results_bates, label=f'Bates {corr_bates}', c='#0eff0e40')
plt.scatter(*results_bates_old, label=f'Bates old {corr_bates_old}', c='#ff7f0e40')


plt.legend()

trained_weights_e4b2ea = torch.load('log/1766239291-e4b2ea70e969e33151f878439ac1c7c895fb9b24//phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_aa9ec7 = torch.load('log/1765923389-aa9ec76d4d2359d653ea4c6a584c3a324ead0607/phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_5303f0 = torch.load('log/1765837449-5303f0399ec89b6b1b1a71436f5b6a9bbe78ca85/phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_66cba4 = torch.load('log/1765730982-66cba4c56ab77c0d62f9e60692b96f64de4f8cae//phase_1/final_net.zip', map_location=torch.device('cpu'))

results_e4b2ea = _analyze(nupc3d_bates, trained_weights_e4b2ea)
results_aa9ec7 = _analyze(nupc3d_bates, trained_weights_aa9ec7)
results_5303f0 = _analyze(nupc3d_bates, trained_weights_5303f0)
results_66cba4 = _analyze(nupc3d_bates, trained_weights_66cba4)


trained_weights_c67688 = torch.load('log/1765997082-c67688f4c08fbb970a939d2d3c092665acc6ba76/phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_ce0f1c = torch.load('log/1766269628-ce0f1c48ac3538fbcda3f5840fa64d5cbe92f612/phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights_fda1bd = torch.load('log/1766315401-fda1bd5a319d286cffae2e9d0f51d54aca1385c0/phase_1/final_net.zip', map_location=torch.device('cpu'))
results_c67688 = _analyze(nupc3d_resi, trained_weights_c67688)
results_ce0f1c = _analyze(nupc3d_resi, trained_weights_ce0f1c)
results_fda1bd = _analyze(nupc3d_resi, trained_weights_fda1bd)

plt.clf()
#corr_X=scipy.stats.pearsonr(*results_e4b2ea[0:2])

plt.scatter(*results_e4b2ea[0:2], label='e4b2ea')
plt.scatter(*results_aa9ec7[0:2], label='aa9ec7')
plt.scatter(*results_5303f0[0:2], label='5303f0', c='#1f77b480')
plt.scatter(*results_66cba4[0:2], label='66cba4', c='#0eff0e40')



plt.scatter(*results_aa9ec7[0:2], label='aa9ec7')
plt.scatter(*results_c67688[0:2], label='c67688')
plt.scatter(*results_ce0f1c[0:2], label='ce0f1c', c='#0eff0ef0')
plt.scatter(*results_fda1bd[0:2], label='fda1bd')
plt.legend()
plt.show()

