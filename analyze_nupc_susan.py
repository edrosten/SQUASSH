from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import train
import device
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


def _render(res: _NetRes)->torch.Tensor:
    
    baseline_model = res.net.get_model()[0].to(device.device)
    intensities = res.net.get_model()[1].to(device.device)
    # Rotate R so Z aligns with the stretch axis
    R = (euler(90*torch.tensor([torch.pi])/180, 'y').squeeze() @ res.parameterisation.get_R().cpu()).to(device.device)

    top_mask = ((R @ baseline_model.permute(1,0)).permute(1,0)[:,2] > 0).to(device.device)
    bot_mask = top_mask.logical_not()


    sigma=torch.tensor([3.0]).to(baseline_model)
    nm_per_pixel=3.9
    size=64

    res_list = []

    for pts in tqdm(res.points_model):
        points = pts.to(device.device) @trn(R)

        top_points_2d = points[top_mask, 0:2].unsqueeze(0)
        bot_points_2d = points[bot_mask, 0:2].unsqueeze(0)

        top_render = render.render_batch_weights(top_points_2d, sigma, intensities[top_mask].unsqueeze(0), nm_per_pixel, size).squeeze(0).cpu()
        bot_render = render.render_batch_weights(bot_points_2d, sigma, intensities[bot_mask].unsqueeze(0), nm_per_pixel, size).squeeze(0).cpu()
        
        res_list.append(torch.stack([top_render, bot_render], 0))

    return torch.stack(res_list, 0)



nupc3d_resi, nupc3d_resi_means = resi_data.load_3d_with_means()
nupc3d_resi = [t.to(device.device).half() for t in resi_data.load_3d()]
trained_weights_resi = torch.load('log/1766516868-66b60604c41adb3c784b829cbd0205da1b12c1cd/phase_2/final_net.zip', map_location=torch.device('cpu'))
res_resi = _apply_net(nupc3d_resi, trained_weights_resi)
good_means_resi = torch.stack(nupc3d_resi_means)[res_resi.indices,:]

rendered = _render(res_resi)

