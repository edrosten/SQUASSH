from typing import cast

import torch
import torch._dynamo
import resi_data         # noqa
import mark_bates_data   # noqa
import train
import network
import device
from localisation_data import LocalisationDataSetMultipleDan6


def _main()->None:
    nupc3d = [t.to(device.device).half() for t in resi_data.load_3d()]
    #nupc3d = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]

    mult = 20
    scatter = 0.01

    SCALE=1.3
    rejection = 1.0

    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 3*SCALE,
        z_scale = 2
    )
    
    params_initial = train.TrainingParameters()
    params_initial.batch_size = 160 
    params_initial.validity_weight=rejection

    params_initial.schedule[0].epochs = 90
    params_initial.schedule[0].initial_psf = 50*SCALE
    params_initial.schedule[0].final_psf = 26*SCALE
    params_initial.schedule[0].psf_step_every= 30
    params_initial.schedule[0].initial_lr= 0.0001
    params_initial.schedule[0].final_lr= 0.0001

    params_initial.schedule.append(train.TrainingSegment())
    params_initial.schedule[1].epochs = 300
    params_initial.schedule[1].initial_psf = 19*SCALE
    params_initial.schedule[1].final_psf = 10.0*SCALE
    params_initial.schedule[1].psf_step_every= 100
    params_initial.schedule[1].initial_lr= 0.0001
    params_initial.schedule[1].final_lr= 0.0001

    dataset_initial = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=8, device=device.device)


    params_refine = train.TrainingParameters()
    params_refine.batch_size = 10
    params_refine.validity_weight=rejection

    params_refine.schedule[0].epochs = 1000
    params_refine.schedule[0].initial_psf = 10.0*SCALE
    params_refine.schedule[0].final_psf = 10.0*SCALE
    params_refine.schedule[0].psf_step_every= 300
    params_refine.schedule[0].initial_lr= 0.0002
    params_refine.schedule[0].final_lr= 0.00005

    dataset_refine = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)

    torch._dynamo.config.cache_size_limit=512  # pylint: disable=protected-access

    torch.compiler.reset()

    net, parameterisation =network.PredictReconstructionStretchExpandValidDan6(model_size=35, **vars(data_parameters), data=nupc3d)
    parameterisation.max_stretch_factor_axis = 2.0
    parameterisation.max_stretch_factor_expand = 1.0
    net.to(device.device)
    
    net._model_intensities.requires_grad=False  # pylint: disable=protected-access


    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset_initial, params_initial, 'phase_0')
    
    scale = net.get_model()[0].abs().max().item()
    
    old_pts, old_weights = (j.detach() for j in net.get_model())

    new_pts = torch.nn.functional.interpolate(old_pts.unsqueeze(0).unsqueeze(0), scale_factor=[mult,1]).squeeze(0).squeeze(0)
    new_pts += torch.randn(new_pts.shape, device=device.device) * scale * scatter

    new_weights = torch.nn.functional.interpolate(old_weights.unsqueeze(0).unsqueeze(0), scale_factor=mult).squeeze(0).squeeze(0)
    
    net.set_model(new_pts, new_weights)
    net._model_intensities.requires_grad=True  # pylint: disable=protected-access
    parameterisation.max_stretch_factor_expand = 1.3
    
    torch.compiler.reset() # Otherwise it crashes on torch 2.7
    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset_refine, params_refine, 'phase_1')

        
if __name__ == "__main__":
    _main()



