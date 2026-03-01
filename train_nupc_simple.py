# cast allows us to cast to specific types
from typing import cast

# importing deep learning packages
import torch
import torch._dynamo

# these are the imports for the data
# you will need to adapt one of these if you want to import your own data
import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import

# here we are importing the specific training information, network architecture and device
import train
import network
import device

# this specifies the type of 3D representation that will be used (this will vary between SMLM data and others)
from localisation_data import LocalisationDataSetMultipleDan6


def _main()->None:

    # Load the data. If you wish to load Bates data rather than resi, comment the top line and uncommet the one underneath
    nupc3d = [t.to(device.device).half() for t in resi_data.load_3d()]
    #nupc3d = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]

    
    # the rejection parameters is a weighting for how likely the optimisation is to reject a given patch based on quality
    # if you want the model to reject less, make the rejection parameter higher
    # if you want the model to reject more, make the rejection parameter lower
    rejection = 1.0

    # Patch size here is 64x64x32????????????????????????????????????????
    # z_scale is difference in scaling between xy axis and z axis
    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 3.9,
        z_scale = 2
    )
    
    params_initial = train.TrainingParameters()
    params_initial.batch_size = 50 
    params_initial.validity_weight=rejection

    # Optimisation parameters

    # The number of epochs needs to be set so that the system has reached a stable state by the time the optimisation terminates
    # If you want to check this ********

    # The blur reduction schedule below is not the fastest (see train_nupc.py for an optimised one) but is designed to be single step and easy 
    # to understand. The main limiting factors for speed are the number of points in the model (here 700) and the number of PSF steps taken
    # Faster optimisation can be achieved by optimising an initial model with fewer points and then re-seeding.

    # The learning rate is set to allow faster optimisation at the beginning of the run, when gradients are likely to be steeper.
    # A lower learning rate can be used for the whole optimisation at the cost of a longer run time.

    params_initial.schedule[0].epochs = 1500
    params_initial.schedule[0].initial_psf = 15*nm_per_pixel_xy
    params_initial.schedule[0].final_psf = 3*nm_per_pixel_xy
    params_initial.schedule[0].psf_step_every= 100
    params_initial.schedule[0].initial_lr= 0.0001
    params_initial.schedule[0].final_lr= 0.00005

    dataset_initial = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=nupc3d, augmentations=8, device=device.device)


    # torch dynamo optimises speed performance
    # resetting the torch compiler is advisable as it is occasionally prone to writing data into the compiled code if this is not done
    torch._dynamo.config.cache_size_limit=512  # pylint: disable=protected-access
    torch.compiler.reset()

    # This sets the size of the model and the parameters of the heterogeneity
    net, parameterisation =network.PredictReconstructionStretchExpandValidDan6(model_size=700, **vars(data_parameters), data=nupc3d)
    parameterisation.max_stretch_factor_axis = 2.0
    parameterisation.max_stretch_factor_expand = 1.0
    net.to(device.device)
    
    net._model_intensities.requires_grad=False  # pylint: disable=protected-access


    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset_initial, params_initial, 'phase_0')
    
        
if __name__ == "__main__":
    _main()



