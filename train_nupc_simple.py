# cast needed to tell the type checker what a specific type is, since torch.compile returns a very generic type
from typing import cast

# Used for nice status display
from tqdm import tqdm

# Used for plotting
import matplotlib.pyplot as plt

# importing deep learning packages
import torch
from torch.utils.data import DataLoader

# these are the imports for the data
# you will need to adapt one of these if you want to import your own data
import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import

# here we are importing the specific training information, network architecture and device
import train
import network
import device

# this specifies the type of 3D representation that will be used (this will vary between SMLM data and others)
import localisation_data


def analyze_data(net: network.GeneralPredictReconstruction, parameterisation: network.ModelParameterisation, dataset: localisation_data.GeneralLocalisationDataSet)->tuple[torch.Tensor, torch.Tensor]:
    '''Runs data through the network. Returns the scales and axes of valid data'''
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # We need to pass the rendering sigma into the network. This needs to 
    # be as a tensor on the correct device
    fwhm_t = torch.tensor(13.0).to(device.device)

    scales = []
    axes = []

    net.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            # net() runs the network and produces a rendered image output. We want to analyze the 
            # results, not just images, so we use process_input which outputs all the intermediate
            # data including logits for the parameterisation
            _,r,_,is_valid,parameters = net.process_input(batch, min_sigma_nm=localisation_data.fwhm_to_sigma(fwhm_t))

            # Compute various things from the parameterisation
            scale = parameterisation.compute_scale_from_parameters(parameters).cpu().item()
            expand = parameterisation.compute_expand_from_parameters(parameters).cpu().item()
            
            # Reject invalid data from the network. Also, a number of NPCs have only one visible ring 
            # which the network will reproduce by outputting strong in-axis squash in an attempt to 
            # merge the two rings together. These NPCs are not useful to analyze.
            if is_valid > 0.5 and scale > 0.8:
                scales.append([scale, expand])
                
                # Also record the direction of the axis as it appears in image space. This is so we
                # can identify which ring has positive Z and which has negative Z
                axes.append((r @ parameterisation.get_axis().unsqueeze(1)).squeeze())

    return torch.tensor(scales), torch.stack(axes, 0)




def _main()->None:

    # Load the data. If you wish to load Bates data rather than resi, comment the top line and uncommet the one underneath
    nupc3d = [t.to(device.device).half() for t in resi_data.load_3d()]
    #nupc3d = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]

    
    # the rejection parameters is a weighting for how likely the optimisation is to reject a given patch based on quality
    # if you want the model to reject less, make the rejection parameter higher
    # if you want the model to reject more, make the rejection parameter lower
    rejection = 1.0


    # z_scale is difference in scaling between xy axis and z axis
    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 3.9,
        z_scale = 2
    )
    
    params_initial = train.TrainingParameters()
    params_initial.batch_size = 10 
    params_initial.validity_weight=rejection

    # Optimisation parameters

    # The number of epochs needs to be set so that the system has reached a stable state by the time the optimisation terminates
    # If you want to check this you would need to load the log file and plot the loss (see below)

    # The blur reduction schedule below is not the fastest (see train_nupc.py for an optimised one) but is designed to be single step and easy 
    # to understand. The main limiting factors for speed are the number of points in the model (here 700) and the number of PSF steps taken
    # Faster optimisation can be achieved by optimising an initial model with fewer points and then re-seeding.

    # The learning rate is set to allow faster optimisation at the beginning of the run, when gradients are likely to be steeper.
    # A lower learning rate can be used for the whole optimisation at the cost of a longer run time.

    params_initial.schedule[0].epochs = 90
    params_initial.schedule[0].initial_psf = 65 # in units of nm
    params_initial.schedule[0].final_psf = 33.4 # in units of nm
    params_initial.schedule[0].psf_step_every= 30
    params_initial.schedule[0].initial_lr= 0.0001
    params_initial.schedule[0].final_lr= 0.0001

    params_initial.schedule.append(train.TrainingSegment())
    params_initial.schedule[1].epochs = 900
    params_initial.schedule[1].initial_psf = 24.7
    params_initial.schedule[1].final_psf = 13
    params_initial.schedule[1].psf_step_every= 100
    params_initial.schedule[1].initial_lr= 0.0001
    params_initial.schedule[1].final_lr= 0.0001
    
    # DataSet6Plane defines how the data will be rendered, this was used for all SMLM data
    # If you wish to use non-SMLM data you will need to change to a different renderer 
    dataset = localisation_data.DataSet6Plane(**vars(data_parameters), data=nupc3d, augmentations=1, device=device.device)


    # resetting the torch compiler is advisable as it is occasionally prone to writing data into the compiled code if this is not done
    torch.compiler.reset()

    # This sets the size of the model and the parameters of the heterogeneity
    net, parameterisation =network.PredictReconstructionStretchExpandValidDan6(model_size=700, **vars(data_parameters), data=nupc3d)
    parameterisation.max_stretch_factor_axis = 2.0
    parameterisation.max_stretch_factor_expand = 1.0
    net.to(device.device)
    
    fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
    train.retrain(fast, dataset, params_initial, 'phase_0')
    


    # Second training phase
    #
    # At this point the system should have found the structure pretty well and
    # the axis of stretch. To refine the structure, we allow expansion in the
    # parameterisation and we continue the training at the finest scale with a
    # decreasing learning rate. 
    params_final = train.TrainingParameters()
    params_final.batch_size = 10
    params_final.validity_weight=rejection
    params_final.schedule[0].epochs = 1000
    params_final.schedule[0].initial_psf = 13
    params_final.schedule[0].final_psf = 13
    params_final.schedule[0].psf_step_every= 300
    params_final.schedule[0].initial_lr= 0.0001
    params_final.schedule[0].final_lr= 0.0005
    parameterisation.max_stretch_factor_expand = 1.3
    torch.compiler.reset()
    train.retrain(fast, dataset, params_final, 'phase_1')

    # Do some data plotting
    scales, _ = analyze_data(net, parameterisation, dataset)
    plt.hist(scales[:,0], 30)
    plt.xlabel('Relative scaling of the underlying model in Z')
    plt.ylabel('Count')
    plt.show()


        
if __name__ == "__main__":
    _main()



