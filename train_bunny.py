from typing import cast

import torch
import generate_data
import train
import device
import network
import localisation_data
torch.set_float32_matmul_precision('high')

def _main()->None:
    t_seed =  int(torch.randint(0xffff_ffff_ffff, []).item())

    # Big model, so subsample it
    dataset_vertices = generate_data.load_ply_vertices('data/test_data/bunny.ply')

    # Simulated data params
    teapot_size = 800
    scatter_nm = 10

    bunnies3D = [ t.cuda().half() for t in 
                generate_data.pointcloud_dataset3D(dataset_vertices, 
                                                 size=6000, 
                                                 dropout=0.999,
                                                 teapot_size_nm=teapot_size, 
                                                 seed=t_seed, 
                                                 offset_percent_sigma=0.0,
                                                 scatter_xy_nm_sigma=scatter_nm, 
                                                 scatter_z_nm_sigma=scatter_nm,
                                                 anisotropic_scale_3_sigma=1.5) ]
    
    
    print("Average number of points: ", torch.tensor([a.shape[0]  for a in bunnies3D]).float().mean().item())
    data_parameters = train.DataParametersXYYZ(
        image_size_xy=64,
        image_size_z=64,
        nm_per_pixel_xy=10.,
        z_scale = 1.0
    )


    data_parameters_2d = train.DataParameters(
        image_size=64,
        nm_per_pixel=10.,
    )

    params = train.TrainingParameters()
    params.batch_size=75
    params.schedule[0].epochs = 400
    params.schedule[0].initial_psf = 160
    params.schedule[0].final_psf = 40.0
    params.schedule[0].psf_step_every= 40
    params.schedule[0].initial_lr= 0.0001
    params.schedule[0].final_lr= 0.0001

    def f3()->None:
        net = network.PredictReconstructionStretch2D(**vars(data_parameters_2d), model_size=400)
        net.to(device.device)
        torch.compiler.reset()
        fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
        
        dataset = localisation_data.LocalisationDataSet(data=bunnies3D, **vars(data_parameters_2d), device=device.device)

        train.retrain(fast, dataset, params, '2d')
    #f3()
    
    def f1()->None:
        net, parameterisation = network.PredictReconstructionStretchExpandValidDan6(**vars(data_parameters), model_size=400, data=bunnies3D)
        parameterisation.max_stretch_factor_axis = 2
        net.to(device.device)
        torch.compiler.reset()
        fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
        
        dataset = localisation_data.LocalisationDataSetMultipleDan6(data=bunnies3D, **vars(data_parameters), device=device.device)

        train.retrain(fast, dataset, params, 'dan6')
    f1()

    def f2()->None:
        net, _ = network.PredictReconstructionStretchExpandValid(**vars(data_parameters), model_size=400)
        net.to(device.device)
        torch.compiler.reset()
        fast = cast(network.GeneralPredictReconstruction, torch.compile(net))
        
        dataset = localisation_data.LocalisationDataSetMultiple(data=bunnies3D, **vars(data_parameters), device=device.device)

        train.retrain(fast, dataset, params, 'xyz-3-plane')
    #f2()


if __name__ == "__main__":
    _main()
