import random

import math
import torch
from torch import Tensor

from train_dan_microtubules import LocalisationDataSetMultipleDan6, PredictReconstructionRepetitionD6
import device
import train
import matrix

def fixed_structure(reps: int=1)->tuple[torch.Tensor, torch.Tensor]:
    """Create a microrubule simulated structure"""
    n_per_rev = 14
    spiral_radius = 12.5
    single_turn_pitch = 12.3
    rep_pitch = 8.2

    pts = []

    for rep in range(reps):
        z_off = rep_pitch * rep
        for i in range(n_per_rev):
            angle = i * 2 * torch.pi / n_per_rev

            z = z_off + i/n_per_rev * single_turn_pitch

            x = math.cos(angle) * spiral_radius
            y = math.sin(angle) * spiral_radius

            pts.append([x, y, z])
    return torch.tensor(pts), torch.ones(len(pts)) * 0.5
                               

def _spherical_rng(N: int)->torch.Tensor:
    # Volume of a sphere radius 1 is 4/3 * pi * r^3 ~= 4.2
    # Volume of a cube of radius 1 is 2^3 = 8
    # Just over half the points are good
    # prob_ok = 4/3 * torch.pi / 8

    # Just do a bunch extra
    while True:

        v = torch.rand(10*N, 3) * 2 - 1
        spherical = v[ (v*v).sum(1) <= 1, :] 

        if spherical.shape[0] >= N:
            return spherical[0:N,:]


#No scatter or dropout
def _make_random_utubule()->Tensor:
    
    min_reps=6
    max_reps=10

    ideal_tubule, _ = fixed_structure(max_reps)
    
    
    max_pts = ideal_tubule.shape[0]
    min_pts = max_pts * min_reps // max_reps

    actual_pts = random.randint(min_pts, max_pts)

    start_pos_max = max_pts - actual_pts
    start_pos = random.randint(0, start_pos_max)

    cut_tubule = ideal_tubule[start_pos:start_pos+actual_pts, :]

    # Tubule is aligned on the Z axis, so make it be aligned to x
    # with a rotation, not reflection
    cut_tubule = cut_tubule[:, [2,0,1]]

    # Centre on x
    cut_tubule[:,0] -= cut_tubule[:,0].mean()

    cut_tubule = cut_tubule + _spherical_rng(cut_tubule.shape[0])*2.0 + torch.randn_like(cut_tubule)*2.0

    # Now rotate bout X followed by Z

    a = torch.rand(2) * torch.pi*2

    return (matrix.euler(a[0:1], 'z') @ matrix.euler(a[1:2], 'x') @ cut_tubule.permute(1,0)).squeeze(0).permute(1,0)




def _main()->None:
     
    data3d = [_make_random_utubule().to(device.device).half() for _ in range(400)]
 
 
    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 2.0,
        z_scale = 1
    )

    dataset_initial = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=data3d, augmentations=1, device=device.device)
    dataset_initial.set_batch_size(1)
    torch._dynamo.config.cache_size_limit=512

    for i in range(1):

        net, parameterisation = PredictReconstructionRepetitionD6(
            model_size=280, 
            **vars(data_parameters), 
            data=data3d,
            min_repetitions = 3,
            max_repetitions = 5
        )

        parameterisation.min_repetition_length = torch.tensor(14.)
        parameterisation.max_repetition_length = torch.tensor(18.)
        parameterisation.semi_radial_expand = torch.tensor(1.2)

        net.to(device.device)
        
        params = train.TrainingParameters()
        params.batch_size = 20
        params.validity_weight=1.0

        params.schedule[0].epochs = 3000
        params.schedule[0].initial_psf = 20
        params.schedule[0].final_psf = 4
        params.schedule[0].psf_step_every= 100
        params.schedule[0].initial_lr= 0.0001
        params.schedule[0].final_lr= 0.0001
        

        fast = torch.compile(net)
        train.retrain(fast, dataset_initial, params, f'run-{i:03}-phase_0')

if __name__ == "__main__":
    #_adhoc_test()
    _main()



def _analyze_microtuble_sim()->None:
    import matplotlib.pyplot as plt

    FIGSCALE=2
    cm = FIGSCALE/2.54  # centimeters in inches
    FS=8*FIGSCALE

    plt.figure(figsize=(6.*cm, 6.0*cm), layout="constrained")

    data3d = [_make_random_utubule().to(device.device).half() for _ in range(400)]
 
    data_parameters = train.DataParametersXYYZ(
        image_size_xy = 64,
        image_size_z = 32,
        nm_per_pixel_xy = 2.0,
        z_scale = 1
    )
        
    net, parameterisation = PredictReconstructionRepetitionD6(
        model_size=280, 
        **vars(data_parameters), 
        data=data3d,
        min_repetitions = 3,
        max_repetitions = 5
    )

    
    results={
        "1746310898-ca7519e34b90a8abfffef224a2b7edb798f8c6d4": 1.0,
        "1746347136-0024dac4e770b11fdaad5cb3db746e9796d6c7b3": 2.0,
        "1746350859-fc3adb86435c6be95ce09d1e5375217583056352": 1.5,
        "1746356251-2dabb5e9ff56100c4f7909d31629589d1a2a3623": 1.75,
    }
   
    noise_val=[]
    spacing=[]

    for k,v in results.items():
        final =  f"log/{k}/run-000-phase_0/final_net.zip"
        noise_val.append(v)
        state_dict = torch.load(final)
        state_dict = {k[10:]:v for k,v in state_dict.items()}
        net.load_state_dict(state_dict)

        print(net._parameterisation.get_spacing())
        spacing.append(net._parameterisation.get_spacing())


    plt.plot(noise_val, torch.cat(spacing).detach(), '*')
    plt.xlabel('Added noise σ (nm)', fontsize=FS)
    plt.ylabel('Spacing from SQUASSH', fontsize=FS)
    plt.xticks(fontsize=FS)
    plt.yticks(fontsize=FS)
    plt.savefig('hax/weird-spacing.pdf')

