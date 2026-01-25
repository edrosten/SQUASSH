import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader

import data_dan_microtubules
from train_dan_microtubules import PredictReconstructionRepetitionD6
from localisation_data import LocalisationDataSetMultipleDan6

#import data_dan_microtubules
import device
import train
from train import fwhm_to_sigma
import save_ply


data3d = [u.to(device.device).half() for _,t in data_dan_microtubules.load_3d_3(segment_length=128).items() for u in t]

data_parameters = train.DataParametersXYYZ(
    image_size_xy = 64,
    image_size_z = 32,
    nm_per_pixel_xy = 2.0,
    z_scale = 1
)

dataset = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=data3d, device=device.device)
dataset.set_batch_size(1)

#final =  "log/1731367553-d11ee478119437294d1774383657b51058714d90/run-000-phase_0/final_net.zip"
#final = "log/1748018915-fdf7a45f7f8927faa7adf19f9ea2d8359aaf7527/run-000-phase_0/final_net.zip"
#final = "sample_logs/1748018915-fdf7a45f7f8927faa7adf19f9ea2d8359aaf7527-final_net-utubule.zip"
#final = "log/1747596426-981c285635c0a429e4428b4bbae8c6e2f130ce53//run-000-phase_0/final_net.zip"
final = "log/1768588470-12917b900aa0e51fe3543d8b559dc26be5c7cb6b/run-000-phase_0/final_net.zip" 

net, parameterisation = PredictReconstructionRepetitionD6(
    model_size=280, 
    **vars(data_parameters), 
    data=data3d,
    min_repetitions = 3,
    max_repetitions = 5
)


state_dict = torch.load(final)
state_dict = {k[10:]:v for k,v in state_dict.items()}

net.load_state_dict(state_dict)
net.to(device.device)
net.eval()

fwhm = 8
dataset.set_sigma(fwhm_to_sigma(fwhm))

loader = DataLoader(dataset, batch_size=1, shuffle=False)
fwhm_t = torch.tensor(fwhm)
expansions = []
for batch in tqdm.tqdm(loader):
    _,R,_,is_valid,parameters = net.process_input(batch, min_sigma_nm=fwhm_to_sigma(fwhm_t))
    if is_valid > 0.5 :
        expansions.append(parameterisation.compute_semi_radial_expansion_from_parameters(parameters))


net.to("cpu")
model, intensities, _ = parameterisation(*net.get_model(), torch.tensor([[10.,10., 0., 0.]]))
both=torch.cat([model[0].detach(), intensities[0].unsqueeze(1).detach()], 1)
np.savetxt('hax/tubule.txt', both)


points = model[0].detach()
weights = intensities[0].detach()


centre = points.mean(0)
distance = (points - centre.unsqueeze(0).expand_as(points)).pow(2).sum(1)

filt=0.95

_, ordered_indices = distance.sort()
last = int(points.shape[0] * filt+.5)
points = points[ordered_indices[0:last], :]
weights = weights[ordered_indices[0:last]]



save_ply.save_pointcloud_as_mesh('hax/microtubule.ply', points, weights, sigma=2.0, threshold=0.2)

