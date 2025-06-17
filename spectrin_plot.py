# pylint: disable=unused-wildcard-import,wildcard-import]
from matplotlib.pyplot import *
from matplotlib import colormaps
import torch
import tqdm

import data.leterrier_spectrin
import train
import network
from localisation_data import LocalisationDataSetMultipleDan6, fwhm_to_sigma

state = torch.load('log/1702497804-ab7e02133694c362690fd8739f775101e73de379/run-000-phase_0/final_net.zip')

# Since we used the compiled model, the state_dict is
# prepended with '_orig_mod.'. so stripit

stripped_state_dict = { k[10:]:v for k, v in state.items()}

data_parameters = train.DataParametersXYYZ(
    image_size_xy = 64,
    image_size_z = 32,
    nm_per_pixel_xy = 25,
    z_scale = 1
)

rawdata = data.leterrier_spectrin.load_unfiltered_3d()

s_data, positions = data.leterrier_spectrin.load_3d_and_means()

data3d = [t.to(train.device).half() for t in s_data ]

net=network.PredictReconstructionStretchExpandValidDan6(model_size=300, **vars(data_parameters), data=data3d)
net.to(train.device)
net.load_state_dict(stripped_state_dict)
net.max_stretch_factor_axis = 3
net.max_stretch_factor_expand = 3


net.eval()

fwhm = 45

dataset = LocalisationDataSetMultipleDan6(**vars(data_parameters), data=data3d, augmentations=1, device=train.device)
dataset.set_sigma(fwhm_to_sigma(45))

fwhm_t = torch.tensor(fwhm, device=dataset[0][0].device)


xysl = []
scalesl=[]
indices = []

ion()
plot(rawdata[:,0], rawdata[:,1], ".", markersize=0.1, color=[0,0,0,.1])

with torch.no_grad():
    for i in tqdm.tqdm(range(len(dataset))):
        datum = [x.unsqueeze(0) for x in dataset[i]]
        reconstruction, scale, predicted_sigma, is_valid = net(datum, min_sigma_nm=fwhm_to_sigma(fwhm_t))
        
        if is_valid > 0.5:
            xysl.append(positions[i][0:2])
            scalesl.append(scale)
            indices.append(i)



scales = torch.cat(scalesl).cpu()
xys = torch.stack(xysl, 0).cpu()

s_lo = scales.min()
s_hi = scales.max()

cmap = colormaps['jet']

for index, scale, (cx, cy) in zip(indices, scales, xys):
    s = (scale - s_lo)/(s_hi-s_lo)
    plot(data3d[index][:,0].cpu()+cx, data3d[index][:,1].cpu()+cy, '.', markersize=.1, color=cmap(s))



