import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


import resi_data         # noqa pylint:disable=unused-import
import mark_bates_data   # noqa pylint:disable=unused-import
import train
import device
from localisation_data import LocalisationDataSetMultipleDan6
from train_nupc import PredictReconstruction


nupc3d = [t.to(device.device).half() for t in resi_data.load_3d()]
#nupc3d = [t.to(device.device).half() for l in mark_bates_data.load_3d_list() for t in l]

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


trained_weights = torch.load('log/1765490795-dea2d3dfafefca44b9fa5e1ea06ba6b8148439c1/phase_1/final_net.zip', map_location=torch.device('cpu'))
trained_weights = { k[10:]:v for k,v in trained_weights.items()}
net.load_state_dict(trained_weights)

net.to(device.device)
net.eval()
        
loader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset.set_sigma(final_sigma)

R=3
C=6
plt.ion()

scale_list = []

plot=False

with torch.no_grad():
    for datum in tqdm(loader):

        _,_,_,is_valid,parameters = net.process_input(datum, min_sigma_nm=final_sigma_t)

        scale1 = parameterisation.max_stretch_factor_axis**torch.tanh(parameters[:,0])
        scale2 = parameterisation.max_stretch_factor_expand**torch.tanh(parameters[:,1])
        scale3 = parameterisation.max_stretch_factor_expand**torch.tanh(parameters[:,2])

        scale_list.append([scale1.item(), scale2.item(), scale3.item()])


        plt.suptitle(f'Validity = {is_valid.item():0.3}, {scale2.item():0.3}, {scale3.item():0.3}')


        if plot:
            plt.clf()
            for n, i in enumerate(datum):
                plt.subplot(R, C, n+1)
                plt.imshow(i[0,0].cpu(), cmap='grey')
                plt.axis('off')
        
            reconstruction, _, _, _ = net(datum, final_sigma_t) 
            for n, i in enumerate(reconstruction):
                plt.subplot(R, C, C + n+1)
                plt.imshow(i[0,0].cpu(), cmap='grey')
                plt.axis('off')
            
            for n, i in enumerate(train._normalized_difference(datum,reconstruction)):
                plt.subplot(R, C, 2*C + n+1)
                plt.imshow(i[0,0].cpu(), cmap='grey')
                plt.axis('off')

            plt.pause(.2)
            plt.waitforbuttonpress()


scales = torch.tensor(scale_list)
