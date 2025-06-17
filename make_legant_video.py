import os
import tifffile

import torch
from matplotlib.pyplot import axis, gcf, subplot_mosaic, imshow, clf, style, sca, title, xlabel, ylabel, hist, pause, savefig, ion
import data.Legant
import volumetric
import train
import device

import train_legant

os.mkdir('hax/anim')

#data, metadata, data_fullsize, fmetadata = data.Legant.load_and_crop_extra('560nm')

data, data642, metadata, data_fullsize, data_fullsize642, fmetadata = data.Legant.load_and_crop_both()


dataset = volumetric.SimpleVolumetric3Plane(data, metadata, device.device)

sigma=train.fwhm_to_sigma(500)

net, parameterisation = train_legant.PredictReconstruction(
        model_size=500,
        nm_per_pixel_xy=metadata.xy_nm_pix,
        image_size_xy=32,
        z_scale=metadata.z_nm_pix/metadata.z_nm_pix,
        image_size_z=32)


net.to(device.device)
state = torch.load('log/1716672695-c679fce89a1db197c40c0ea6fc68588144d80a80-unclean/run-000-phase_1/final_net.zip', map_location=device.device)
state = torch.load('log/1740952805-815901218db4d71fc6b57d883a3a5cecbcad504c/run-000-phase_1/final_net.zip', map_location=device.device)

mapped = {k[10:]:v for k,v in state.items()}

net.load_state_dict(mapped)
net.eval()



dataset.set_sigma(sigma)

loader = torch.utils.data.DataLoader(dataset, batch_size=1)

ion()
sigmat = torch.tensor(sigma).to(device.device)

spacingsl = []
sizesl = []
for batch in loader:

    recon, _, _, _, = net(batch, sigmat)

    r, _, _, _, parameters = net.process_input(batch, sigmat)

    spacing = parameterisation.compute_spacing_from_parameters(parameters)
    expansion = parameterisation.compute_expansion_from_parameters(parameters)

    spacingsl.append(spacing.item())
    sizesl.append(expansion.item())

    #subplot(1,2,1)
    #imshow(batch[0].squeeze().cpu())
    #subplot(1,2,2)
    #imshow(recon[0].squeeze().cpu().detach())
    #waitforbuttonpress()
spacings = torch.tensor(spacingsl)
sizes = torch.tensor(sizesl)

spcs = spacings.sort().values
inds = spacings.sort().indices

# Reorder data in order od spacing
data2 = data[inds]
data_fullsize2 = data_fullsize[inds]
data_fullsize642_2 = data_fullsize642[inds]

dataset = volumetric.SimpleVolumetric3Plane(data2, metadata, device.device)
dataset.set_sigma(sigma)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)


# Now render the original data, ordered by spacing and rotated so that
# the axis of expansion aligns with 1,0,0 

# Construct a rotation matric to make R*axis = (1,0,0)
# so it aligns with the view

# First row of R, a.a is trvially 1
a = parameterisation.get_axis().cpu().detach()   # a.a is trivialy 1

# Construct the second row
v1 = a.cross(torch.tensor([1,0.,0]))   # a.v1 is 0
v = torch.nn.functional.normalize(v1, dim=0)

# Final row, orthogonal to v and a
w = v.cross(a)

modelR = torch.stack([a, v, w], 0).to(parameterisation.get_axis()).permute(1,0)

clf()
style.use('dark_background')
fig, axes = subplot_mosaic(''' 
BCEAA
FGHAA
DDDAA
''')

gcf().text(0.1, 0.7, 'Z projection', rotation=90)
gcf().text(0.1, 0.45, 'X projection', rotation=90)
def clear_all()->None:
    for i in axes.values():
        i.clear()

def norm(A: torch.Tensor)->torch.Tensor:
    return A/A.sum()

chromatin_l: list[torch.Tensor] = []
tubulin_l: list[torch.Tensor] = []

for i, (batch, spc) in enumerate(zip(loader, spcs)):

    recon, _, _, _, = net(batch, sigmat)
    t, r, _, _, parameters = net.process_input(batch, sigmat)
    
    cube = data_fullsize2[i]

    # t is in nm
    #Image size in nm.
    sx = cube.shape[-1] * fmetadata.xy_nm_pix
    sy = cube.shape[-2] * fmetadata.xy_nm_pix
    sz = cube.shape[-3] * fmetadata.z_nm_pix
    # The transform is +/-1
    t_norm = t / torch.tensor([[sx, sy, sz]]).to(t) * 2 
    
    # Need to scale z (and y but it's the same as x) to match x
    z = sz/sx
    z_scale = torch.tensor([
        [ 1., 0, 0, 0],
        [ 0., 1, 0, 0],
        [ 0., 0, z, 0],
        [ 0., 0, 0, 1]
    ])

    iz_scale = torch.tensor([
        [ 1., 0, 0, 0],
        [ 0., 1, 0, 0],
        [ 0., 0,1/z, 0],
        [ 0., 0, 0, 1]
    ])


    # Wait do I need to muliply t by r/rinverse?
    A = iz_scale @ torch.cat([
        torch.cat([r@modelR, -t_norm.unsqueeze(2)], -1).detach().cpu(),
        torch.tensor([[[0, 0, 0, 1.]]])
    ], 1) @ z_scale



    grid = torch.nn.functional.affine_grid(A[:,0:3,:], list(cube.unsqueeze(0).unsqueeze(0).shape))

    resampled = torch.nn.functional.grid_sample(cube.unsqueeze(0).unsqueeze(0), grid).squeeze(0).squeeze(0)
    resampled2 = torch.nn.functional.grid_sample(data_fullsize642_2[i].unsqueeze(0).unsqueeze(0), grid).squeeze(0).squeeze(0)

    chromatin_l.append(resampled)
    tubulin_l.append(resampled2)

    for j in range(3): 
        clear_all()
        sca(axes['A'])

        if j >= 2:
            img = torch.zeros(3, *resampled.sum(0).shape)

            rs = resampled.sum(0)
            rs /= rs.max()

            rs2 = resampled2.sum(0)
            rs2 /= rs2.max()

            img[0,:,:] = rs
            img[2,:,:] = rs
            img[1,:,:] = rs2
            
            imshow(img.permute(1,2,0))
            title('Aligned data')
        axis('off')

        
        sca(axes['B'])
        imshow(batch[0].squeeze().cpu())
        axis('off')
        title('Data', fontsize=8)

        
        sca(axes['F'])
        imshow(batch[1].squeeze().cpu())
        axis('off')
        title('Data', fontsize=8)


        sca(axes['C'])
        if j >= 1:
            imshow(recon[0].squeeze().cpu().detach())
            title('Reconstruction', fontsize=8)
        axis('off')

        sca(axes['E'])
        if j >= 1:
            imshow((norm(batch[0])-norm(recon[0])).squeeze().cpu().detach(), cmap='gray')
            title('Difference', fontsize=8)
        axis('off')

        sca(axes['G'])
        if j >= 1:
            imshow(recon[1].squeeze().cpu().detach())
            title('Reconstruction', fontsize=8)
        axis('off')

        sca(axes['H'])
        if j >= 1:
            imshow((norm(batch[1])-norm(recon[1])).squeeze().cpu().detach(), cmap='gray')
            title('Difference', fontsize=8)
        axis('off')


        sca(axes['D'])

        if j >= 2:
            count, start, bins = hist(spacings, 10)
            bin_ind =  (spc >= torch.tensor(start[0:-1])).nonzero()[-1].item()
            bins[bin_ind].set(color='red')
            xlabel('Spacing (nm)')
            ylabel('Frequency')
        else:
            axis('off')



        
        pause(.1)
        savefig(f'hax/anim/img-{j}-{i:04}.png')
        #savefig(f'hax/anim/img-{j}-{i:04}.svg')


aligned_tubulin = torch.stack(tubulin_l)
aligned_chromatin = torch.stack(chromatin_l)
tifffile.imwrite('hax/aligned_tubulin.tiff', aligned_tubulin.numpy(), imagej=True, resolution=(1/fmetadata.xy_nm_pix, 1/fmetadata.xy_nm_pix),  metadata={'axes': 'TZYX', "unit":"nm", 'spacing':fmetadata.z_nm_pix}, compression='zlib', compressionargs={'level':9})
tifffile.imwrite('hax/aligned_chromatin.tiff', aligned_chromatin.numpy(), imagej=True, resolution=(1/fmetadata.xy_nm_pix, 1/fmetadata.xy_nm_pix),  metadata={'axes': 'TZYX', "unit":"nm", 'spacing':fmetadata.z_nm_pix}, compression='zlib', compressionargs={'level':9})

