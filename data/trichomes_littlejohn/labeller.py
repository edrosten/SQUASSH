#Run as python -m...

from __future__ import annotations
import itertools
from pathlib import Path

from torch import Tensor
import torch

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from matplotlib.backend_bases import MouseEvent, KeyEvent

from . import Label, _R_Y, _vol_nm_scale_xyz, _vol_coords_xyz, _cut_volume, _load_and_reshape
#from . import _FILES_Oct_28, _labels_Oct_28
#from . import _FILES_Nov_26, _labels_Nov_26
#from . import _FILES_10_17_2024, _labels_10_17_2024
#stack, metadata  = _load_and_reshape(_FILES_10_17_2024)
#label_file = _labels_10_17_2024

from ..trichomes import load_dataset_1
label_file = Path(__file__).parent.parent/'trichomes'/'labels.zip'
stack, metadata  = load_dataset_1()


labels: list[Label|None] = [None] * stack.shape[0]
if label_file.exists():
    labels = torch.load(label_file)



# Generate a 3D volume (for example, a random array)
index = 0
gamma = 1

def _norm(x: Tensor)->Tensor:
    return x/x.max()


def _rotate_volume(volume: Tensor, angle: float)->Tensor:
    d = volume.device
    # Volume is rotated around it's centre around the Y axis.
    # Can't even with rectangular region
    assert volume.shape[2] == volume.shape[1]

    R = _R_Y(-angle, d).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(*volume.shape, 3, 3)

    scale = _vol_nm_scale_xyz(volume.shape, metadata).reshape(1,1,1,3).expand(*volume.shape, 3)
    xyz_nm = _vol_coords_xyz(volume.shape, d) * scale
    rotated_coords = (R @ xyz_nm.unsqueeze(-1)).squeeze(-1) / scale

    rotated_volume = torch.nn.functional.grid_sample(volume.unsqueeze(0).unsqueeze(1), rotated_coords.unsqueeze(0), padding_mode="zeros", align_corners=False)
    return rotated_volume.squeeze(0).squeeze(0)


# Function to compute max intensity projection after rotating around the Z-axis
def _max_intensity_projection(volume: Tensor, angle:float)->Tensor:
    # Rotate the volume around the Z-axis by the given angle
    #rotated_volume = scipy.ndimage.rotate(volume, angle, axes=(0, 2), reshape=False)
    rotated_volume = _rotate_volume(volume, angle * torch.pi / 180)
    return _norm(rotated_volume.max(0).values)  # MIP along the Z-axis after rotation



# Initialize the plot
fig, ax = plt.subplots(nrows=2, ncols=3)
plt.subplots_adjust(bottom=0.25)
projection = ax[0,0].imshow(_max_intensity_projection(stack[index], 0), cmap='gray')
annotate_projection_ax = ax[0,1].imshow(_max_intensity_projection(stack[index], 0), cmap='gray')
cut_projection = ax[0,2].imshow(_max_intensity_projection(stack[index], 0), cmap='gray')
proj_xy = ax[1,0].imshow(_norm(stack[index].sum(0)), cmap='gray')
proj_xz = ax[1,1].imshow(_norm(stack[index].sum(1)), cmap='gray')
proj_yz = ax[1,2].imshow(_norm(stack[index].sum(2)), cmap='gray')


# Slider for controlling the rotation angle
ax_slider = plt.axes((0.1, 0.01, 0.8, 0.03), facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Rotation Angle', 0, 360, valinit=0)


drawdata: dict[str|int, list[matplotlib.lines.Line2D]] = {}

# Update function for the slider
def _update(_:float)->None:
    angle = slider.val
    proj = _max_intensity_projection(stack[index], angle)
    projection.set_data(proj**(1/gamma))

    for d in itertools.chain(*drawdata.values()):
        d.remove()
    drawdata.clear()

    label = labels[index]
    
    point_1 = None if label is None else label.point_1
    point_2 = None if label is None else label.point_2
    annotate_angle = None if label is None else label.angle

    plt.sca(ax[0,1])
    if point_1 is not None and point_2 is not None:
        drawdata['line'] = plt.plot( *torch.stack([point_1, point_2]).permute(1,0), 'y-')
        assert annotate_angle is not None
        fixed_datum = _cut_volume(stack[index], metadata, annotate_angle * torch.pi/180, point_1, point_2)
        cut_projection.set_data(_max_intensity_projection(fixed_datum**(1/gamma), angle))
        proj_xy.set_data(_norm(fixed_datum.sum(0)))
        proj_xz.set_data(_norm(fixed_datum.sum(1)))
        proj_yz.set_data(_norm(fixed_datum.sum(2)))
    else:
        cut_projection.set_data(proj)
        proj_xy.set_data(_norm(stack[index].sum(0)))
        proj_xz.set_data(_norm(stack[index].sum(1)))
        proj_yz.set_data(_norm(stack[index].sum(2)))
            
    if point_1 is not None:
        drawdata[1] = plt.plot(*point_1, 'ro')
    if point_2 is not None:
        drawdata[2] = plt.plot(*point_2, 'go')
    if annotate_angle is None:
        annotate_projection_ax.set_data(proj**(1/gamma))
    else:
        annotate_projection_ax.set_data(_max_intensity_projection(stack[index], annotate_angle)**(1/gamma))
    

    ax[0,0].set_title(f'Max Intensity Projection at {angle:.1f}°')
    ax[0,1].set_title(f'Image {index} of {stack.shape[0]}')

    fig.canvas.draw_idle()

    torch.save(labels, label_file)

# Connect the slider to the update function
slider.on_changed(_update)



def _on_click(event: MouseEvent)->None:
    if isinstance(event, MouseEvent):
        if event.inaxes and event.inaxes != ax_slider:
            
            if event.inaxes == ax[0,0]:
                print("Clearing!")
                labels[index]=None
            elif event.inaxes == ax[0,1]:
                
                if labels[index] is None:
                    labels[index] = Label(angle=slider.val)
                
                if event.button == 1:
                    labels[index].point_1 = torch.tensor([event.xdata, event.ydata])
                elif event.button == 2:
                    labels[index].point_2 = torch.tensor([event.xdata, event.ydata])

    _update(0.)


def _on_keypress(event:KeyEvent)->None:
    global index, gamma
    print(event)
    print(event.key)
    if event.key == 'x':
        if labels[index] is not None:
            labels[index].point_1, labels[index].point_2 = labels[index].point_2, labels[index].point_1
        _update(0)
    elif event.key == 'left':
        index = max(index-1, 0)
    elif event.key == 'right':
        index = min(index+1, stack.shape[0]-1)
    elif event.key in "123456":
        gamma = (int(event.key)-1)/2.0 + 1.0
    
    print(gamma, 1/gamma)
    _update(0)


plt.connect('button_press_event', _on_click)
plt.connect('key_press_event', _on_keypress)

_update(0)


# Display the interactive plot
plt.show()
