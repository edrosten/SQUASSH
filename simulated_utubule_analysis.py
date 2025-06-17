import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from train_microtubule_sim import fixed_structure
from device import device as dev
from matplotlib.ticker import FuncFormatter


tubule0, _  = fixed_structure(35)
tubule0[:,2] = tubule0[:,2] - tubule0[:,2].mean()

plt.ion()
fig = plt.gcf()
ax = fig.add_subplot(projection='3d')
ax.scatter(*tubule0.permute(1,0))
ax.axis('equal')


tubule =  torch.tensor(np.loadtxt('basestruc_really.txt'))
tubule = tubule[:,[2,1,0]]
ax.scatter(*tubule.permute(1,0))

tubule = tubule0
N = tubule.shape[0]

# Distance between all pairs
tubule=tubule.to(dev).float()
expanded_rows = tubule.unsqueeze(0).expand(N, N, 3)
expanded_cols = expanded_rows.permute(1,0,2)
distances = ((expanded_rows - expanded_cols)**2).sum(2).sqrt()
distances = distances[*torch.triu_indices(N, N, 1)]




def fast_hist(n: torch.Tensor, d: float)->tuple[torch.Tensor, torch.Tensor]:
    counts = torch.bincount((n*d).floor().int())
    bins = (torch.arange(counts.shape[0]) + .5)/d
    return bins, counts


# Make a bunch of tubules with scatter
M = 1000


counts_1 = None
counts_2 = None
bins = None

for _ in tqdm.tqdm(range(100)):

    noisy_tubules = tubule.unsqueeze(0).expand(M, N, 3) + (torch.rand(M, N, 3, device=dev)-.5) * 1.0
    expanded_rows = noisy_tubules.unsqueeze(1).expand(M, N, N, 3)
    expanded_cols = expanded_rows.permute(0, 2, 1, 3)
    noisy_distances1 = ((expanded_rows - expanded_cols)**2)[...,2].sqrt()
    noisy_distances1 = noisy_distances1[:, *torch.triu_indices(N, N, 1)].reshape(-1)
    noisy_distances1 = noisy_distances1[noisy_distances1<50]

    b, c1 =  fast_hist(noisy_distances1, 5)

    if counts_1 is None:
        counts_1 = torch.zeros_like(c1)
    counts_1 = counts_1+ c1

    noisy_tubules = tubule.unsqueeze(0).expand(M, N, 3) + (torch.rand(M, N, 3, device=dev)-.5) * 6.5
    expanded_rows = noisy_tubules.unsqueeze(1).expand(M, N, N, 3)
    expanded_cols = expanded_rows.permute(0, 2, 1, 3)
    noisy_distances2 = ((expanded_rows - expanded_cols)**2)[...,2].sqrt()
    noisy_distances2 = noisy_distances2[:, *torch.triu_indices(N, N, 1)].reshape(-1)
    noisy_distances2 = noisy_distances2[noisy_distances2<50]

    b, c2=  fast_hist(noisy_distances2, 5)

    if counts_2 is None:
        counts_2 = torch.zeros_like(c2)
    counts_2 = counts_2 + c2
    

def cpu(*args):
    return tuple(i.cpu() for i in args)


FIGSCALE=2
cm = FIGSCALE/2.54  # centimeters in inches
FS=8*FIGSCALE

plt.figure(figsize=(8.*cm, 6.0*cm), layout="constrained")

plt.clf()
plt.plot(b[0:counts_1.shape[0]].cpu(), counts_1.cpu()/counts_1.sum().cpu(), label='σ=1.0 nm')
plt.plot(b[0:counts_2.shape[0]].cpu(), counts_2.cpu()/counts_2.sum().cpu(), label='σ=6.5 nm', color=[.8,.3,0,.5])

for i in range(1,2):
    plt.plot([8.2*i]*2, [0, 1], ':', color=(0.1216, 0.4667, 0.7059))

plt.plot([7.3*i]*2, [0, 1], '--', color=[.8,.3,0,.5])

def _sci_notation(val, pos):
    if val == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(val))))
    coeff = val / 10**exponent
    return rf"${coeff:.1f} \times 10^{{{exponent}}}$"


#formatter = FuncFormatter(lambda val, pos: f'{val:.1e}')
#plt.gca().yaxis.set_major_formatter(_sci_notation)
plt.ylim(0.0035, 0.005)
plt.xlim(0, 20)
plt.grid(True)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.xlabel('Spacing (nm)', fontsize=FS)
plt.ylabel('Relative frequency', fontsize=FS)
plt.legend(fontsize=FS)
plt.savefig('hax/mode-shift.pdf')
