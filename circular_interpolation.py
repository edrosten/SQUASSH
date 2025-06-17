'''Functions for interpolation, i.e. grid_sample on a 1D toroidal domain'''
from matplotlib.pyplot import plot, ion, subplot, axis, clf

import torch
import torch.fft


def circular_interpolation_linear(radii: torch.Tensor, angles: torch.Tensor)->torch.Tensor:
    '''Fourier interpolation over the circular domain described by radii'''

    assert radii.ndim == 2 # Batched
    assert angles.ndim == 2
    assert radii.shape[0] == angles.shape[0]

    b = radii.shape[0]
    N = angles.shape[1]
    L = radii.shape[1]

    # Circular linear interpolation
    # 0 in corresponds exactly to radii[0], so does 2pi
    units = (angles /(2*torch.pi)) %1.0
    grid = torch.stack([units*2-1, torch.zeros_like(angles)], 2).reshape(b, 1, N, 2)
    c_radii = torch.cat([radii, radii[:,0:1]], 1)
    print(c_radii.shape)
    return torch.nn.functional.grid_sample(c_radii.reshape(b,1,1,L+1), grid, mode='bilinear', align_corners=True).reshape(b, N)





def circular_interpolation_fourier(radii: torch.Tensor, angles: torch.Tensor)->torch.Tensor:
    '''Fourier interpolation over the circular domain described by radii'''
    ###########################
    #
    # For cosine, there are L//2+1 elements
    # For sine there are (L-1)//2 elements
    #
    # If x is a row vector, the the fourier transform 𝐅[x] = xFᵀ
    # where F is the fourier matrix F. RFFT truncates the number
    # of columns of F, since they are duplicates.
    #
    # F[0].imag == 0 (sine DC component is zero)
    # F[(L+1)/2].imag == 0 (if L is even)
    #
    # The sin components are negated. This is essentially an arbitrary choice
    assert radii.ndim == 2 # Batched
    assert angles.ndim == 2
    assert radii.shape[0] == angles.shape[0]


    L = radii.shape[1]
    N = angles.shape[1]
    b = radii.shape[0]

    rft: torch.Tensor = torch.fft.rfft(radii) # pylint: disable=not-callable

    # Number of frequency components
    fN = L//2+1
    tL = torch.tensor([L], device=radii.device)
    normalization = torch.cat([tL, tL.expand((L-1)//2)/2, tL.expand((L+1)%2)])

    rft /= normalization
    
    return ((angles.unsqueeze(1).expand(b, fN, N)*torch.arange(fN, device=radii.device).reshape(1,fN,1).expand(b, fN, N)).cos() * rft.real.unsqueeze(2).expand(b, fN, N)).sum(1) \
         - ((angles.unsqueeze(1).expand(b, fN, N)*torch.arange(fN, device=radii.device).reshape(1,fN,1).expand(b, fN, N)).sin() * rft.imag.unsqueeze(2).expand(b, fN, N)).sum(1)




def _adhoc_test()->None:

    # Make a couple of different shapes (circle and epitrochoid) and distorting them
    # with two different patterns
    radii = torch.tensor([
     [.0, 0.2],
     [.0, -.1],
     [.1, -.1],
     [.2, 0.2],
     [.5, 0.2],
     [1.0, 0],
     [.5, 0],
     [.2, .1],
     [.1, 0],
     [.0, 0],
    ]).permute(1,0)+1


    A = 1000
    angles = (torch.arange(0, A)/A * 2 * torch.pi).unsqueeze(0).expand(radii.shape[0], -1).clone()
    radiii = torch.ones(radii.shape[0], A)

    ##Make a spirally thing
    xx = angles[0].cos() + (angles[0]*10).cos()*0.2
    yy = angles[0].sin() + (angles[0]*10).sin()*0.2
    sangles = torch.atan2(yy, xx)
    sradii = (xx*xx + yy*yy).sqrt()

    angles[0,:] = sangles
    radiii[0,:] = sradii




    recon_linear = circular_interpolation_linear(radii, angles)
    recon_dft = circular_interpolation_fourier(radii, angles)


    ion()
    clf()
    for i in range(2):
        subplot(2,2,1+i*2)
        plot(angles[i] % (torch.pi*2), recon_linear[i], '.')
        plot(angles[i] % (torch.pi*2), recon_dft[i], '.')


        subplot(2,2,2+i*2)
        plot(angles[i].cos()*recon_linear[i]*radiii[i], angles[i].sin()*recon_linear[i]*radiii[i], '.', markersize=.5)
        plot(angles[i].cos()*recon_dft[i]*radiii[i], angles[i].sin()*recon_dft[i]*radiii[i], '.', markersize=.5)
        axis('square')


if __name__ == "__main__":
    _adhoc_test()
