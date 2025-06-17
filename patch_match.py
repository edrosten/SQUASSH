import itertools

import torch
from torch import Tensor
import torchvision


# Window indexing
#
# What we want is something like:
# 
#  Image:
#   a b d c e f
#   g h i j k l
#   m n o p q r
#   s t u v w x 
#   
#  With 3x3 patches:
#  
#  Patches
#   a b d    Row1
#   g h i
#   m n o
#
#   b d c
#   h i j
#   n o p
#
#   d c e
#   i j k
#   o p q
#
#   c e f
#   j k l
#   p q r
#
#   g h i   Row 2
#   m n o
#   s t u
#
#   etc...
#
# Strategy: 
# 1. Flatten the image to 1D
# 2. Generate 1D indices for each 2D patche
# 3. Concatenate the indices
# 4. One large bulk indexing operation to get all flattened patches concatenated
# 5. Reshape to undo the concatenation (and flattening)

def image_to_patches(img: torch.Tensor, patch_size: tuple[int,int], stride: int=1)->tuple[torch.Tensor,torch.Tensor]:
    """
        Extract overlapping patches from img.

        Also returns the (r,c) position of the top-left of the patches
    """
    
    assert img.ndim == 2, "LOL, no channel support har de har"

    rows = img.shape[0]
    cols = img.shape[1]

    patch_v, patch_h = patch_size

    # Create a basic patch and then shift by adding multiples of 1 and cols
    patch_column_indices = torch.arange(patch_h).unsqueeze(0).expand(patch_v,patch_h)
    patch_row_indices =  torch.arange(patch_v).unsqueeze(1).expand(patch_v,patch_h)
    patch_indices = (patch_column_indices + patch_row_indices*cols).view(-1)

    # Create the set of shifts
    shift_h = cols - patch_h + 1
    shift_v = rows - patch_v + 1
    
    # How many elements after striding?
    #
    # 0 1 2 3 4 5 6   N=7                        0 1 2 3 4 5 6 7  N=8                       
    # 0 1 2           Stride = 1,  N=5 (N-3+1)   0 1 2            Stride = 1,  N=6 (N-3+1)  
    #   1 2 3                                      1 2 3                                   
    #     2 3 4                                      2 3 4                                 
    #       3 4 5                                      3 4 5                               
    #         4 5 6                                      4 5 6                             
    #                                                      5 6 7                             
    #                                                                                      
    # 0 1 2           Stride = 2, N=3            0 1 2           Stride = 2, N=3          
    #     2 3 4                                      2 3 4                                 
    #         4 5 6                                      4 5 6                             
    #
    # 0 1 2           Stride = 3,  N=2           0 1 2            Stride = 3,  N=2
    #       3 4 5                                      3 4 5                               

    # first = 0
    # last = N-3

    # n = (last + 1 + (stride -1)) // stride
    # n = (last + stride) // stride

    # 0 1 2 3 4
    # 0   2   4
    # 0     3
        


    n_shift_h = (shift_h-1+stride)//stride
    n_shift_v = (shift_v-1+stride)//stride
    shift_column_indices = torch.arange(0,shift_h,stride).unsqueeze(0).expand(n_shift_v,n_shift_h)
    shift_row_indices =  torch.arange(0,shift_v,stride).unsqueeze(1).expand(n_shift_v,n_shift_h)
    shift_indices = (shift_column_indices + shift_row_indices*cols).view(-1)

    # Now take the outer product so patches at all shifts exist
    dup_patch_indices = patch_indices.unsqueeze(0).expand(shift_indices.numel(), -1)
    dup_shift_indices = shift_indices.unsqueeze(1).expand(-1, patch_h*patch_v)

    all_patch_indices = (dup_patch_indices + dup_shift_indices).view(-1)

    all_patches = img.reshape(-1)[all_patch_indices].reshape(n_shift_v, n_shift_h, patch_v, patch_h)

    return all_patches, torch.stack((shift_row_indices, shift_column_indices), -1)


# Doesn't work on 1x1 kernels (harmless!)
def _image_to_patches2(img: torch.Tensor, patch_size: tuple[int,int], stride:int=1)->torch.Tensor:
    """
        Extract overlapping patches from img.

        Also returns the (r,c) position of the top-left of the patches
    """
    
    assert img.ndim == 2, "LOL, no channel support har de har"
    assert patch_size != (1,1)
    rows = img.shape[0]
    cols = img.shape[1]

    patch_v, patch_h = patch_size

    # Create the set of shifts
    shift_h = cols - patch_h + 1
    shift_v = rows - patch_v + 1
    n_shift_h = (shift_h-1+stride)//stride
    n_shift_v = (shift_v-1+stride)//stride
   
    return torch.nn.functional.unfold(img.view(1,1,*img.shape), kernel_size=patch_size, stride=stride).squeeze(0).squeeze(0).permute(1,0).reshape(n_shift_v, n_shift_h, *patch_size)

 
def _test()->None:

    t = torch.tensor([
        [ 0,  1,  2,  3,  4,  5.],
        [ 6,  7,  8,  9, 10, 11],
        [12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])

    assert (image_to_patches(t, (3,2))[0][1,2] == torch.tensor(
        [[ 8,  9],
         [14, 15],
         [20, 21.]])).all()

    assert (image_to_patches(t, (1,3))[0][2,3] == torch.tensor(
        [[ 15, 16, 17]])).all()

    # Test that post cropping matches pre cropping and that passing in non contiguous images
    # works as expected
    assert (image_to_patches(t[1:-1,1:-1], (2,2))[0] == image_to_patches(t, (2,2))[0][1:-1,1:-1]).all()

    assert (image_to_patches(t, (3,2), stride=2)[0][0,0] == torch.tensor(
        [[ 0,  1],
         [ 6,  7],
         [12, 13]])).all()

    assert (image_to_patches(t, (3,2), stride=2)[0][0,1] == torch.tensor(
        [[ 2,  3],
         [ 8,  9],
         [14, 15]])).all()

    
    # Exhaustively teat a load of combinations
    t = torch.rand(57, 2*3*4*5)
    for ks in itertools.islice(itertools.product(range(1,7), range(1,7)), 1, None):
        for stride in range(1,10):
            ptchs = image_to_patches(t, ks, stride=stride)[0]

            for rr,r in enumerate(range(0, t.shape[0]-ks[0]+1, stride)):
                for cc,c in enumerate(range(0, t.shape[1]-ks[1]+1, stride)):
                    ptch2 = t[r:r+ks[0], c:c+ks[1]]
                    assert (ptchs[rr,cc] == ptch2).all()

if __name__ == "__main__":
    _test()



def patchwise_cross_correlation(img1: Tensor, img2: Tensor, area: int, kernel: int, stride: int)->tuple[Tensor, Tensor]:
    ''' 
    Search for patches from img2 of size kernel in image 1 over area of size area

    Stride determines the density of the patches.

    Returns the result of the correlation and the cooordinates of the patches.

    NOTE PATCH CENTRES ARE ROW,COL, not X,Y
    '''
    
    img1_patches, img1_patch_corners = image_to_patches(img1, (area, area), stride)
    img1_patch_centres = img1_patch_corners + (area-1)/2

    offset = (area-kernel)//2
    img2_patches, img2_patch_corners = image_to_patches(img2[offset:-offset,offset:-offset], (kernel,kernel), stride)
    img2_patch_centres = img2_patch_corners + (kernel-1)/2 + offset

    assert (img2_patch_centres == img1_patch_centres).all()

    # Move the positional arrangement of patches into the channels
    img1_patches_as_channel = img1_patches.reshape(1, img1_patches.shape[0]*img1_patches.shape[1], img1_patches.shape[2], img1_patches.shape[3])
    img2_patches_as_channel = img2_patches.reshape(img2_patches.shape[0]*img2_patches.shape[1], 1, img2_patches.shape[2], img2_patches.shape[3])

    corr = torch.nn.functional.conv2d(img1_patches_as_channel, img2_patches_as_channel, groups=img1_patches_as_channel.shape[1])   # pylint: disable=not-callable
    corr = corr.reshape(*img2_patches.shape[0:2], *corr.shape[2:])
    
    return corr, img1_patch_centres


def argmax_2D(corr: torch.Tensor)->torch.Tensor:
    '''
    Find the argmax (returned as row,col), for all the 2D patches passed in.
    

    Assume the last two dims are the patches
    '''
    corr_cols = corr.shape[-1]
    corr_rows = corr.shape[-2]

    max_inds = corr.reshape(*corr.shape[0:-2], corr_cols*corr_rows).max(dim=-1)[1]
    max_inds_row = max_inds // corr_cols
    max_inds_col = max_inds % corr_cols
    
    return torch.stack((max_inds_row, max_inds_col), -1)

def patchwise_matching(img1: Tensor, img2: Tensor, area: int, kernel: int, stride: int)->tuple[Tensor, Tensor]:
    ''' 
    Search for patches from img2 of size kernel in image 1 over area of size area

    Stride determines the density of the patches.

    Returns the cooordinates of the patches and matched positions

    NOTE PATCH CENTRES ARE ROW,COL, not X,Y
    '''
    corr, patch_centres = patchwise_cross_correlation(img1, img2, area, kernel, stride)
    maxima_pos = argmax_2D(corr)
    return patch_centres, patch_centres + maxima_pos - (area -kernel)/2


def zero_normalize(img: Tensor, kernel: int)->Tensor:
    '''
    Zero-normalize all patches independently.

    Preprocessing with this will turn NCC into zNCC
    '''
    return (img -  torch.nn.functional.conv2d(img.reshape(1,1,*img.shape), torch.ones(1, 1, kernel, kernel), padding='same')/(kernel*kernel)).reshape_as(img)  # pylint: disable=not-callable


def _adhoc_test_3()->None:
    import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel
    # from mpl_toolkits.axes_grid1 import ImageGrid
    frog1 = torch.nn.functional.avg_pool2d(torchvision.io.read_image('data/test_images/Bullfrog1_out.png').float(), kernel_size=(6,6))[0] # pylint: disable=not-callable
    frog2 = torch.nn.functional.avg_pool2d(torchvision.io.read_image('data/test_images/Bullfrog2_out.png').float(), kernel_size=(6,6))[0] # pylint: disable=not-callable

    area = 31
    kernel = 15
    stride = 5

    # Normalize the images so this isn't just a brightest point detector
    frog1 = zero_normalize(frog1, kernel)
    frog2 = zero_normalize(frog2, kernel)


    centres, matches = patchwise_matching(frog1, frog2, area, kernel, stride)

    plt.figure()
    plt.imshow(frog1)

    for (r,c), (mr, mc) in zip(centres.reshape(-1, 2), matches.reshape(-1, 2)):
        plt.plot([c, mc], [r, mr], 'r')




def _adhoc_test_2()->None:
    import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel
    # from mpl_toolkits.axes_grid1 import ImageGrid
    frog1 = torch.nn.functional.avg_pool2d(torchvision.io.read_image('data/test_images/Bullfrog1_out.png').float(), kernel_size=(6,6))[0] # pylint: disable=not-callable
    frog2 = torch.nn.functional.avg_pool2d(torchvision.io.read_image('data/test_images/Bullfrog2_out.png').float(), kernel_size=(6,6))[0] # pylint: disable=not-callable



    area = 31
    kernel = 15
    stride = 5

    # Normalize the images so this isn't just a brightest point detector
    frog1 = (frog1 -  torch.nn.functional.conv2d(frog1.reshape(1,1,*frog1.shape), torch.ones(1, 1, kernel, kernel), padding='same')/(kernel*kernel)).squeeze()  # pylint: disable=not-callable
    frog2 = (frog2 -  torch.nn.functional.conv2d(frog2.reshape(1,1,*frog1.shape), torch.ones(1, 1, kernel, kernel), padding='same')/(kernel*kernel)).squeeze()  # pylint: disable=not-callable



    frog1_patches, frog1_patch_corners = image_to_patches(frog1, (area,area), stride=stride)


    frog1_patch_centres = frog1_patch_corners + (area-1)/2

    offset = (area-kernel)//2

    frog2_patches, frog2_patch_corners = image_to_patches(frog2[offset:-offset,offset:-offset], (kernel,kernel), stride)
    frog2_patch_centres = frog2_patch_corners + (kernel-1)/2 + offset

    assert (frog2_patch_centres == frog1_patch_centres).all()


    #   plt.ion()
    #
    #   fig = plt.figure()
    #   grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                    nrows_ncols=(12, 20),  # creates 2x2 grid of Axes
    #                    axes_pad=0.1,  # pad between Axes in inch.
    #                    )
    #   for ax, im in zip(grid, frog1_patches.reshape(-1,31,31)):
    #       ax.imshow(im)
    #       ax.axis('off')
    #
    #   fig = plt.figure()
    #   grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                    nrows_ncols=(12, 20),  # creates 2x2 grid of Axes
    #                    axes_pad=0.1,  # pad between Axes in inch.
    #                    )
    #   for ax, im in zip(grid, frog2_patches.reshape(-1,11,11)):
    #       ax.imshow(im)
    #       ax.axis('off')
    #

    # Perform the correlation

    # Move the positional arrangement of patches into the channels
    frog1_patches_as_channel = frog1_patches.reshape(1, frog1_patches.shape[0]*frog1_patches.shape[1], frog1_patches.shape[2], frog1_patches.shape[3])
    frog2_patches_as_channel = frog2_patches.reshape(frog2_patches.shape[0]*frog2_patches.shape[1], 1, frog2_patches.shape[2], frog2_patches.shape[3])


    corr = torch.nn.functional.conv2d(frog1_patches_as_channel, frog2_patches_as_channel, groups=frog1_patches_as_channel.shape[1])   # pylint: disable=not-callable
    corr = corr.reshape(*frog2_patches.shape[0:2], *corr.shape[2:])
    max_inds = corr.reshape(*corr.shape[0:2], -1).max(dim=2)[1]
    max_inds_row = max_inds // (area-kernel+1)
    max_inds_col = max_inds % (area-kernel+1)


    #   plt.ion()
    #
    #   fig = plt.figure()
    #   grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                    nrows_ncols=corr.shape[0:2],  # creates 2x2 grid of Axes
    #                    axes_pad=0.1,  # pad between Axes in inch.
    #                    )
    #   for ax, im, r, c in zip(grid, corr.reshape(-1,*corr.shape[2:]), max_inds_row.reshape(-1), max_inds_col.reshape(-1)):
    #       ax.imshow(im)
    #       plt.sca(ax)
    #       plt.plot(c, r, 'r*')
    #       ax.axis('off')


    plt.figure()
    plt.imshow(frog1)

    for (r,c), dr, dc in zip(frog1_patch_centres.reshape(-1, 2), max_inds_row.reshape(-1), max_inds_col.reshape(-1)):
        plt.plot([c, c+dc - (area-kernel+1)/2], [r, r+dr - (area-kernel+1)/2], 'r')



