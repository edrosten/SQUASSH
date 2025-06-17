from typing import List, cast

import tqdm
import torch
import scipy
#pylint: disable=no-name-in-module
from torch import pi, sin, cos, sqrt
import plyfile # type: ignore
import tifffile

from matrix import trn, eye
import matrix
import render
 
def random_rotation_matrix(batch_size: int=1)->torch.Tensor:
    ''' Generate a batch of random rotations'''
    # Generate uniform random rotations according to this techinque
    #
    # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf
    # https://doi.org/10.1016/B978-0-08-050755-2.50034-8
    #
    # The techinque is to compose two rotations, first rotating around the Z axis,
    # then picking a random point on the unit sphere for the Z axis to go to.
    #
    # The latter is done by picking a random plane to reflect through, then negating
    # the reflection to make it a rotation. A Householder reflection generated from
    # a random unit vector fits the bill

    x1, x2, x3 = torch.rand([3, batch_size])

    # Uniform rotation around Z
    c = cos(2*pi*x1)
    s = sin(2*pi*x1)
    o = torch.zeros(batch_size)
    l = torch.ones(batch_size)  #noqa

    # Create a batch of rotation matrices
    Rz = torch.stack((
        torch.stack(( c, s, o), dim=1),
        torch.stack((-s, c, o), dim=1),
        torch.stack(( o, o, l), dim=1)), dim=1)


    # Uniform unit row vector (matrix)
    v_row  = torch.stack((
        cos(2*pi*x2)*sqrt(x3),
        sin(2*pi*x2)*sqrt(x3),
        sqrt(1-x3)
    ), dim=1).unsqueeze(1)

    # Construct householder reflection
    H = eye(3, batch_size) - 2 * trn(v_row) @ v_row

    return -H @ Rz

def test_random_rotation_matrix()->None:
    ''' Test for correctness and uniformity of matrices'''
    B = 1000000
    r = random_rotation_matrix(B)

    #Check unit vectorness along dimensions
    assert (abs((r*r).sum(2)-1) < 1e-5).all()
    assert (abs((r*r).sum(1)-1) < 1e-5).all()

    #Check orthogonality
    assert (abs((r@trn(r))-eye(3, B)) < 1e-5).all()

    #Check determinant
    assert (abs(r.det() -1) < 1e-5).all()

    #Check for uniformity of randomness
    for i in range(3):
        for j in range(3):
            assert scipy.stats.kstest(r[:,i,j]/2+.5, 'uniform').pvalue > 0.01



def load_ply_vertices(filename: str)->torch.Tensor:
    '''
    Load a PLY file's vertices only
    Returns Nx3, i.e points as row vectors
    '''

    ply = plyfile.PlyData.read(filename)

    # There really ought to be precisely 1 vertex section in the header
    verts = [ e for e in ply if e.name == "vertex"][0]
    
    # Pretty weird if this isn't 0,1,2
    names = verts.dtype().names
    ind = (names.index('x'), names.index('y'), names.index('z'))

    # Can't think of a better way of converting, since
    # the dtype of verts is weird.
    return torch.tensor([[r[i] for i in ind] for r in verts])




def multiplicative_range(x: torch.Tensor, scale_3_sigma: float)->torch.Tensor:
    '''
    Map x to go from from scale to 1/scale (at 3 sigma probsbility)
    Capping happens at 4 sigma
    x should have zero mean, unit variance
    '''
    m = torch.tensor(4.)
    capped = torch.maximum(-m, torch.minimum(m, x))
    return cast(torch.Tensor, scale_3_sigma**(capped/3.0))


# pylint: disable=too-many-positional-arguments
def pointcloud_dataset3D(vertices: torch.Tensor,
                       size: int,
                       dropout: float,
                       teapot_size_nm: float,
                       offset_percent_sigma: float,
                       seed: int,
                       scatter_xy_nm_sigma: float,
                       scatter_z_nm_sigma: float,
                       anisotropic_scale_3_sigma: float,
                       random_rotations: bool=True
                       )->List[torch.Tensor]:

    ''' Generate a localisation dataset from the teapot '''
    save_rng = torch.get_rng_state()
    torch.manual_seed(seed)

    # Dataset is row vectors so transpose to column vectors
    # so that left multiplication with rotations works, then
    # expand out to the batch size.
    #
    # The test models are close enough to centred, so assume
    # it's +/- this size
    hi = vertices.abs().max()
    v = vertices / (hi * 2) * teapot_size_nm

    Nv = v.shape[0]
    v = trn(v).unsqueeze(0).expand(size, 3, Nv)

    if random_rotations:
        rotations = random_rotation_matrix(size)
    else:
        rotations = eye(3, size)

    xyz_scatter_sig = torch.tensor([scatter_xy_nm_sigma, scatter_xy_nm_sigma, scatter_z_nm_sigma], device=vertices.device)

    scatter = torch.randn(size, 3, Nv) * xyz_scatter_sig.unsqueeze(1).unsqueeze(0).expand_as(v)

    scales = multiplicative_range(torch.randn(size), anisotropic_scale_3_sigma)
    scale_mats = matrix.scale_along_axis_matrix(torch.tensor([1., 0, 0]), scales)

    # Projection is cropping z, which happens later
    vertices = (rotations[:,0:3,:] @ scale_mats @ v) + scatter

    data: List[torch.Tensor] = []

    for b in tqdm.tqdm(range(size)):
        keep = torch.rand(Nv) > dropout

        points = trn(vertices[b])

        pts = points[keep,:]
        #print(pts.shape)

        #replicate the data centering process
        # x, y are approximately centered with data on the way in
        # z has no native centering so it's "perfectly" centered on
        # the way in
        centre = pts.mean(0)

        #Now add in a slight shift
        centre += torch.cat((offset_percent_sigma * 0.01 * teapot_size_nm * torch.randn(2), torch.zeros(1)))
        data.append(pts - centre)

    torch.set_rng_state(save_rng)
    return data

def pointcloud_dataset(vertices: torch.Tensor,
                       size: int,
                       dropout: float=0.01,
                       teapot_size_nm: float=800,
                       offset_percent_sigma: float=10.0,
                       seed: int=0,
                       scatter_xy_nm_sigma: float=10,
                       anisotropic_scale_3_sigma: float=1
                       )->List[torch.Tensor]:
    
    ''' Generate a localisation dataset from the teapot '''
    data = pointcloud_dataset3D(vertices=vertices,
                                size=size,
                                dropout=dropout,
                                teapot_size_nm=teapot_size_nm,
                                offset_percent_sigma=offset_percent_sigma,
                                seed=seed,
                                scatter_xy_nm_sigma=scatter_xy_nm_sigma,
                                scatter_z_nm_sigma=0,
                                anisotropic_scale_3_sigma=anisotropic_scale_3_sigma)

    # Project by discarding Z
    for i,d in enumerate(data):
        data[i] = d[:,0:2]

    return data

def _write_test()->None:
    '''test'''
    dataset_vertices = load_ply_vertices('data/test_data/bunny.ply')
    t_seed =  0

    # Simulated data params
    teapot_size = 200
    t_seed =  0
    scatter_nm = 2

    bunnies3D = pointcloud_dataset3D(dataset_vertices, 
                                                 size=150, 
                                                 dropout=0.99,
                                                 teapot_size_nm=teapot_size, 
                                                 seed=t_seed, 
                                                 offset_percent_sigma=0.0,
                                                 scatter_xy_nm_sigma=scatter_nm, 
                                                 scatter_z_nm_sigma=scatter_nm,
                                                 anisotropic_scale_3_sigma=1) 
    
    S=2
    #stack = (torch.stack(montage.make_stack_multiple(bunnies3D, 2., 2.0/S, 64*S, 2, device=torch.device('cpu')), 0).permute(0, 2, 3, 1)*255).char().numpy()

    rendered =  [ render.render_multiple_scale(
                    centres=pts.unsqueeze(0),
                    sigma_xy_nm = 2. * torch.ones(1),
                    weights = torch.ones(1, pts.shape[0]),
                    nm_per_pixel_xy=2.0/S,
                    z_scale=2,
                    xy_size=64*S,
                    z_size=32*S)[0] for pts in tqdm.tqdm(bunnies3D)]
    normed = [ i / i.max() for i in rendered ]



    tifffile.imwrite('hax/many_bunnies.tiff', (torch.stack(normed,0)*255.9).to(torch.uint8).numpy(), imagej=True)
if __name__ == "__main__":
    _write_test()
