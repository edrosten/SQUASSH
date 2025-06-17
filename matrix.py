from typing import Optional
import torch
from torch import nn
def trn(t: torch.Tensor)->torch.Tensor:
    ''' Perform a matrix transpose of the tensor, i.e. the last two dims'''
    return t.transpose(t.ndim-1, t.ndim-2)

def eye(size:int, batch_size:int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None)->torch.Tensor:
    '''Batched identity matrix'''
    return torch.eye(size, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, size, size)


def batch_dot(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    '''Batched dot product of the second dimension'''
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape == y.shape
    return (x.unsqueeze(1) @ y.unsqueeze(2)).squeeze(2).squeeze(1)

def test_batched_dot()->None:
    '''Check the dot product function'''
    a = torch.randn(1000,4)
    b = torch.randn(1000,4)
    assert (batch_dot(a,b) - (a*b).sum(1)).abs().max() < 1e-6

def inner_dot(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    '''Dot products on the innermost dimension'''
    assert x.shape == y.shape
    inner = x.shape[-1]
    return batch_dot(x.reshape(-1, inner), y.reshape(-1, inner)).reshape(*x.shape[:-1])


def test_inner_dot()->None:
    '''Check the dot product function'''
    a = torch.randn(100,5,3)
    b = torch.randn(100,5,3)
    c =inner_dot(a, b)
    assert c.shape == torch.Size((100,5))
    assert (c - (a*b).sum(2)).abs().max() < 1e-6



def normalized_gram_schmidt_reduce(x: torch.Tensor, y: torch.Tensor)->torch.Tensor:
    '''
    Return y Gram-Schmidt reduced to be orthogonal to x
    x must be a unit vector
    A unit vector is returned
    This is rather precision sensitive.
    '''
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape == y.shape
    return nn.functional.normalize(y - batch_dot(x,y).unsqueeze(1).expand(x.shape)*x)



def test_normalized_gram_schmidt_reduce()->None:
    '''Check normalized_gram_schmidt_reduce'''

    a = torch.nn.functional.normalize(torch.randn(10000,3))
    b = torch.nn.functional.normalize(torch.randn(10000,3))

    c = normalized_gram_schmidt_reduce(a, b)

    assert batch_dot(a,c).abs().max() < 1e-4


def so3_6D(x: torch.Tensor)->torch.Tensor:
    '''
    Turn the input into a rotation matrix in 3D.
    The input must be 6DoF, essentially representing two
    vectors in ℝ³. The first is normalized, creating the rotation matrix.
    The second is gram-schmid reduced to be orthogonal to the first, then
    normalized, creating the second row. The third row is then constructed
    by cross producting the first.

    See:
    https://arxiv.org/abs/1812.07035
    '''

    assert x.ndim == 2
    assert x.shape[1] == 6

    r1 = nn.functional.normalize(x[:,0:3])
    r2 = normalized_gram_schmidt_reduce(r1, x[:,3:6])
    r3 = torch.linalg.cross(r1, r2, dim=1) # pylint: disable=not-callable

    # These are all nx3
    return torch.stack((r1, r2, r3), 1)


def scale_along_axis_matrix(v: torch.Tensor, scale: torch.Tensor)->torch.Tensor:
    '''
    Let 𝐯 be the input, a column vector in ℝ³
    𝐯̂ is the corresponding unit vector.

    The outer product matrix is  𝐯̂ 𝐯̂ᵀ.

    The eigendecomposition is based on:
    𝐯̂ 𝐯̂ᵀ 𝐱 = λ𝐱
    By inspection 𝐱 ∝ 𝐯̂ gives λ = 1
    A vector 𝐱₂ orthogonal to 𝐱 gives  λ = 0
    𝐱₃ = 𝐱₂ ⨯ 𝐱, must also be orthogonal to 𝐱 and so also gives  λ = 0

    So:
            [   |    |   ] [ 1     ] [―――𝐯̂―――]
    𝐯̂ 𝐯̂ᵀ =  [ 𝐯̂ | 𝐱₂ | 𝐱₃] [   0   ] [―――𝐱₂――]
            [   |    |   ] [     0 ] [―――𝐱₃――]

    Clearly from inspection, the RHS multiplies out to the LHS. We shall
    define:

         [―――𝐯̂―――] 
    𝐑 =  [―――𝐱₂――]
         [―――𝐱₃――]

    Now consider the expression:

    𝐒 = 𝐈³ + s 𝐯̂ 𝐯̂ᵀ

    Expanding all matrices using a valid eigendecomposition gives:

           [ 1     ]           [ 1     ]
    𝐒 = 𝐑ᵀ [   1   ] 𝐑  + s 𝐑ᵀ [   0   ] 𝐑 
           [     1 ]           [     0 ]

    giving

                         [1+s     ]
    𝐒 = 𝐈³ + s 𝐯̂ 𝐯̂ᵀ = 𝐑ᵀ [    1   ] 𝐑 
                         [      1 ]

    𝐒 therefore rotates the world so the x axis is aligned to 𝐯, scales that 
    axis and then rotates back, in other words 𝐒 scales along 𝐯. The scale is
    1+s.

    '''
    assert v.ndim == 1
    assert v.shape[0] == 3
    assert scale.ndim == 1

    vhat = nn.functional.normalize(v.unsqueeze(0)).squeeze(0)

    vvt = vhat.outer(vhat)
    # 1+s = scale
    s = scale -1

    eyes = torch.eye(3, dtype=v.dtype, device=v.device).unsqueeze(0).expand(len(scale), 3, 3)

    return eyes + s.unsqueeze(1).unsqueeze(1).expand(eyes.shape) * vvt.unsqueeze(0).expand(eyes.shape)


def scale_along_axis_and_expand_matrix(v: torch.Tensor, axis_scale: torch.Tensor, expand: torch.Tensor)->torch.Tensor:
    '''
    Let 𝐯 be the input, a column vector in ℝ³

    This function scales bu axis_scale (a) along 𝐯̂ and by expand (b) uniformly
    orthogonal to 𝐯̂.

    One axis and a batch of scales and expansions is expected

    Following the derivation of scale_along_axis_matrix:

    let:

    𝐒 = 𝛽𝐈³ + 𝛼𝐯̂ 𝐯̂ᵀ

    Expanding gives:

           [𝛽+𝛼     ]
    𝐒 = 𝐑ᵀ [    𝛽   ] 𝐑 
           [      𝛽 ]

    Clearly:
    𝛽 = b = expand
    𝛼 = a - b = axis_scale - expand
    
    Note that if expand=1 (no expansion), this is identical to scale_along_axis_matrix.
    '''
    assert v.ndim == 1
    assert v.shape[0] == 3
    assert axis_scale.ndim == 1
    assert expand.ndim == 1
    assert axis_scale.shape == expand.shape

    vhat = nn.functional.normalize(v.unsqueeze(0)).squeeze(0)

    vvt = vhat.outer(vhat)
    # 1+s = scale
    alpha = axis_scale - expand
    beta = expand

    eyes = torch.eye(3, dtype=v.dtype, device=v.device).unsqueeze(0).expand(len(axis_scale), 3, 3)

    return beta.unsqueeze(1).unsqueeze(1).expand(eyes.shape)* eyes + \
        alpha.unsqueeze(1).unsqueeze(1).expand(eyes.shape) * vvt.unsqueeze(0).expand(eyes.shape)


def test_scale_along_axis_matrix()->None:
    '''Test scaling function, by scaling a Gaussian blob of points and checking the covariance''' 
    B = 30
    N = 10_000_000
    
    data = torch.randn(B, N, 3)
    direction = torch.tensor([1., 2., 3.])
    scale = torch.rand(B)*3 + .5

    M = scale_along_axis_matrix(direction, scale)
    assert M.ndim == 3
    assert M.shape[1] == 3
    assert M.shape[2] == 3

    d = trn(M @ trn(data))

    failures=0

    for i in range(B):
        x = torch.cov(trn(d[i]))
        L, R = torch.linalg.eigh(x) # pylint: disable=not-callable

        #Make the non unitary one be at the beginning
        if scale[i] > 1:   
            L = torch.tensor([L[2],L[0], L[1]])
            R = torch.stack([-R[:,2], R[:,0], R[:,1]],1)
        
        # Std dev should match the scale
        if not abs(scale[i] - torch.sqrt(L[0])) < 1e-2:
            failures+=1

        
        # Rotation matrix should line up (or be negative) with direction
        norm_dir_col = (direction/torch.sqrt((direction*direction).sum())).unsqueeze(1)
        if not abs(torch.mm(trn(R), norm_dir_col).squeeze(1).abs() - torch.tensor([1.,0,0])).max() < 1e-2:
            failures+=1
    
    # Conservative?
    assert failures < 4


def test_scale_along_axis_and_expand_matrix()->None:
    '''Test scaling function, by scaling a Gaussian blob of points and checking the covariance''' 
    B = 30
    N = 10_000_000
    
    data = torch.randn(B, N, 3)
    direction = torch.tensor([1., 2., 3.])
    scale = torch.rand(B)*3 + .5
    scale2 = torch.rand(B)*3 + .5

    M = scale_along_axis_and_expand_matrix(direction, scale, scale2)
    assert M.ndim == 3
    assert M.shape[1] == 3
    assert M.shape[2] == 3

    d = trn(M @ trn(data))

    failures=0

    for i in range(B):
        x = torch.cov(trn(d[i]))
        L, R = torch.linalg.eigh(x) # pylint: disable=not-callable

        # L is ordered smallest to biggest
        #Reorder to match
        if scale[i] > scale2[i]:   
            L = torch.tensor([L[2],L[0], L[1]])
            R = torch.stack([-R[:,2], R[:,0], R[:,1]],1)

        # Std dev should match the scale
        if not abs(scale[i] - torch.sqrt(L[0])) < 1e-2:
            failures+=1
        if not abs(scale2[i] - torch.sqrt(L[1])) < 1e-2:
            failures+=1
        if not abs(scale2[i] - torch.sqrt(L[2])) < 1e-2:
            failures+=1

        
        # Rotation matrix should line up (or be negative) with direction
        norm_dir_col = (direction/torch.sqrt((direction*direction).sum())).unsqueeze(1)
        if not abs(torch.mm(trn(R), norm_dir_col).squeeze(1).abs() - torch.tensor([1.,0,0])).max() < 1e-2:
            failures+=1
    
    # Conservative?
    assert failures < 4


def test_so3_6D()->None:
    '''Test so3_6D'''
    B=10000
    x = torch.randn(B,6)
    r = so3_6D(x)
    assert r.ndim == 3
    assert r.shape == torch.Size((B,3,3))
    # Check orthogonality
    assert (abs((r@trn(r))-eye(3, B)) < 1e-4).all()
    # Check for rotation rather than reflection
    assert (abs(r.det() -1) < 1e-4).all()


def test_so3_6D_2()->None:
    '''Another test, comparing to scale_along_axis_and_expand_matrix'''
    x = torch.randn(6)

    pts = torch.randn(1, 3, 20)


    scale_axis = torch.ones(1)*4
    expand = torch.ones(1)*10
    mat1 = scale_along_axis_and_expand_matrix(x[0:3], scale_axis, expand).squeeze(0) 
    stretch_expand_pts = mat1 @ pts



    R = so3_6D(x.unsqueeze(0)).squeeze(0)

    M = R.permute(1,0) @ torch.tensor([scale_axis, expand, expand]).diag() @ R
    so3_pts = M @ pts

    assert (so3_pts - stretch_expand_pts).abs().max() < 1e-5






def euler(angle: torch.Tensor, axis: str)->torch.Tensor:
    '''Euler axis around an axis'''
    assert angle.ndim == 1 #Batch only

    c = angle.cos()
    s = angle.sin()


    O = torch.zeros_like(c) # noqa: E741
    l = torch.ones_like(c) # noqa: E741
    
    if axis == 'x':
        r = torch.stack([
            torch.stack([l, O, O,]),
            torch.stack([O, c, s,]),
            torch.stack([O,-s, c,]),
        ])
    elif axis == 'y':
        r = torch.stack([
            torch.stack([ c, O, s,]),
            torch.stack([ O, l, O,]),
            torch.stack([-s, O, c,]),
        ])
    elif axis == 'z':
        r = torch.stack([
            torch.stack([ c, s, O,]),
            torch.stack([-s, c, O,]),
            torch.stack([ O, O, l,]),
        ])
    else:
        assert False, "Bad euler axis"

    return r.permute(2,0,1)




if __name__ == "__main__":
    test_so3_6D_2()
    
