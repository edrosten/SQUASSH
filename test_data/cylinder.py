import torch


verts=[]

N=30
M=10
radius = 300 / 2
zd = 170

angles = torch.arange(0, N) / N * 2 * torch.pi 

x = torch.cos(angles) * radius
y = torch.sin(angles) * radius


z = torch.arange(0, M)/M * zd

zs = z.unsqueeze(1).expand(M, N).reshape(-1)
xs = (x.unsqueeze(0).expand(M, N)).reshape(-1)
ys = (y.unsqueeze(0).expand(M, N)).reshape(-1)


verts = torch.stack([xs, ys, zs], 1)


header=f"""ply
format ascii 1.0
element vertex {len(verts)}
property float x
property float y
property float z
end_header
"""


print(header)

for row in verts:
    print(" ".join([str(i.item()) for i in row]))


