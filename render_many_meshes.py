import glob 
import os 
import tqdm 
import torch
import save_ply


os.mkdir('hax/anim')

old = torch.tensor([])
direc="1711985336-4d7cc96effb6e4740278bd39261837986110b4a2"

for num,filename in enumerate(tqdm.tqdm(sorted(glob.glob(f'log/{direc}/*/*/current_model.txt')))):
    with open(filename, encoding='utf8') as f:
        txtlines = f.readlines()

    txt = [l.split() for l in txtlines]

    datalines = [ [float(d) for d in l ] for l in txt ]

    data = torch.tensor(datalines)

    if old.shape != data.shape:
        old = data

    a=.92

    old = old*a + data*(1-a)

    points = old[:,0:3]

    if points.shape[1] == 4:
        weights = old[:,3]
    else:
        weights = torch.ones(old.shape[0])

    if len(datalines) > 100:
        radius=2
    else:
        radius=2
    if num%5 == 0:
        save_ply.save_pointcloud_as_mesh(f'hax/anim/{num:05}.ply', points.cuda().half(), weights.cuda().half(), radius, 0.1)
