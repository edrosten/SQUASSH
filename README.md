# SQUASSH

## Prerequisites

This code has only been tested on Linux (Mint and Ubuntu). It will likely work
on other systems.

The code currently uses `torch.compile`, so you will need a version of python
compatible with the version of torch you are using. The latest python 3.11 is
well tested but other versions should work too. All the code is know to run on 
torch 2.2.1. Torch compile can be quite buggy and not all of the code has been
fully tested on more recent torch versions.

If you want to train you will need a GPU. The examples were all tested on a
2080Ti (11GB RAM), so may not run on a GPU with less RAM without modification.
The code will execute on a CPU, but will be too slow to be useful in most cases.

You will probably want a program for viewing 3D models in PLY format.
[Meshlab](https://www.meshlab.net/) is a very good choice.

## Getting SQUASSH

You will need [git LFS](https://github.com/git-lfs/git-lfs/tree/main) if you
want to get the sample data and be able to run the examples. On apt
based Linux distributions you can install it with `sudo apt install git-lfs`

Then get SQUASSH with:
```
git clone https://github.com/edrosten/SQUASSH
```

## Setting up squash

SQUASSH depends on a number of packages. You can install them with:

```
pip install -r requirements.txt
```

## Running SQUASSH for the first time

To get started, running SQUASSH on RESI data of nuclear pore complexes, run:
```
python train_nupc.py
```
Output from the execution will be in the `log/` directory. The file name will be
the timestamp that the run started followed by the current version of the
respoitory.

Note this may take some time, but you can skip to the next step straight away.

## Analyzing the results

Since SQUASSH can take hours to run, the results of some previous runs have 
been provided. For example a run of the RESI data is provided in `sample_logs/1711985336-4d7cc96effb6e4740278bd39261837986110b4a2/`.

To view the raw point cloud (brightnesses not shown), along with the learned
axis, open
`sample_logs/1711985336-4d7cc96effb6e4740278bd39261837986110b4a2/run-000-phase_1/final.ply`
in meshlab. 

A more useful thing is a mesh of the isosurface of the model. You can get this
by running:
```
./render_marching_cubes.py -r 2 -t .2 sample_logs/1711985336-4d7cc96effb6e4740278bd39261837986110b4a2/run-000-phase_1/final_model.txt 
```
This will create an output file `hax/mesh-0.2000.ply`, which you can open in
meshlab. If you open both files, you can see the mesh and axis. 

Further analysis will depend on the specifics of the data and what information
you want to extract. A complete example is given in `figure_2_plot_nupc.py`. If
you run this it will output the following files:
```
hax/figure2_bates.svg
hax/figure2_bates_3d.ply
hax/figure2_historgram.svg
hax/figure2_resi.svg
hax/figure2_resi_3d.ply
hax/figure2_z_correlation.svg
```
which form the panels in figure 2 of the paper.

## Continuing on

The training schemes for the datasets used in the paper are provided in the
following files, all of which can be readily run:
```
train_bunny.py
train_dan_microtubules.py
train_legant.py
train_nupc.py
train_spectrin.py
train_trichomes.py
```

There is no configuration system. If you wish to run SQUASSH on the
4Pi-STORM data, you will need to edit `train_nupc.py`, and uncomment line 15.


Note that if the repository is not clean (i.e. uncommitted changes or untracked
files), training will not execute. This ensures that every run is traceable to a
precise and complete version of the source code.
