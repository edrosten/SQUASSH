#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import argparse
import torch
import save_ply

os.environ["OVERRIDE_UNCLEAN_REPO"]="1" 
from git import dirname #noqa pylint:disable=wrong-import-position


def _parse_comma_separated_values(value: str)->list[float]:
    return [float(v) for v in value.split(',')]


parser= argparse.ArgumentParser(prog=__file__[__file__.rfind('/')+1:])
parser.add_argument('filename')
parser.add_argument('-r', '--radius', type=float, default=2.0)
parser.add_argument('-q', '--radius2', type=float, default=2.0)
parser.add_argument('-t', '--threshold', type=_parse_comma_separated_values, default=[.1, .2, .3, .4, .5])
parser.add_argument('-g', '--gpu', action='store_true')
parser.add_argument('-s', '--split', type=int)
parser.add_argument('-c', '--colour', action='store_true')
parser.add_argument('-f', '--filter', type=float, default=1.0, help='Filter out fistant points (specify fraction to keep)')
parser.add_argument('-o', '--output-pattern', type=str, default='hax/mesh-{:.4f}.ply', help='Format string for the output files')

args = parser.parse_args()

if bool(args.colour) != (args.split is not None):
    print(args)
    raise argparse.ArgumentError(None, '-c/--colour and -s/--split must both be specified for colour renderings')

if args.filter !=1 and bool(args.colour):
    raise argparse.ArgumentError(None, '-c/--colour and nontrivial -f/--filter cannot be used together')

with open(args.filename, encoding='utf8') as f:
    txtlines = f.readlines()

color=args.colour


txt = [line.split() for line in txtlines]

datalines = [ [float(d) for d in line ] for line in txt ]

data = torch.tensor(datalines)

points = data[:,0:3]


centre = points.mean(0)
distance = (points - centre.unsqueeze(0).expand_as(points)).pow(2).sum(1)


_, ordered_indices = distance.sort()
print(points.shape[0] * args.filter)
last = int(points.shape[0] * args.filter+.5)


if data.shape[1] == 4:
    weights = data[:,3]
else:
    print("oooh")
    weights = torch.ones(data.shape[0])


points = points[ordered_indices[0:last], :]
weights = weights[ordered_indices[0:last]]


#save_ply.save_pointcloud_as_mesh("/dev/stdout", points.cuda().half(), weights.cuda().half(), 20)
class _Slice:
    def __getitem__(self, s:slice)->slice:
        return s

S = _Slice()


radius = args.radius
colour =  ( 127, 127, 127, 255)

if args.gpu:
    e = torch.tensor([1.0]).half().cuda()
else:
    e = torch.tensor([1.0])


pts = points.to(e)
wts = weights.to(e)

for threshold in args.threshold:
    print("threshold", threshold)
    comments = f"git={dirname} args={sys.argv} radius={radius} threshold={threshold}"
    save_ply.save_pointcloud_as_mesh(args.output_pattern.format(threshold), pts, wts, radius, threshold, 100, colour, comments)
