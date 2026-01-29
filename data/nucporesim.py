import math as m
import random

import numpy as np
import torch

def _generate_rand_offsets()->tuple[float,float,float]:
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)
    z = np.random.normal(0, 2)
    return x,y,z

#simulate num_nps of nuclear pore images, returned as a list of pytorch tensors
def sim_nups(num_nps: int)->list[torch.Tensor]:
    '''Generate N simulated nuclear pore complexes'''
    asp_rat = 0.85

    # note will correlate roundness with size but as this is not of interest have not corrected
    mj = 59.0
    mn = mj*asp_rat
    num_rings = 2
    ring_sp = 57.0
    num_sites = 8
    num_indimer = 2
    dimer_d = 3
    phi = m.pi/12
    alpha = 0 

    #set dropout and number of observations for each site
    base_num_fluo = num_sites*num_rings*num_indimer

    # start point = each nuclear pore has 18 nup96 proteins spaced evenly around
    # two rings, spaced 57nm apart.
    nps = []
    for i in range(num_nps):
        counter = 0
        fluo_counter = 0


        occupancy = []
        for i in range(base_num_fluo):
            occnum = random.randint(1,100)
            if occnum <=60:
                occupancy.append(0)
            elif occnum <=85:
                occupancy.append(1)
            elif occnum <=95:
                occupancy.append(2)
            else:
                occupancy.append(3)

        totfluo = 0
        for i in range(base_num_fluo):
            totfluo+=occupancy[i]

        col_arrays = np.zeros((totfluo, 3))
        for k in range(num_rings):
            #print(k)
            if k==0:
                z = -ring_sp/2
                delta_angle = 0.0
            else:
                z = ring_sp/2
                delta_angle = m.pi/20
            
            for j in range(0,num_sites):
                angle = 2*m.pi*j/num_sites + delta_angle
                basex = mj*m.cos(angle)*m.cos(alpha) - mn*m.sin(angle)*m.sin(alpha)
                basey = mj*m.cos(angle)*m.sin(alpha) + mn*m.sin(angle)*m.cos(alpha)
                
                for d in range(num_indimer):
                    angle2 = (m.pi/2)-(angle+phi)
                    if d:
                        addx = dimer_d*m.cos(angle2)
                        addy = -dimer_d*m.sin(angle2)
                        addz = -1.2
                    else:
                        addx = -dimer_d*m.cos(angle2)
                        addy = dimer_d*m.sin(angle2)
                        addz = +1.2
                    xrand,yrand,zrand = _generate_rand_offsets()
                    if occupancy[counter]==1:
                        col_arrays[fluo_counter,:] = np.array([basex+addx+xrand,basey+addy+yrand,z+addz+zrand])
                        fluo_counter+=1
                    elif occupancy[counter]==2:
                        col_arrays[fluo_counter,:] = np.array([basex+addx+xrand,basey+addy+yrand,z+addz+zrand])
                        xrand,yrand,zrand = _generate_rand_offsets()
                        col_arrays[fluo_counter+1,:] = np.array([basex+addx+xrand,basey+addy+yrand,z+addz+zrand])
                        fluo_counter+=2
                    elif occupancy[counter]==3:
                        col_arrays[fluo_counter,:] = np.array([basex+addx+xrand,basey+addy+yrand,z+addz+zrand])
                        xrand,yrand,zrand = _generate_rand_offsets()
                        col_arrays[fluo_counter+1,:] = np.array([basex+addx+xrand,basey+addy+yrand,z+addz+zrand])
                        xrand,yrand,zrand = _generate_rand_offsets()
                        col_arrays[fluo_counter+2,:] = np.array([basex+addx+xrand,basey+addy+yrand,z+addz+zrand])
                        fluo_counter+=3
                    counter+=1
        
        nps.append(torch.from_numpy(col_arrays))
    return nps

