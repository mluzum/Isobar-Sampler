#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:33:43 2022

@author: mauricio

Usage: ./make_isobar_seeds.py conf.yaml

Creates hdf5 file with shape (number_configs, number_nucleons, 2) 
with random numbers to generate isobar configurations
"""

import numpy as np
import h5py
import sys
import yaml
import numpy.random as random
from yaml.loader import SafeLoader

from joblib import Parallel, delayed

N_SEEDS = 3 + 3 # 3 for r, theta, phi + 3 for delta_{x,y,z} Gaussian
POS_SEEDS = {'radius':0,'costheta':1,'phi':2,'gauss_x':3,'gauss_y':4,'gauss_z':5 }
GAUSS_SEEDS = ['gauss_x','gauss_y','gauss_z'] # Gaussian seeds
UNIFORM_SEEDS = ['radius','costheta','phi']
#%% 

def seed_nucleus(n_nucleons):
    nucleus = np.empty((n_nucleons, N_SEEDS))
    for seed in UNIFORM_SEEDS:
        nucleus[:, POS_SEEDS[seed]] = random.uniform(low=0.0, high=1.0, size=n_nucleons)
    for seed in GAUSS_SEEDS:
        nucleus[:, POS_SEEDS[seed]] = random.normal(loc=0.0, scale=1.0, size=n_nucleons)
    return nucleus
    
#%%
def main():
    conffile = sys.argv[1]
    with open(conffile, 'r') as stream:
        confs = yaml.load(stream,Loader=SafeLoader)

    conf_seed = confs['isobar_seeds']
    n_configs = conf_seed['number_configs']['value']
    n_nucleons = conf_seed['number_nucleons']['value']
    out_file = conf_seed['output_file']['filename']
    njobs = 1  # default to serial calculation
    if 'number_of_parallel_processes' in conf_seed:
        njobs = conf_seed['number_of_parallel_processes']['value']
       
    data = Parallel(n_jobs=njobs)(delayed(seed_nucleus)(n_nucleons) for i in range(n_configs))
#     data = np.zeros((n_configs,n_nucleons,N_SEEDS),dtype=np.float)
#     for i in range(n_configs):
#         data[i,:,:] = seed_nucleus(n_nucleons)
    # print(data[0])
    with h5py.File(out_file, 'w') as f:
        data_set = f.create_dataset('isobar_seeds',(n_configs,n_nucleons,N_SEEDS), compression="gzip", compression_opts=9)
        data_set[:] = data 

#%%
if __name__ == "__main__":
        main()   