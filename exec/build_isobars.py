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
import os
import yaml
import math
# import numpy.random as random
from yaml.loader import SafeLoader


N_SEEDS = 3 + 3 # 3 for r, theta, phi + 3 for delta_{x,y,z} Gaussian
POS_SEEDS = {'radius':0,'costheta':1,'phi':2,'gauss_x':3,'gauss_y':4,'gauss_z':5 }
GAUSS_SEEDS = ['gauss_x','gauss_y','gauss_z'] # Gaussian seeds
UNIFORM_SEEDS = ['radius','costheta','phi']
#%%

def sph_harmonic_22(costheta,phi):
    cos2phi = math.cos(2*phi)
#    cos2phi = 1.
    return 1./4.*np.sqrt(15./np.pi)*math.cos(2*phi)*(1.-costheta**2)

def int_sph_harmonic_22(costheta,phi):
    cos2phi = math.cos(2*phi)
#    cos2phi = 1.
    return 1./4.*np.sqrt(15./np.pi)*cos2phi*(costheta -costheta**3/3.)

def sph_harmonic_20(costheta,phi):
    return 1./4.*np.sqrt(5./np.pi)*(3.*costheta**2 -1)

def int_sph_harmonic_20(costheta,phi):
    return 1./4.*np.sqrt(5./np.pi)*(costheta**3 -costheta)

def sph_harmonic_3(costheta,phi):
    return np.sqrt(7./(4.*np.pi))*(5.*costheta**3 -3.*costheta)/2.
    
def int_sph_harmonic_3(costheta,phi):
    return (np.sqrt(7./np.pi)/4.)*(5.*costheta**4/4. - 3.*costheta**2/2.)

def deform(r,costheta,phi,R_step, w_gauss,beta2,gamma,beta3):
    w = w_gauss
#    rt2 = math.sqrt(2)
    rt6 = math.sqrt(6)
#    rho = math.sqrt(np.pi/2)*w_gauss*(math.erf(r/w/rt2) - math.erf((R_step-r)/w/rt2))
#    gaussr = math.exp(-r**2/2/w**2)
#    gaussRstep = math.exp(-R_step**2/2/w**2)
#    exprR = math.exp(r*R_step/w**2)
#    rtpi2 = math.sqrt(np.pi/2)
#    int2xrho = rtpi2*w_gauss*(
#            gaussr*gaussRstep*rtpi2*w*(
#                gaussr**(-1)*R_step - exprR*(r+R_step)
#                )
#            + (R_step**2 + w**2 - r**2) + math.erf((r-R_step)/w/rt2) + (R_step**2 + w**2 + r**2)*math.erf(R_step/w/rt2)
#            )
    int2xrhooverr2rho = 2*(-12*math.exp(-(r-R_step)**2/6/w**2)*w**2
            +2*math.exp(-R_step**2/6/w**2) * (-r*R_step+6*w) + math.sqrt(6*np.pi)*w*(
                (R_step - r)*math.erf((r-R_step)/w/rt6) + (R_step + r)*math.erf(R_step/w/rt6)))/r**2/(
                    2*math.exp(-(r-R_step)**2/6/w**2)*(r-R_step) - 2*math.exp(-R_step**2/6/w**2)*R_step
                        + math.sqrt(6*np.pi)*w*(-math.erf((r-R_step)/w/rt6) + math.erf(R_step/w/rt6)))
    beta20 = beta2*math.cos(gamma)
    beta22 = beta2*math.sin(gamma)
#    r = r*(1. + beta20*sph_harmonic_20(costheta, phi) + beta22*sph_harmonic_22(costheta, phi) + beta3*sph_harmonic_3(costheta, phi))
#    r += beta20*sph_harmonic_20(costheta, phi) + beta22*sph_harmonic_22(costheta, phi) + beta3*sph_harmonic_3(costheta, phi)*(1 - int2xrho/r**2/rho)
    r += beta20*sph_harmonic_20(costheta, phi) + beta22*sph_harmonic_22(costheta, phi) + beta3*sph_harmonic_3(costheta, phi)*(1 - int2xrhooverr2rho)
    costheta = costheta #- 3.*(beta20*int_sph_harmonic_20(costheta, phi) + beta22*int_sph_harmonic_22(costheta, phi) + beta3*int_sph_harmonic_3(costheta, phi))
    phi = phi
#    if costheta > 1.:
#        print("Cos(theta) > 1!")
#    if costheta < -1.:
#        print("Cos(theta) < -1!")
    
    return r, costheta, phi

def cartesian(r,costheta,phi):
    sintheta = np.sqrt(1.-costheta**2)
    z = r*costheta
    x = r*sintheta*np.cos(phi)
    y = r*sintheta*np.sin(phi)
    
    return x, y, z
#%% 
def place_nucleon(R_step, w_gauss, beta2, gamma, beta3, seed):
    # radial coordinate
    radius_seed = seed[POS_SEEDS['radius']]
    # p(r) ~ r²dr, cdf(r) ~ r³, r ~ seed_r^(1/3), cdf sampled uniform [0,1]
    # plus max radius is R_step
    r = radius_seed**(1./3.)*R_step
    # azimuthal angle
    costheta_seed = seed[POS_SEEDS['costheta']]
    # uniform in [-1,1]  
    costheta = 2.*costheta_seed - 1.
    # axial angle
    phi_seed = seed[POS_SEEDS['phi']]
    # uniform in [0,2 pi]
    phi = 2.*np.pi*phi_seed
    
    # folding with 3d gaussian
    gauss_seed_x = seed[POS_SEEDS['gauss_x']]
    gauss_seed_y = seed[POS_SEEDS['gauss_y']]
    gauss_seed_z = seed[POS_SEEDS['gauss_z']]
    # combine three gaussians in quadrature to get '3d' number
    diffusiveness_seed = np.sqrt(gauss_seed_x*gauss_seed_x  + gauss_seed_y*gauss_seed_y + gauss_seed_z*gauss_seed_z)
    
    # diffuse theta function parametrization
    r = r + diffusiveness_seed*w_gauss;
    
    # deformation
    r, costheta, phi = deform(r,costheta,phi,R_step,w_gauss,beta2, gamma,beta3)
    
    # cartesian coordinates
    x, y, z = cartesian(r,costheta,phi)
    
    return np.array([x,y,z])
    
def build_nucleus(seeds_nucleus, n_nucleons, R_step, w_gauss, beta2, gamma, beta3):

    nucleus = np.zeros((n_nucleons,3))
    for n in range(n_nucleons):
        nucleus[n,:] = place_nucleon(R_step, w_gauss, beta2, gamma,  beta3, seeds_nucleus[n])
        
    return nucleus
    
#%%
def main():
    conffile = sys.argv[1]
    with open(conffile, 'r') as stream:
        confs = yaml.load(stream,Loader=SafeLoader)

    conf_samples = confs['isobar_samples']
    n_configs = conf_samples['number_configs']['value']
    n_nucleons = conf_samples['number_nucleons']['value']
    seeds_file = conf_samples['seeds_file']['filename']
    out_dir = conf_samples['output_path']['dirname']
    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    
    with h5py.File(seeds_file, 'r') as f:
        seeds = f['isobar_seeds'][:,:,:]

    if seeds.shape[0] < n_configs:
        n_configs = seeds.shape[0]

    n_isobars = 0
    isobars = []
    isobar_names = []
    # print(confs.keys())

    while ('isobar'+str(n_isobars+1) in confs['isobar_properties'].keys()):
            isobar_conf = confs['isobar_properties']['isobar'+str(n_isobars+1)]
            Rstep = isobar_conf['step_radius']['value']
            diffusiveness = isobar_conf['diffusiveness']['value']
            beta2 = isobar_conf['beta_2']['value']
            gamma = isobar_conf['gamma']['value']
            beta3 = isobar_conf['beta_3']['value']
            isobars += [ [Rstep,diffusiveness,beta2,gamma,beta3] ]
            isobar_names += [ isobar_conf['isobar_name'] ]
            n_isobars +=1
        
    # print(isobar_names)

    isobars = np.array(isobars)
    for i in range(n_isobars):
        data = np.zeros((n_configs,n_nucleons,3),dtype=np.float)
        for s in range(n_configs):
            data[s,:,:] = build_nucleus(seeds[s],n_nucleons,*isobars[i])
    
        with h5py.File(out_dir+'/'+isobar_names[i]+'.hdf', 'w') as f:
            data_set = f.create_dataset(isobar_names[i],(n_configs,n_nucleons,3))
            data_set[:] = data 
            
#%%
if __name__ == "__main__":
        main()   
