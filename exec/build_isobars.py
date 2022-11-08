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
from scipy.integrate import quad
from joblib import Parallel, delayed

from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


N_SEEDS = 3 + 3 # 3 for r, theta, phi + 3 for delta_{x,y,z} Gaussian
POS_SEEDS = {'radius':0,'costheta':1,'phi':2,'gauss_x':3,'gauss_y':4,'gauss_z':5 }
GAUSS_SEEDS = ['gauss_x','gauss_y','gauss_z'] # Gaussian seeds
UNIFORM_SEEDS = ['radius','costheta','phi']
#%%



# Let's try solving an ODE numerically

# rho'/rho for spherical Woods-Saxon
def rhodotoverrho(r,R,a):
    temp = -1.
    temp /= a
    temp /= 1 + math.exp((R-r)/a)
    return temp

# dimension-reduced differential equation
def deriv_z(z,r,Rws,a,l,nl):
    f, fdot = z
    temp = -fdot*(2/r + rhodotoverrho(r,Rws,a)*nl)
    temp += l*(l+1)/r**2*f
    temp += rhodotoverrho(r,Rws,a)*nl
    return [fdot, temp]


def der_z(r,z,Rint,aint,lint,nlint):
    f, fdot, fhomo, fhomodot = z
    temp = -fdot*(2/r + rhodotoverrho(r,Rint,aint)*nlint)
    temp += lint*(lint+1)/r**2*f
    temp += rhodotoverrho(r,Rint,aint)*nlint
    temphomo = -fhomodot*(2/r + rhodotoverrho(r,Rint,aint)*nlint)
    temphomo += lint*(lint+1)/r**2*fhomo
    return [fdot, temp, fhomodot, temphomo]



def sph_harmonic_22(costheta,phi):
    cos2phi = math.cos(2*phi)
#    cos2phi = 1.
    return 1./4.*np.sqrt(15./np.pi)*math.cos(2*phi)*(1.-costheta**2)

def sph_harmonic_20(costheta,phi):
    return 1./4.*np.sqrt(5./np.pi)*(3.*costheta**2 -1)

def sph_harmonic_3(costheta,phi):
    return np.sqrt(7./(4.*np.pi))*(5.*costheta**3 -3.*costheta)/2.
    

# def deform(r,costheta,phi,Rws,Rstep, w,b20,b22,b3,db20,db22,db3):
# def deform(r,costheta,phi,Rws,Rstep, w,beta20,beta22,beta3, f2, fp2,f3,fp3):
def deform(r,costheta,phi,Rws,beta20,beta22,beta3, f2, fp2,f3,fp3):
#     w = w_gauss
    rmin = 1.0e-1
#     beta20 = b2*math.cos(gamma)
#     beta22 = b2*math.sin(gamma)

    sintheta = math.sqrt(1-costheta**2)
    theta = np.arccos(costheta)

    dtheta=0
    dphi = 0
    dr = 0
    if (r > rmin):
        dtheta += Rws/r*beta20*f2(r)*(-3./2.)*math.sqrt(5./np.pi)*costheta*sintheta/r
        dtheta += Rws/r*beta22*f2(r)*1/2.*math.sqrt(15./np.pi)*sintheta*costheta*np.cos(2*phi)/r
        dtheta += Rws/r*beta3*f3(r)*1./4.*math.sqrt(7./np.pi)*(3*sintheta-15*sintheta*costheta**2)/r
    
        dphi += Rws*beta22*f2(r)*(-1./2.)*math.sqrt(15./np.pi)*sintheta**2*math.sin(2*phi)/r/sintheta/r/sintheta
           
        dr += Rws*beta20*fp2(r)*sph_harmonic_20(costheta,phi)
        dr += Rws*beta22*fp2(r)*sph_harmonic_22(costheta,phi)
        dr += Rws*beta3*fp3(r)*sph_harmonic_3(costheta,phi)
    
    
    if np.abs(dr) >= np.abs(r):
        print("dr, r = " + str(dr) + ", " + str(r))
    if (theta+(dtheta) < 0 or theta+(dtheta) > np.pi):
        print("dtheta, theta = " + str(dtheta) + ", " + str(theta))
#     if (phi+dphi < 0 or phi+dphi > 2*np.pi):
#         print("dphi, phi = " + str(dphi) + ", " + str(phi))
              
    
    r += dr
    theta += dtheta
    costheta = np.cos(theta)
    phi += dphi
    
    return r, costheta, phi

def cartesian(r,costheta,phi):
    sintheta = np.sqrt(1.-costheta**2)
    z = r*costheta
    x = r*sintheta*np.cos(phi)
    y = r*sintheta*np.sin(phi)
    
    return x, y, z
#%% 
def place_nucleon(R_step, w_gauss, beta2, gamma, beta3, Rws, seed, f2, fp2, f3, fp3):
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
    
    sintheta = np.sqrt(1.-costheta**2)
    oldz = r*costheta
    oldx = r*sintheta*np.cos(phi)
    oldy = r*sintheta*np.sin(phi)
    
    # folding with 3d gaussian
    gauss_seed_x = seed[POS_SEEDS['gauss_x']]
    gauss_seed_y = seed[POS_SEEDS['gauss_y']]
    gauss_seed_z = seed[POS_SEEDS['gauss_z']]
    
    newz = oldz + w_gauss*gauss_seed_z
    newx = oldx + w_gauss*gauss_seed_x
    newy = oldy + w_gauss*gauss_seed_y
    
    r = math.sqrt(newz**2+newx**2+newy**2)
    phi = math.atan2(newy,newx)
    costheta = newz/r
    
    
#     # deformation
    beta20 = beta2*math.cos(gamma)
    beta22 = beta2*math.sin(gamma)
    if (beta20 != 0 or beta22 != 0 or beta3 != 0):
        nsteps = 15
        for step in range(nsteps):
            r,costheta,phi = deform(r,costheta,phi,Rws,beta20/nsteps,beta22/nsteps,beta3/nsteps, f2, fp2,f3,fp3)
    
    # break into small steps
#     beta20 = beta2*math.cos(gamma)
#     beta22 = beta2*math.sin(gamma)
#     steps = 100
#     db20 = beta20/steps
#     db22 = beta22/steps
#     db3 = beta3/steps
#     for step in range(steps):
#         b20 = db20*int(step)
#         b22 = db22*int(step)
#         b3 = db3*int(step)
#         r = deform(r,costheta,phi,R_step,R_step, w_gauss,b20,b22,b3,db20,db22,db3)

    
    # cartesian coordinates
    x, y, z = cartesian(r,costheta,phi)
    
    return np.array([x,y,z])
    
def build_nucleus(seeds_nucleus, n_nucleons, R_step, w_gauss, beta2, gamma, beta3, f2, fp2, f3, fp3):

#     nucleus = Parallel(n_jobs=60)(delayed(place_nucleon)(R_step, w_gauss, beta2, gamma,  beta3, seeds_nucleus[n], f2, fp2, f3, fp3) for n in range(n_nucleons))
#     nucleus = np.array(nucleus)
    
    nucleus = np.zeros((n_nucleons,3))
    for n in range(n_nucleons):
        nucleus[n,:] = place_nucleon(R_step, w_gauss, beta2, gamma,  beta3, seeds_nucleus[n], f2, fp2, f3, fp3)
        
    return nucleus

# def run_isobar(seeds_nucleus, n_nucleons, R_step, w_gauss, beta2, gamma, beta3):
    

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
    Rws = 0
    aws = 0
    # print(confs.keys())

    while ('isobar'+str(n_isobars+1) in confs['isobar_properties'].keys()):
            isobar_conf = confs['isobar_properties']['isobar'+str(n_isobars+1)]
            R_step = isobar_conf['step_radius']['value']
            diffusiveness = isobar_conf['diffusiveness']['value']
            beta2 = isobar_conf['beta_2']['value']
            gamma = isobar_conf['gamma']['value']
            beta3 = isobar_conf['beta_3']['value']
            Rws = isobar_conf['WS_radius']['value']
            aws = isobar_conf['WS_diffusiveness']['value']
            isobars += [ [R_step,diffusiveness,beta2,gamma,beta3] ]
            isobar_names += [ isobar_conf['isobar_name'] ]
            n_isobars +=1
        
    # print(isobar_names)

    isobars = np.array(isobars)
    for i in range(n_isobars):
        R_step = isobars[i,0]
        a = isobars[i,1]
        if Rws == 0 or aws == 0:
            raise Exception('Need to specify WS_radius and WS_diffusiveness in input file')
        print("Solving differential equation")
        
#         Rint = 5.09
#         aint = 0.46
        lint = 3
#         nlint = .001
        # rsp = np.linspace(1.0e-4,1000,1000000)
        rmin = 1.0e-1
        rmax = 2.2*Rws
        # rsp2 = range(1.0e-0,8,1000)
        zinit = [rmax,1,1,0]
        # solver = ode(deriv_z)
        # z = odeint(deriv_z, zinit, rsp,args=(Rint,aint,lint,nlint))
#         res2 = solve_ivp(der_z, y0=zinit, args=(Rint,aint,2,1),t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)
#         res3 = solve_ivp(der_z, y0=zinit, args=(Rint,aint,3,1),t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)
        args=(Rws,aws,2,1)
        res2 = solve_ivp(fun=lambda t,y: der_z(t,y,*args), y0=zinit,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)
        args=(Rws,aws,3,1)
        res3 = solve_ivp(fun=lambda t,y: der_z(t,y,*args), y0=zinit,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)

        # f, fdot = z.T
        f2 = interp1d(res2.t,res2.y[0] - res2.y[1,-1]/res2.y[3,-1]*res2.y[2])
        fp2 = interp1d(res2.t,res2.y[1] - res2.y[1,-1]/res2.y[3,-1]*res2.y[3])
        f3 = interp1d(res3.t,res3.y[0] - res3.y[1,-1]/res3.y[3,-1]*res3.y[2])
        fp3 = interp1d(res3.t,res3.y[1] - res3.y[1,-1]/res3.y[3,-1]*res3.y[3])
        
        
        
        
#         rsp = np.linspace(1.0e-9,1.8*R_step,10000)
#         zinit = [0,0]
#         z2 = odeint(deriv_z, zinit, rsp,args=(R_step,a,2,.01))
#         z3 = odeint(deriv_z, zinit, rsp,args=(R_step,a,3,.01))
#         f2 = interp1d(rsp,z2[:,0])
#         fp2 = interp1d(rsp,z2[:,1])
#         f3 = interp1d(rsp,z3[:,0])
#         fp3 = interp1d(rsp,z3[:,1])
#         data = np.zeros((n_configs,n_nucleons,3),dtype=np.float)
        data = Parallel(n_jobs=60)(delayed(build_nucleus)(seeds[s],n_nucleons,*isobars[i],f2,fp2,f3,fp3) for s in range(n_configs))
#         for s in range(n_configs):
#             data[s,:,:] = build_nucleus(seeds[s],n_nucleons,*isobars[i],f2,fp2,f3,fp3)
    
        with h5py.File(out_dir+'/'+isobar_names[i]+'.hdf', 'w') as f:
            data_set = f.create_dataset(isobar_names[i],(n_configs,n_nucleons,3))
            data_set[:] = data 
            
#%%
if __name__ == "__main__":
        main()   
