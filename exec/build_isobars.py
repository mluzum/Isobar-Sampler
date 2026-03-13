#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:33:43 2022

@author: mauricio+matt

Usage: ./build_isobars.py conf.yaml

Creates hdf5 file for each isobar configuration defined in conf.yaml, 
Each data set has shape (number_nuclei, number_nucleons, 3) 
with 3D nucleon positions in fm.  Format compatible with default Trento.
"""

# from numba import jit


import numpy as np
import h5py
import sys
import os
import yaml
import math


# import pickle
from scipy.optimize import fsolve
from scipy.spatial.distance import pdist, squareform

# from scipy import interpolate

# import numpy.random as random
from yaml.loader import SafeLoader
from scipy.integrate import quad
from joblib import Parallel, delayed

# from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from mpmath import polylog

# from scipy.optimize import brentq




# import copy


# import cProfile


# N_SEEDS = 3 + 3 # 3 for r, theta, phi + 3 for delta_{x,y,z} Gaussian
POS_SEEDS = {'radius':0,'costheta':1,'phi':2,'gauss_x':3,'gauss_y':4,'gauss_z':5 }
EPS = 1.0e-12
# GAUSS_SEEDS = ['gauss_x','gauss_y','gauss_z'] # Gaussian seeds
# UNIFORM_SEEDS = ['radius','costheta','phi']
#%%







# @jit(nopython=True)
# Real spherical harmonics
def Y_22(costheta,phi):
    return 1./4.*np.sqrt(15./np.pi)*math.cos(2*phi)*(1.-costheta**2)

# @jit(nopython=True)
def Y_20(costheta,phi):
    return 1./4.*np.sqrt(5./np.pi)*(3.*costheta**2 -1)

# @jit(nopython=True)
def Y_30(costheta,phi):
    return np.sqrt(7./(4.*np.pi))*(5.*costheta**3 -3.*costheta)/2.

def Y_31(costheta,phi):
    sintheta = np.sqrt(1 - costheta**2)
    return (-1./8.)*np.sqrt(21./np.pi)*sintheta*(5.*costheta**2 -1)*math.cos(phi)

def Y_32(costheta,phi):
    sintheta = np.sqrt(1 - costheta**2)
    return (1./4.)*np.sqrt(105./(2*np.pi))*sintheta**2*costheta*math.cos(2*phi)

def Y_33(costheta,phi):
    sintheta = np.sqrt(1 - costheta**2)
    return (-1./8.)*np.sqrt(35./np.pi)*sintheta**3*math.cos(3*phi)

def Y_40(costheta,phi):
    return (3./16.)*(1./(math.sqrt(np.pi)))*(35.*costheta**4 - 30.*costheta**2 + 3)

# @jit(nopython=True)
# Derivative with respect to theta, phi
def dY20_dtheta(costheta,sintheta,phi):
    return (-3./2.)*math.sqrt(5./np.pi)*costheta*sintheta

# @jit(nopython=True)
def dY22_dtheta(costheta,sintheta,phi):
    return 1/2.*math.sqrt(15./np.pi)*sintheta*costheta*np.cos(2*phi)

# @jit(nopython=True)
def dY30_dtheta(costheta,sintheta,phi):
    return 1./4.*math.sqrt(7./np.pi)*(3*sintheta-15*sintheta*costheta**2)

# @jit(nopython=True)
def dY40_dtheta(costheta,sintheta,phi):
    return (15./4.)*(1./(math.sqrt(np.pi)))*costheta*(3 - 7.*costheta**2)*sintheta

# @jit(nopython=True)
def dY22_dphi(costheta,sintheta,phi):
    return (-1./2.)*math.sqrt(15./np.pi)*sintheta**2*math.sin(2*phi)

# @jit(nopython=True)
# Spherical Woods-Saxon
def Woods_Saxon_unnormalized(r,R,a):
    return  1/(np.exp((r-R)/a)+1)


# Properly normalized probability density
def Woods_Saxon(r,R,a):
#     norm = quad(lambda r: r**2*WS(r,R,a), 0,np.inf)[0]*4*np.pi
# Use mppath implementation of polylog instead of numerical integral to normalize
    norm = -8*a**3*np.pi*polylog(3,-math.exp(R/a))
    return float(Woods_Saxon_unnormalized(r,R,a)/norm)

# @jit(nopython=True)
# rho'/rho for spherical Woods-Saxon
def rhodot_over_rho(r,R,a):
    temp = -1.
    temp /= a
    temp /= 1 + np.exp((R-r)/a)
    return temp

# @jit(nopython=True)
# dimension-reduced differential equation -- solve both inhomogenious (f, fdot) and homogenious (fhomo, fhomodot)
# returns derivative of (f, fdot, fhomo, fhomodot) = (f'_I, f''_I, f'_H, f''_H)
def diff_eq(r,z,R,a,l_int):
    f, fdot, fhomo, fhomodot = z
    fdot_dot = -fdot*(2/r + rhodot_over_rho(r,R,a))
    fdot_dot += l_int*(l_int+1)/r**2*f
    fdot_dot += rhodot_over_rho(r,R,a)
    fhomodot_dot = -fhomodot*(2/r + rhodot_over_rho(r,R,a))
    fhomodot_dot += l_int*(l_int+1)/r**2*fhomo
    return [fdot, fdot_dot, fhomodot, fhomodot_dot]

# @jit(nopython=True)
# Analytic form for (properly normalized) step+Gauss distribution
def Step_Gauss(r,R_step,w):
    rt2 = np.sqrt(2)
    rtpi = np.sqrt(np.pi)
    rho1 = 0.
    rho = 0.
    if r == 0:
        rho1 = math.erf(R_step/w/rt2)
        rho1 *= 2*rtpi
        rho2 = -np.exp(-R_step**2/2/w**2)*2*rt2*R_step/w
    else:
        erfplus = math.erf((r + R_step)/w/rt2)
        erfminus = math.erf((r - R_step)/w/rt2)
        rho1 = rtpi*(erfplus - erfminus)
        expplus = np.exp(-((r+R_step)**2)/2/(w**2))
        expminus = np.exp(-((r-R_step)**2)/2/(w**2))
        rho2 = expplus - expminus
        rho2 *= rt2*w/r
    rho = rho1 + rho2
    rho *= 3/8/R_step**3/np.pi**(3/2)
    return rho


# KL divergence between Woods-Saxon and step+Gauss.  inp = (R_step,w).  We will choose these parameters to minimize the KL divergence for a given Woods-Saxon parameter a/R (i.e., an example corresponding to R=1, which can be scaled to obtain any R)
def KL_WS_step(inp, a_over_R):
    R_step = inp[0]
    w = inp[1]
    KL = quad(lambda r: r**2*Woods_Saxon(r,1,a_over_R)*np.log(Woods_Saxon(r,1,a_over_R)/Step_Gauss(r,R_step,w)),0,1+5*a_over_R)[0]
    KL *= 4*np.pi
    return KL


# Returns best value of parameters (R_step, w) corresponding to Woods-Saxon parameters (R,a)
def Rst_w_from_WS(R,a):
    a_over_R = a/R
    x0 = (1,a_over_R)
    res = minimize(KL_WS_step, x0, args=a_over_R)
    return R*res.x



# @jit(nopython=True)
# Spherical to cartesian coordinates
def cartesian(r,costheta,phi):
    sintheta = np.sqrt(1.-costheta**2)
    z = r*costheta
    x = r*sintheta*np.cos(phi)
    y = r*sintheta*np.sin(phi)
    
    return x, y, z
#%% 


# def corr_shift_realistic(r, c_volume, c_extremum, integrated_correlation):
#     func = lambda rtilde: integrated_correlation(rtilde) (rtilde**3)/3 - (r**3)/3
#     return fsolve(func, r) - r
def corr_shift_realistic(r, integrated_correlation, strength_scale, length_scale, avgprob):
    func = lambda rtilde: length_scale**3*strength_scale*integrated_correlation(rtilde/length_scale)*3/4/np.pi + (1 + 0.16*length_scale**3*strength_scale*avgprob)*(rtilde**3) - (r**3)
#     func = lambda rtilde: strength_scale*integrated_correlation(rtilde/length_scale) + (1 + 0.16*length_scale**3*strength_scale*avgprob)*(rtilde**3) - (r**3)
    return fsolve(func, r + 0.5) - r
    # return brentq(func, 0, r+10) - r
    

# @jit(nopython=True)
# Compute magnitude of the coordinate shift of a pair of nucleons 
# consistent with step-function correlation function characterized by 
# parameters c_strength (> -1) and c_length (in fm)
def corr_shift_step(r, c_length, c_strength, avgprob):
    Vcorr = c_strength*4*np.pi/3*c_length**3
    C_inf = -Vcorr*avgprob
#     c_length += C_inf
#     C_inf = 0
    dr = 0
    
    # shift distance r->rp, dr = rp - r
    # change in step function occurs at rp = c_length,
    # or r = r_sw:
    r_sw = c_length*(1 + C_inf + c_strength)**(1/3)
    if r < r_sw:
        rp3 = r**3
        rp3 /= 1 + c_strength + C_inf
        rp = rp3**(1/3)
        dr = rp - r
    else:
        rp3 = r**3 - c_strength*c_length**3
        rp3 /= 1 + C_inf
        rp = rp3**(1/3)
        dr = rp - r
    return dr
        

# Add correlations to a nucleus by shifting nucleon positions
def add_correlations_realistic(nucleus, c_volume, c_extremum, corr_shift_interp):
    # Read integrated correlation function computed numerically from Monte Carlo configurations
    # (\int dr r^2 C(r))/(4\pi)
    # Corresponds to reference https://doi.org/10.1103/PhysRevC.101.061901
#     integrated_correlation_list = np.load('Ru_integrated_correlation.npy')
#     nrbins = 125
#     rmax = 2.5
#     rbins = np.linspace(0,rmax,nrbins)
#     deltar = rmax/nrbins
#     rlist=np.linspace(deltar/2,rmax,nrbins-1)
#     # Interpolate as a function of r
#     integrated_correlation = interp1d(np.insert(rlist,0,0), np.insert([sum(integrated_correlation_list[:i+1]) for i in range(len(rlist))],0,0), bounds_error=False, fill_value=(0,sum(integrated_correlation_list)))
#     extremum_reference = -1
#     strength_scale = c_extremum/extremum_reference
#     C_vol_reference = -0.16*strength_scale
# #     C_vol_reference = -0.0509*strength_scale
#     C_vol_scale = c_volume/C_vol_reference
#     length_scale = C_vol_scale**(1/3.)

    # Shift position of nucleons
    dist_matrix = squareform(pdist(nucleus))
    A = len(nucleus)
    cumulative_shift = np.zeros((A,3))
    for nucleonA in range(A-1):
        posA = nucleus[nucleonA]
        for nucleonB in range(nucleonA+1,A):
            posB = nucleus[nucleonB]
            r = dist_matrix[nucleonA, nucleonB]
            r_vec = posB - posA
            # r = np.linalg.norm(r_vec)
#             shift_magnitude = corr_shift_realistic(r, integrated_correlation, strength_scale, length_scale, avgprob)
            shift_magnitude = corr_shift_interp(r)
            if r <= EPS:
                continue
            shift = shift_magnitude*r_vec/r
            cumulative_shift[nucleonA] -= shift/2
            cumulative_shift[nucleonB] += shift/2
    return nucleus + cumulative_shift


def add_correlations_step(nucleus, c_length, c_strength, avgprob):
    # Create a distance matrix for the nucleus
    dist_matrix = squareform(pdist(nucleus))
    A = len(nucleus)
    cumulative_shift = np.zeros((A,3))
    for nucleonA in range(A-1):
        posA = nucleus[nucleonA]
        for nucleonB in range(nucleonA+1,A):
            posB = nucleus[nucleonB]
            r = dist_matrix[nucleonA, nucleonB]
            r_vec = posB - posA
            # r = np.linalg.norm(r_vec)

            shift_magnitude = corr_shift_step(r, c_length, c_strength, avgprob)
            if r <= EPS:
                continue
            shift = shift_magnitude*r_vec/r
            cumulative_shift[nucleonA] -= shift/2
            cumulative_shift[nucleonB] += shift/2
#             cumulative_shift[nucleonA] -= shift
    return nucleus + cumulative_shift



# Modifies coordinates of a nucleon according to angular deformation parameterized by coefficients beta_{l,m}
# def deform(r,costheta,phi,Rws,Rstep, w,b20,b22,b3,db20,db22,db3):
# def deform(r,costheta,phi,Rws,Rstep, w,beta20,beta22,beta3, f2, fp2,f3,fp3):
def deform_nucleon(r,costheta,phi,R,beta20,beta22,beta3,beta4,f2,fp2,f3,fp3, f4, fp4):
#     beta20 = b2*math.cos(gamma)
#     beta22 = b2*math.sin(gamma)/np.sqrt(2)
    costheta = float(np.clip(costheta, -1.0, 1.0))
    theta = np.arccos(costheta)
    dtheta=0
    dphi = 0
    dr = 0
    sintheta = math.sqrt(max(0.0, 1-costheta**2))
    if r <= EPS:
        return r, costheta, phi
    f2r = f2(r)
    fp2r = fp2(r)
    f3r = f3(r)
    fp3r = fp3(r)
    f4r = f4(r)
    fp4r = fp4(r)

    # Angular shifts
    dtheta += R/r/r*beta20*f2r*dY20_dtheta(costheta,sintheta,phi)
    dtheta += R/r/r*beta22*f2r*dY22_dtheta(costheta,sintheta,phi)
    dtheta += R/r/r*beta3*f3r*dY30_dtheta(costheta,sintheta,phi)
    dtheta += R/r/r*beta4*f4r*dY40_dtheta(costheta,sintheta,phi)

    sintheta2 = max(sintheta**2, EPS)
    dphi += R/r**2/sintheta2*beta22*f2r*dY22_dphi(costheta,sintheta,phi)

    # Radial shifts
    dr += R*beta20*fp2r*Y_20(costheta,phi)
    dr += R*beta22*fp2r*Y_22(costheta,phi)
    dr += R*beta3*fp3r*Y_30(costheta,phi)
    dr += R*beta4*fp4r*Y_40(costheta,phi)
    
    
#     if np.abs(dr) >= np.abs(r):
#         print("dr, r = " + str(dr) + ", " + str(r))
#     if (theta+(dtheta) < 0 or theta+(dtheta) > np.pi):
#         print("dtheta, theta = " + str(dtheta) + ", " + str(theta))
#     if (phi+dphi < 0 or phi+dphi > 2*np.pi):
#         print("dphi, phi = " + str(dphi) + ", " + str(phi))
              
    
    r += dr
    theta += dtheta
    costheta = np.cos(theta)
    phi += dphi
    
    return r, costheta, phi


# Take configuration of a nucleus and deform it by shifting each nucleon with deform() function
# Uncorrelated nucleons will still be uncorrelated after deformation.
# Updated to include beta4 (hexadecapole deformation)
def deform_nucleus(nucleus, R, beta2, gamma, beta3, beta4, f2, fp2, f3, fp3, f4, fp4):
#     rmin = 1.0e-1
    rmin = R/10
    for nucleon, position in enumerate(nucleus):
        x,y,z = position
        r = np.linalg.norm(position)
        if r > rmin:
            phi = math.atan2(y,x)
            costheta = z/r

            beta20 = beta2*math.cos(gamma)
            beta22 = beta2*math.sin(gamma)/np.sqrt(2)

            nsteps = 10
            for step in range(nsteps):
                r,costheta,phi = deform_nucleon(r,costheta,phi,R,beta20/nsteps,beta22/nsteps,beta3/nsteps,beta4/nsteps,f2,fp2,f3,fp3,f4,fp4)

            x, y, z = cartesian(r,costheta,phi)
            nucleus[nucleon] = np.array([x,y,z])
        
    return nucleus


# Place nucleons according 3D step function + 3D Gaussion, from pre-generated random seeds
def place_nucleon(R_step, w_gauss, seed):    
    # radial coordinate
    radius_seed = seed[POS_SEEDS['radius']]
    # p(r) ~ r²dr, cdf(r) ~ r³, r ~ seed_r^(1/3), cdf sampled uniform [0,1]
    # plus max radius is R_step
    r = radius_seed**(1./3.)*R_step
    # polar angle
    costheta_seed = seed[POS_SEEDS['costheta']]
    # uniform in [-1,1]  
    costheta = 2.*costheta_seed - 1.
    # axial angle
    phi_seed = seed[POS_SEEDS['phi']]
    # uniform in [0,2 pi]
    phi = 2.*np.pi*phi_seed
    
    x,y,z = cartesian(r,costheta,phi)
    
    
    # folding with 3d gaussian
    gauss_seed_x = seed[POS_SEEDS['gauss_x']]
    gauss_seed_y = seed[POS_SEEDS['gauss_y']]
    gauss_seed_z = seed[POS_SEEDS['gauss_z']]
    
    x += w_gauss*gauss_seed_x
    y += w_gauss*gauss_seed_y
    z += w_gauss*gauss_seed_z
    
    
    return np.array([x,y,z])




# Build nucleus by randomly placing nucleons independently according to a
# spherically-symmetric distribution, then adding angular deformation,
# then adding short-range pair correlation.  Result is list of positions in cartesian coordinates.
def build_nucleus(seeds_nucleus, n_nucleons, isobar_config, avgprob, f2, fp2, f3, fp3, f4, fp4, corr_shift_interp):
    R_ws = isobar_config['R_ws']
    R_step = isobar_config['R_step']
    w_gauss = isobar_config['w_gauss']
    beta2 = isobar_config['beta2']
    gamma = isobar_config['gamma']
    beta3 = isobar_config['beta3']
    beta4 = isobar_config['beta4']
    c_volume = isobar_config['correlation_volume']
    c_extremum = isobar_config['correlation_extremum']
    realistic_correlation = isobar_config['realistic_correlation']

    # Place nucleons via 3D step + Gaussian
    nucleus = np.empty((n_nucleons,3))
    for n in range(n_nucleons):
#         nucleus[n,:] = place_nucleon(Rws, R_step, w_gauss, beta2, gamma, beta3, seeds_nucleus[n], f2, fp2, f3, fp3)
        nucleus[n,:] = place_nucleon(R_step, w_gauss, seeds_nucleus[n])
        
    # Perform angular deformation by shifting nucleon positions
    if (beta2 != 0 or beta3 != 0 or beta4 != 0):
        nucleus = deform_nucleus(nucleus, R_ws, beta2, gamma, beta3, beta4, f2, fp2,f3,fp3, f4, fp4)
        

    # Add short-range correlations by shifting nucleon positions
    if c_volume != 0:
        if realistic_correlation == 1:
            nucleus = add_correlations_realistic(nucleus, c_volume, c_extremum, corr_shift_interp)
        else:
            c_strength = c_extremum
            c_length = (abs(c_volume)*3/4/np.pi)**(1/3.)
            nucleus = add_correlations_step(nucleus, c_length, c_strength, avgprob)
        
    return nucleus

    

def _nested_key_exists(mapping, keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _nested_value(mapping, keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def validate_configuration(confs):
    errors = []

    if not isinstance(confs, dict):
        return ["Top-level YAML document must be a mapping."]

    required_top_level = [
        ('isobar_samples',),
        ('isobar_properties',),
        ('isobar_samples', 'number_configs', 'value'),
        ('isobar_samples', 'number_nucleons', 'value'),
        ('isobar_samples', 'seeds_file', 'filename'),
        ('isobar_samples', 'output_path', 'dirname'),
    ]
    for path in required_top_level:
        if not _nested_key_exists(confs, path):
            errors.append(f"Missing required configuration key: {'.'.join(path)}")

    if errors:
        return errors

    n_configs = _nested_value(confs, ('isobar_samples', 'number_configs', 'value'))
    if not isinstance(n_configs, int) or n_configs <= 0:
        errors.append("isobar_samples.number_configs.value must be a positive integer")

    n_nucleons = _nested_value(confs, ('isobar_samples', 'number_nucleons', 'value'))
    if not isinstance(n_nucleons, int) or n_nucleons <= 0:
        errors.append("isobar_samples.number_nucleons.value must be a positive integer")

    njobs = _nested_value(confs, ('isobar_samples', 'number_of_parallel_processes', 'value'))
    if njobs is not None and (not isinstance(njobs, int) or njobs == 0):
        errors.append("isobar_samples.number_of_parallel_processes.value must be a non-zero integer")

    start_configuration = _nested_value(confs, ('isobar_samples', 'start_configuration', 'value'))
    if start_configuration is not None and (not isinstance(start_configuration, int) or start_configuration < 0):
        errors.append("isobar_samples.start_configuration.value must be a non-negative integer")

    isobar_properties = confs['isobar_properties']
    if not isinstance(isobar_properties, dict):
        errors.append("isobar_properties must be a mapping")
        return errors

    isobar_index = 1
    found_isobar = False
    while f'isobar{isobar_index}' in isobar_properties:
        found_isobar = True
        isobar_key = f'isobar{isobar_index}'
        isobar_conf = isobar_properties[isobar_key]

        if not isinstance(isobar_conf, dict):
            errors.append(f"{isobar_key} must be a mapping")
            isobar_index += 1
            continue

        required_isobar_paths = [
            ('WS_radius', 'value'),
            ('WS_diffusiveness', 'value'),
            ('isobar_name',),
        ]
        for path in required_isobar_paths:
            if not _nested_key_exists(isobar_conf, path):
                errors.append(f"Missing required configuration key: {isobar_key}.{'.'.join(path)}")

        R_ws = _nested_value(isobar_conf, ('WS_radius', 'value'))
        if R_ws is not None and R_ws <= 0:
            errors.append(f"{isobar_key}.WS_radius.value must be positive")

        a_ws = _nested_value(isobar_conf, ('WS_diffusiveness', 'value'))
        if a_ws is not None and a_ws <= 0:
            errors.append(f"{isobar_key}.WS_diffusiveness.value must be positive")

        step_radius = _nested_value(isobar_conf, ('step_radius', 'value'))
        if step_radius is not None and step_radius <= 0:
            errors.append(f"{isobar_key}.step_radius.value must be positive")

        step_diffusiveness = _nested_value(isobar_conf, ('step_diffusiveness', 'value'))
        if step_diffusiveness is not None and step_diffusiveness <= 0:
            errors.append(f"{isobar_key}.step_diffusiveness.value must be positive")

        realistic_correlation = _nested_value(isobar_conf, ('realistic_correlation', 'value'), 0)
        if realistic_correlation not in (0, 1):
            errors.append(f"{isobar_key}.realistic_correlation.value must be 0 or 1")

        correlation_extremum = _nested_value(isobar_conf, ('correlation_extremum', 'value'))
        if correlation_extremum is not None and correlation_extremum < -1:
            errors.append(f"{isobar_key}.correlation_extremum.value cannot be smaller than -1")

        correlation_length = _nested_value(isobar_conf, ('correlation_length', 'value'))
        if correlation_length is not None and correlation_length <= 0:
            errors.append(f"{isobar_key}.correlation_length.value must be positive")

        length_scale = _nested_value(isobar_conf, ('length_scale', 'value'))
        if length_scale is not None and length_scale <= 0:
            errors.append(f"{isobar_key}.length_scale.value must be positive")

        isobar_index += 1

    if not found_isobar:
        errors.append("No isobar definitions found under isobar_properties")

    return errors



#%%
def main():
    if len(sys.argv) != 2:
        print("Usage: ./build_isobars.py conf.yaml")
        sys.exit(1)

    # Read parameter file containing settings and all isobar configurations
    conffile = sys.argv[1]
    try:
        with open(conffile, 'r') as stream:
            confs = yaml.load(stream, Loader=SafeLoader)
    except IOError:
        print(f"Error: Could not read file {conffile}")
        sys.exit(1)

    validation_errors = validate_configuration(confs)
    if validation_errors:
        print("Error: Invalid configuration file:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    # with open(conffile, 'r') as stream:
    #     confs = yaml.load(stream,Loader=SafeLoader)

    # Read meta settings, applicable to all isobar configurations.  Currently, all nuclei must have the same number of nucleons.
    try:
        conf_samples = confs['isobar_samples']
        n_configs = conf_samples['number_configs']['value']
        n_nucleons = conf_samples['number_nucleons']['value']
        seeds_file = conf_samples['seeds_file']['filename']
        out_dir = conf_samples['output_path']['dirname']
        # njobs = 1  # default to serial calculation
        # if 'number_of_parallel_processes' in conf_samples:
        #     njobs = conf_samples['number_of_parallel_processes']['value']
        # if(not os.path.isdir(out_dir)):
        #     os.mkdir(out_dir)
    except KeyError:
        print(f"Error: Missing key in configuration file.  Check {conffile}")
        sys.exit(1)
    # conf_samples = confs['isobar_samples']
    # n_configs = conf_samples['number_configs']['value']
    # n_nucleons = conf_samples['number_nucleons']['value']
    # seeds_file = conf_samples['seeds_file']['filename']
    # out_dir = conf_samples['output_path']['dirname']
    njobs = 1  # default to serial calculation
    if 'number_of_parallel_processes' in conf_samples:
        njobs = conf_samples['number_of_parallel_processes']['value']

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print(f"Error: Could not create output directory {out_dir}: {exc}")
        sys.exit(1)
    
    try:
        with h5py.File(seeds_file, 'r') as f:
            seeds = f['isobar_seeds'][:,:,:]
    except IOError:
        print(f"Error: Could not read file {seeds_file}")
        sys.exit(1)

    nseeds = seeds.shape[0]
    start_configuration = 0
    if 'start_configuration' in conf_samples:
        start_configuration = conf_samples['start_configuration']['value']
        if start_configuration < 0 or start_configuration >= nseeds:
            print(f"Error: start_configuration {start_configuration} out of range [0, {nseeds})")
            sys.exit(1)

    # Set starting configuration
    seeds = np.array(seeds[start_configuration:,:,:])

    if seeds.shape[0] < n_configs:
        n_configs = seeds.shape[0]
    if seeds.shape[1] < n_nucleons:
        print("Error: Requested number of nucleons per nucleus, A, larger than provided seeds")
        n_nucleons = seeds.shape[1]
        sys.exit(1)

    n_isobars = 0
    isobars = []

    # Read and pre-process parameters for each isobar configuration
    while ('isobar'+str(n_isobars+1) in confs['isobar_properties'].keys()):
            isobar_conf = confs['isobar_properties']['isobar'+str(n_isobars+1)]

            # Default values for all parameters, if they don't appear in the input file
            beta2, gamma, beta3, beta4 = 0, 0, 0, 0 # angular deformation parameters (added beta4)
            realistic_correlation = 0 # 1 if using realistic correlation, 0 if using step function
            correlation_volume = 0 # volume of desired correlation \int dr r^2 C(r)
            correlation_extremum = -1 # minimum (if negative) or maximum (if positive) of desired correlation
            
            # Radius and skin thickness parameters
            # Woods-Saxon
            try:
                R_ws = isobar_conf['WS_radius']['value']
                a_ws = isobar_conf['WS_diffusiveness']['value']
            except KeyError:
                print(f"Error: Missing WS_radius or WS_diffusiveness in isobar{n_isobars+1} configuration.  Check {conffile}")
                sys.exit(1)
            # or step+Gauss function
            R_step = 0
            if 'step_radius' in isobar_conf:
                R_step = isobar_conf['step_radius']['value']
            diffusiveness = 0
            if 'step_diffusiveness' in isobar_conf:
                diffusiveness = isobar_conf['step_diffusiveness']['value']
            
            # Angular deformation parameters
            # beta2, gamma, beta3, beta4 = 0, 0, 0, 0
            if 'beta_2' in isobar_conf:
                beta2 = isobar_conf['beta_2']['value']
            if 'gamma' in isobar_conf:
                gamma = isobar_conf['gamma']['value']
            if 'beta_3' in isobar_conf:
                beta3 = isobar_conf['beta_3']['value']
            if 'beta_4' in isobar_conf:
                beta4 = isobar_conf['beta_4']['value']    


            # Short-range correlation parameters
            # correlation_volume = 0 # volume of desired correlation \int dr r^2 C(r)
            # correlation_extremum = -1 # minimum (if negative) or maximum (if positive) of desired correlation
            if 'correlation_volume' in isobar_conf:
                correlation_volume = isobar_conf['correlation_volume']['value']
            if 'correlation_extremum' in isobar_conf:
                correlation_extremum = isobar_conf['correlation_extremum']['value']
            # realistic_correlation = 0
            if 'realistic_correlation' in isobar_conf:
                realistic_correlation = isobar_conf['realistic_correlation']['value']
#                 print(f'{realistic_correlation=}')
            if realistic_correlation != 0: # correlation from Alvioli, et al. sampled Ru nuclei
                # Read previously-extracted, binned correlation from Monte Carlo configurations C(r)
                correlation_file = 'correlation_files/Ru_correlation_Alvioli.npy'
                if 'correlation_file' in isobar_conf:
                    correlation_file = isobar_conf['correlation_file']['value']
                
                # Can modify correlation function by scaling length or magnitude
                extremum_reference = -1 # minimum of original correlation
                strength_scale = correlation_extremum/extremum_reference
                # print(f'{strength_scale=}, {correlation_extremum=}, {extremum_reference=}')
                if 'strength_scale' in isobar_conf:
                    if correlation_extremum !=-1:
                        print("Both correlation_extremum and strength_scale specified.  Ignoring strength_scale.")
                    else:
                        strength_scale = isobar_conf['strength_scale']['value']
                        correlation_extremum = -strength_scale

                # C_vol_reference = -0.213*strength_scale
                length_scale = 1
                if 'length_scale' in isobar_conf:
                    if correlation_volume !=0:
                        print("Both correlation_volume and length_scale specified.  Ignoring length_scale.")
                        # C_vol_scale = correlation_volume/C_vol_reference
                        # length_scale = C_vol_scale**(1/3.)
                    else:
                        length_scale = isobar_conf['length_scale']['value']
                        correlation_volume = np.pi*correlation_extremum*length_scale**3*4/3
            else: # step function correlation
                correlation_strength = 0
                correlation_length = 0
                if 'correlation_strength' in isobar_conf:
                    correlation_strength = isobar_conf['correlation_strength']['value']
                    if correlation_extremum !=-1:
                        print("Both correlation_extremum and correlation_strength specified.  Ignoring correlation_strength.")
                    else:
                        correlation_extremum = correlation_strength
    #                 correlation_length = (correlation_volume/correlation_strength)**(1/3)/np.pi
                if 'correlation_length' in isobar_conf:
                    if correlation_volume != 0:
                        print("Both correlation_volume and correlation_length specified.  Ignoring correlation_length.")
                    else:
                        correlation_length = isobar_conf['correlation_length']['value']
                        correlation_volume = np.pi*correlation_strength*correlation_length**3*4/3
            
            
            if (correlation_extremum < -1):
                raise Exception('correlation_extremum/correlation_strength cannot be smaller than -1')
                    
            # print(f'{correlation_extremum=}, {correlation_volume=}') 
            isobars.append({
                'name': isobar_conf['isobar_name'],
                'R_ws': R_ws,
                'a_ws': a_ws,
                'R_step': R_step,
                'w_gauss': diffusiveness,
                'beta2': beta2,
                'gamma': gamma,
                'beta3': beta3,
                'beta4': beta4,
                'correlation_volume': correlation_volume,
                'correlation_extremum': correlation_extremum,
                'realistic_correlation': realistic_correlation,
                'correlation_file': correlation_file if realistic_correlation != 0 else None,
            })
            n_isobars +=1
        

    
    # Prepare each isobar configuration
    for isobar_index, isobar_config in enumerate(isobars):
        print(f'Building isobar {isobar_index+1}')
        R_ws = isobar_config['R_ws']
        a_ws = isobar_config['a_ws']
        R_step = isobar_config['R_step']
        w = isobar_config['w_gauss']
        beta2 = isobar_config['beta2']
        gamma = isobar_config['gamma']
        beta3 = isobar_config['beta3']
        beta4 = isobar_config['beta4']
        realistic_correlation = isobar_config['realistic_correlation']
#         print(f'{realistic_correlation=}')
        
        if R_step == 0 or w == 0:
            (R_step, w_step) = Rst_w_from_WS(R_ws,a_ws)
            isobar_config['R_step'] = R_step
            isobar_config['w_gauss'] = w_step
            print(f'R_step and w not specified.  Determining from Woods-Saxon parameters {R_ws=} fm, {a_ws=} fm --> {R_step=:.3f} fm, {w_step=:.3f} fm')
            

        # Prepare angular deformation.  Solve differential equation once and 
        # pass interpolation functions via arguments for evaluation in deform_*()
        if (beta2 != 0 or beta3 != 0 or beta4 != 0):
            print(f'Solving differential equation for angular deformation.  {beta2=}, {gamma=}, {beta3=}, {beta4=}')
            rmin = R_ws/10
            rmax = 3*R_ws
            
            # boundary conditions (at r=rmax) for f, f', f_homo, f'_homo
            z_init = [rmax,1,1,0]
            
            # multipole l = 2
            args=(R_ws,a_ws,2) # l = 2
            res2 = solve_ivp(fun=lambda t,y: diff_eq(t,y,*args), y0=z_init,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)

            f2 = interp1d(res2.t, res2.y[0] - res2.y[1,-1]/res2.y[3,-1]*res2.y[2])
            fp2 = interp1d(res2.t, res2.y[1] - res2.y[1,-1]/res2.y[3,-1]*res2.y[3])
            
            # multipole l = 3
            args=(R_ws,a_ws,3) # l = 3
            res3 = solve_ivp(fun=lambda t,y: diff_eq(t,y,*args), y0=z_init,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)

            f3 = interp1d(res3.t, res3.y[0] - res3.y[1,-1]/res3.y[3,-1]*res3.y[2])
            fp3 = interp1d(res3.t, res3.y[1] - res3.y[1,-1]/res3.y[3,-1]*res3.y[3])

            # multipole l = 4 (hexadecapole)
            args=(R_ws,a_ws,4) # l = 4 
            res4 = solve_ivp(fun=lambda t,y: diff_eq(t,y,*args), y0=z_init,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)

            f4 = interp1d(res4.t, res4.y[0] - res4.y[1,-1]/res4.y[3,-1]*res4.y[2])
            fp4 = interp1d(res4.t, res4.y[1] - res4.y[1,-1]/res4.y[3,-1]*res4.y[3])
        else:
            f2 = 0
            fp2 = 0
            f3 = 0
            fp3 = 0
            f4 = 0
            fp4 = 0

#         Prepare correlation
        correlation_volume = isobar_config['correlation_volume']
        correlation_extremum = isobar_config['correlation_extremum']
        # average probability <\rho> = \int d^3r \rho^2, required for calculation of shift due to short-range correlation.
        # For effiency, compute only once and pass the value via arguments
        avgprob = quad(lambda r: r**2*Woods_Saxon(r,R_ws,a_ws)**2, 0,np.inf)[0]*4*np.pi
        corr_shift_interp = 0 #only needed for realistic correlation
        if correlation_volume != 0:
            # Realistic correlation means correlation function that matches 2-body correlation of Alvioli, Strikman, et al
            # Otherwise, a simple step function is used
            if realistic_correlation == 1:
                correlation_file = isobar_config['correlation_file']
                correlation_list = np.load(correlation_file)
                # binning that was used to generate correlation_file
                nrbins = 125
                rmax = 2.5
                rbins = np.linspace(0,rmax,nrbins)
                deltar = rmax/nrbins
                rlist=np.linspace(deltar/2,rmax,nrbins-1)
                dr3 = [rbins[i]**3-rbins[i-1]**3 for i in range(1,len(rbins))]
                d3rcorrelation_list = dr3*correlation_list
                d3rcorrelation_list *= 4/3*np.pi
                # Interpolate integral \int_0^r d(r^3) C(r) as a function of r
                # integrated_correlation = interp1d(np.insert(rlist,0,0), np.insert([sum((d3r*correlation_list)[:i+1]) for i in range(len(rlist))],0,0), bounds_error=False, fill_value=(0,sum(correlation_list)))
                integrated_correlation = interp1d(np.insert(rlist,0,0), np.insert([sum((d3rcorrelation_list)[:i+1]) for i in range(len(rlist))],0,0), bounds_error=False, fill_value=(0,sum(correlation_list)))
                extremum_reference = -1
                strength_scale = correlation_extremum/extremum_reference
# #                 C_vol_reference = -0.16*strength_scale
                C_vol_reference = -0.213*strength_scale
#             #     C_vol_reference = -0.0509*strength_scale
                C_vol_scale = correlation_volume/C_vol_reference
                length_scale = C_vol_scale**(1/3.)
                print(f'Realistic correlation selected, scaled by {length_scale=:.2f}, {strength_scale=:.2f}')
                R_ws_Ru = 5.085 # Woods-Saxon radius of Ru as used in Alvioli, Strikman, et al
                rfullmax = 3*R_ws_Ru
                nrfullpoints = 1000
                rfulllist = np.linspace(0,rfullmax,nrfullpoints)
                correlation_shift_list = np.array([corr_shift_realistic(r, integrated_correlation, strength_scale, length_scale, avgprob) for r in rfulllist])
                corr_shift_interp = interp1d(rfulllist, correlation_shift_list[:,0], bounds_error=False, fill_value = (0, correlation_shift_list[-1]))
            else:
                correlation_strength = correlation_extremum
                correlation_length = (abs(correlation_volume)*3/4/np.pi)**(1/3.)
                print(f'Step-function correlation selected.  {correlation_strength=}, {correlation_length=} fm')
#                 avgprob = quad(lambda r: r**2*Woods_Saxon(r,R_ws,a_ws)**2, 0,np.inf)[0]*4*np.pi

#         data = np.zeros((n_configs,n_nucleons,3),dtype=np.float)
#         njobs = 60
#         njobs = -1
#         print('building nuclei')
        data = Parallel(n_jobs=njobs)(delayed(build_nucleus)(seeds[s], n_nucleons, isobar_config, avgprob, f2, fp2, f3, fp3, f4, fp4, corr_shift_interp) for s in range(n_configs))
#         for s in range(n_configs):
#             data[s,:,:] = build_nucleus(seeds[s],n_nucleons,*isobars[isobar],f2,fp2,f3,fp3)
    
        with h5py.File(out_dir+'/'+isobar_config['name']+'.hdf', 'w') as f:
            data_set = f.create_dataset(isobar_config['name'],(n_configs,n_nucleons,3))
            data_set[:] = data 
            
#%%
if __name__ == "__main__":
#         cProfile.run('main()', 'stats')
        main()   
