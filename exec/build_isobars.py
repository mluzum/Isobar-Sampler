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

# from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from mpmath import polylog

# import copy


# import cProfile


# N_SEEDS = 3 + 3 # 3 for r, theta, phi + 3 for delta_{x,y,z} Gaussian
POS_SEEDS = {'radius':0,'costheta':1,'phi':2,'gauss_x':3,'gauss_y':4,'gauss_z':5 }
# GAUSS_SEEDS = ['gauss_x','gauss_y','gauss_z'] # Gaussian seeds
# UNIFORM_SEEDS = ['radius','costheta','phi']
#%%





# rho'/rho for spherical Woods-Saxon
def rhodotoverrho(r,R,a):
    temp = -1.
    temp /= a
    temp /= 1 + math.exp((R-r)/a)
    return temp

# dimension-reduced differential equation -- inhomogenious (f, fdot) and homogenious (fhomo, fhomodot)
# returns derivative of (f, fdot, fhomo, fhomodot)
def der_z(r,z,Rint,aint,lint,nlint=1):
    f, fdot, fhomo, fhomodot = z
    temp = -fdot*(2/r + rhodotoverrho(r,Rint,aint)*nlint)
    temp += lint*(lint+1)/r**2*f
    temp += rhodotoverrho(r,Rint,aint)*nlint
    temphomo = -fhomodot*(2/r + rhodotoverrho(r,Rint,aint)*nlint)
    temphomo += lint*(lint+1)/r**2*fhomo
    return [fdot, temp, fhomodot, temphomo]


# Real spherical harmonics
def Y_22(costheta,phi):
    return 1./4.*np.sqrt(15./np.pi)*math.cos(2*phi)*(1.-costheta**2)

def Y_20(costheta,phi):
    return 1./4.*np.sqrt(5./np.pi)*(3.*costheta**2 -1)

def Y_30(costheta,phi):
    return np.sqrt(7./(4.*np.pi))*(5.*costheta**3 -3.*costheta)/2.

# Derivative with respect to theta, phi
def dY20_dtheta(costheta,sintheta,phi):
    return (-3./2.)*math.sqrt(5./np.pi)*costheta*sintheta

def dY22_dtheta(costheta,sintheta,phi):
    return 1/2.*math.sqrt(15./np.pi)*sintheta*costheta*np.cos(2*phi)

def dY30_dtheta(costheta,sintheta,phi):
    return 1./4.*math.sqrt(7./np.pi)*(3*sintheta-15*sintheta*costheta**2)

def dY22_dphi(costheta,sintheta,phi):
    return (-1./2.)*math.sqrt(15./np.pi)*sintheta**2*math.sin(2*phi)

# Spherical Woods-Saxon
def WS(r,R,a):
    return  1/(np.exp((r-R)/a)+1)

# Properly normalized probability density
def WSnm(r,R,a):
#     norm = quad(lambda r: r**2*WS(r,R,a), 0,np.inf)[0]*4*np.pi
# Use mppath implementation of polylog instead of numerical integral to normalize
    norm = -8*a**3*np.pi*polylog(3,-math.exp(R/a))
    return float(WS(r,R,a)/norm)

# Analytic form for (properly normalized) step+Gauss distribution
def ST(r,Rs,w):
    rt2 = np.sqrt(2)
    rtpi = np.sqrt(np.pi)
    rho1 = 0.
    rho = 0.
    if r == 0:
        rho1 = math.erf(Rs/w/rt2)
        rho1 *= 2*rtpi
        rho2 = -np.exp(-Rs**2/2/w**2)*2*rt2*Rs/w
    else:
        erfplus = math.erf((r+Rs)/w/rt2)
        erfminus = math.erf((r-Rs)/w/rt2)
        rho1 = rtpi*(erfplus - erfminus)
        expplus = np.exp(-((r+Rs)**2)/2/(w**2))
        expminus = np.exp(-((r-Rs)**2)/2/(w**2))
        rho2 = expplus - expminus
        rho2 *= rt2*w/r
    rho = rho1 + rho2
    rho *= 3/8/Rs**3/np.pi**(3/2)
    return rho

# KL divergence between Woods-Saxon and step+Gauss.  inp = (R_step,w).  We will choose these parameters to minimize the KL divergence for a given Woods-Saxon parameter a/R (i.e., an example corresponding to R=1, which can be scaled to obtain any R)
def KLwsst(inp, aoverR):
    Rs = inp[0]
    w = inp[1]
    KL = quad(lambda r: r**2*WSnm(r,1,aoverR)*np.log(WSnm(r,1,aoverR)/ST(r,Rs,w)),0,1+5*aoverR)[0]
    KL *= 4*np.pi
    return KL

# Returns best value of parameters (R_step, w) corresponding to Woods-Saxon parameters (R,a)
def Rstw(R,a):
    aoverR = a/R
    x0 = (1,aoverR)
    res = minimize(KLwsst, x0, args=aoverR)
    return R*res.x




# Spherical to cartesian coordinates
def cartesian(r,costheta,phi):
    sintheta = np.sqrt(1.-costheta**2)
    z = r*costheta
    x = r*sintheta*np.cos(phi)
    y = r*sintheta*np.sin(phi)
    
    return x, y, z
#%% 

# Compute magnitude of the coordinate shift of a pair of nucleons 
# consistent with step-function correlation function characterized by 
# parameters c_strength (> -1) and c_length (in fm)
def corr_shift(r, c_length, c_strength, avgprob):
#     avgprob = quad(lambda r: r**2*WSnm(r,R,a)**2, 0,np.inf)[0]*4*np.pi
    Vcorr = c_strength*4*np.pi/3*c_length**3
    Cinf = -Vcorr*avgprob
#     c_length += Cinf
#     Cinf = 0
    dr = 0
    # solve for rp(rsw) = c_length
    rsw = c_length*(1 + Cinf + c_strength)**(1/3)
    if r < rsw:
        rp3 = r**3
        rp3 /= 1 + c_strength + Cinf
        rp = rp3**(1/3)
        dr = rp - r
    else:
        rp3 = r**3 - c_strength*c_length**3
        rp3 /= 1 + Cinf
        rp = rp3**(1/3)
        dr = rp - r
    return dr
        
    
# Add correlations to a nucleus by shifting nucleon positions
def add_correlations(nucleus, c_length, c_strength, avgprob):
    A = len(nucleus)
    tempshift = np.zeros((A,3))
    for nucleonA in range(A-1):
        posA = nucleus[nucleonA]
        for nucleonB in range(nucleonA+1,A):
            posB = nucleus[nucleonB]
            rvec = posB - posA
            r = np.linalg.norm(rvec)
#             rA = np.linalg.norm(posA)
#             rB = np.linalg.norm(posB)
#             rhoA = ST(rA, 5.20678, 0.772445)
#             rhoB = ST(rB, 5.20678, 0.772445)
#             rA = math.sqrt(posA[0]**2 + posA[1]**2 + posA[2]**2)
#             rB = math.sqrt(posB[0]**2 + posB[1]**2 + posB[2]**2)
#             rhoA = WSnm(rA,5.09,0.46)
#             rhoB = WSnm(rB,5.09,0.46)
#             shiftA = corr_shift(r, c_length, c_strength, rhoA)
#             shiftB = corr_shift(r, c_length, c_strength, rhoB)
#             tempshift[nucleonA] -= shiftA*rvec/r/2
#             tempshift[nucleonB] += shiftB*rvec/r/2
            shiftmag = corr_shift(r, c_length, c_strength, avgprob)
            shift = shiftmag*rvec/r
            tempshift[nucleonA] -= shift/2
            tempshift[nucleonB] += shift/2
#             tempshift[nucleonA] -= shift
    return nucleus + tempshift



# Modifies coordinates of a nucleon according to angular deformation parameterized by coefficients beta_{l,m}
# def deform(r,costheta,phi,Rws,Rstep, w,b20,b22,b3,db20,db22,db3):
# def deform(r,costheta,phi,Rws,Rstep, w,beta20,beta22,beta3, f2, fp2,f3,fp3):
def deform(r,costheta,phi,Rws,beta20,beta22,beta3, f2, fp2,f3,fp3):
#     beta20 = b2*math.cos(gamma)
#     beta22 = b2*math.sin(gamma)
    theta = np.arccos(costheta)
    dtheta=0
    dphi = 0
    dr = 0
    sintheta = math.sqrt(1-costheta**2)
    f2r = f2(r)
    fp2r = fp2(r)
    f3r = f3(r)
    fp3r = fp3(r)
    dtheta += Rws/r/r*beta20*f2r*dY20_dtheta(costheta,sintheta,phi)
    dtheta += Rws/r/r*beta22*f2r*dY22_dtheta(costheta,sintheta,phi)
    dtheta += Rws/r/r*beta3*f3r*dY30_dtheta(costheta,sintheta,phi)

    dphi += Rws/r**2/sintheta**2*beta22*f2r*dY22_dphi(costheta,sintheta,phi)

    dr += Rws*beta20*fp2r*Y_20(costheta,phi)
    dr += Rws*beta22*fp2r*Y_22(costheta,phi)
    dr += Rws*beta3*fp3r*Y_30(costheta,phi)
    
    
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
def deform_nucleus(nucleus, Rws, beta2, gamma, beta3, f2, fp2,f3,fp3):
#     rmin = 1.0e-1
    rmin = Rws/10
    for i, nucleon in enumerate(nucleus):
        x,y,z = nucleon
#         x = nucleon[0]
#         y = nucleon[1]
#         z = nucleon[2]
#         r = math.sqrt(x*x + y*y + z*z)
        r = np.linalg.norm(nucleon)
        if r > rmin:
            phi = math.atan2(y,x)
            costheta = z/r

            beta20 = beta2*math.cos(gamma)
            beta22 = beta2*math.sin(gamma)

            nsteps = 10
            for step in range(nsteps):
                r,costheta,phi = deform(r,costheta,phi,Rws,beta20/nsteps,beta22/nsteps,beta3/nsteps, f2, fp2,f3,fp3)

            x, y, z = cartesian(r,costheta,phi)
            nucleus[i] = np.array([x,y,z])
        
    return nucleus


# Place nucleons according 3D step function + 3D Gaussion, from pre-generated random seeds
# def place_nucleon(Rws, R_step, w_gauss, beta2, gamma, beta3, seed, f2, fp2, f3, fp3):
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
    
#     sintheta = np.sqrt(1.-costheta**2)
#     z = r*costheta
#     x = r*sintheta*np.cos(phi)
#     y = r*sintheta*np.sin(phi)
    
    # folding with 3d gaussian
    gauss_seed_x = seed[POS_SEEDS['gauss_x']]
    gauss_seed_y = seed[POS_SEEDS['gauss_y']]
    gauss_seed_z = seed[POS_SEEDS['gauss_z']]
    
    x += w_gauss*gauss_seed_x
    y += w_gauss*gauss_seed_y
    z += w_gauss*gauss_seed_z
    
#     r = math.sqrt(newz**2+newx**2+newy**2)
#     phi = math.atan2(newy,newx)
#     costheta = newz/r
    
    
# #     # deformation
#     beta20 = beta2*math.cos(gamma)
#     beta22 = beta2*math.sin(gamma)
#     if (beta20 != 0 or beta22 != 0 or beta3 != 0):
#         print(f'doing deformation, {beta20}, {beta22}, {beta3}')
#         nsteps = 10
#         for step in range(nsteps):
#             r,costheta,phi = deform(r,costheta,phi,Rws,beta20/nsteps,beta22/nsteps,beta3/nsteps, f2, fp2,f3,fp3)

    
#     # cartesian coordinates
#     x, y, z = cartesian(r,costheta,phi)
    
    return np.array([x,y,z])




# Build nucleus by randomly placing nucleons independently according to a 
# spherically-symmetric distribution, then adding angular deformation,
# then adding short-range pair correlation.  Result is positions in cartesian coordinates.
def build_nucleus(seeds_nucleus, n_nucleons, Rws, aws, R_step, w_gauss, beta2, gamma, beta3, c_length, c_strength, avgprob, f2, fp2, f3, fp3):

#     nucleus = Parallel(n_jobs=60)(delayed(place_nucleon)(R_step, w_gauss, beta2, gamma,  beta3, seeds_nucleus[n], f2, fp2, f3, fp3) for n in range(n_nucleons))
#     nucleus = np.array(nucleus)


    # Place nucleons via 3D step + Gaussian
    nucleus = np.zeros((n_nucleons,3))
    for n in range(n_nucleons):
#         nucleus[n,:] = place_nucleon(Rws, R_step, w_gauss, beta2, gamma, beta3, seeds_nucleus[n], f2, fp2, f3, fp3)
        nucleus[n,:] = place_nucleon(R_step, w_gauss, seeds_nucleus[n])
        
    # Perform angular deformation by shifting nucleon positions
    if (beta2 != 0 or beta3 != 0):
        nucleus = deform_nucleus(nucleus, Rws, beta2, gamma, beta3, f2, fp2,f3,fp3)
        

    # Add short-range correlations by shifting nucleon positions
    if c_length !=0:
        nucleus = add_correlations(nucleus, c_length, c_strength, avgprob)
        
    return nucleus

# def run_isobar(seeds_nucleus, n_nucleons, R_step, w_gauss, beta2, gamma, beta3):
    

#%%
def main():
    
    # Read parameter file
    conffile = sys.argv[1]
    with open(conffile, 'r') as stream:
        confs = yaml.load(stream,Loader=SafeLoader)

    conf_samples = confs['isobar_samples']
    n_configs = conf_samples['number_configs']['value']
    n_nucleons = conf_samples['number_nucleons']['value']
    seeds_file = conf_samples['seeds_file']['filename']
    out_dir = conf_samples['output_path']['dirname']
    njobs = 1  # default to serial calculation
    if 'number_of_parallel_processes' in conf_samples:
        njobs = conf_samples['number_of_parallel_processes']['value']
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
            R_step = 0
            if 'step_radius' in isobar_conf:
                R_step = isobar_conf['step_radius']['value']
            diffusiveness = 0
            if 'diffusiveness' in isobar_conf:
                diffusiveness = isobar_conf['diffusiveness']['value']
            beta2 = isobar_conf['beta_2']['value']
            gamma = isobar_conf['gamma']['value']
            beta3 = isobar_conf['beta_3']['value']
            Rws = isobar_conf['WS_radius']['value']
            aws = isobar_conf['WS_diffusiveness']['value']
            correlation_length = 0
            if 'correlation_length' in isobar_conf:
                correlation_length = isobar_conf['correlation_length']['value']
            correlation_strength = 0
            if 'correlation_strength' in isobar_conf:
                correlation_strength = isobar_conf['correlation_strength']['value']
            isobars += [ [Rws,aws,R_step,diffusiveness,beta2,gamma,beta3, correlation_length, correlation_strength] ]
            isobar_names += [ isobar_conf['isobar_name'] ]
            n_isobars +=1
        
    # print(isobar_names)

    
    # Prepare each isobar configuration
    isobars = np.array(isobars)
    for i in range(n_isobars):
        print(f'Building isobar {i+1}')
        Rws = isobars[i,0]
        aws = isobars[i,1]
        beta2 = isobars[i,4]
        gamma = isobars[i,5]
        beta3 = isobars[i,6]
#         print(f'{beta2=}, {gamma=}, {beta3=}')
        if (Rws == 0 or aws == 0):
            raise Exception('Need to specify WS_radius and WS_diffusiveness in input file')
        
        R_step = isobars[i,2]
        w = isobars[i,3]
        if R_step == 0 or w == 0:
            (R_step, w_step) = Rstw(Rws,aws)
            isobars[i,2] = R_step
            isobars[i,3] = w_step
            print(f'R_step and w not specified.  Determining from Woods-Saxon parameters {Rws=} fm, {aws=} fm --> {R_step=:.3f} fm, {w_step=:.3f} fm')
            

        # Prepare angular deformation.  Solve differential equation once and 
        # pass interpolation functions via arguments for evaluation in deform()
        if (beta2 != 0 or beta3 != 0):
            print(f'Solving differential equation for angular deformation.  {beta2=}, {gamma=}, {beta3=} fm')
#             rmin = 1.0e-1
#             rmax = 2.2*Rws
            rmin = Rws/10
            rmax = 3*Rws
            
            zinit = [rmax,1,1,0]
            # multipole l = 2
            args=(Rws,aws,2) # l = 2
            res2 = solve_ivp(fun=lambda t,y: der_z(t,y,*args), y0=zinit,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)
            # multipole l = 3
            args=(Rws,aws,3) # l = 3
            res3 = solve_ivp(fun=lambda t,y: der_z(t,y,*args), y0=zinit,t_span=[rmax, rmin],rtol=1e-10,atol=1e-10)

            f2 = interp1d(res2.t,res2.y[0] - res2.y[1,-1]/res2.y[3,-1]*res2.y[2])
            fp2 = interp1d(res2.t,res2.y[1] - res2.y[1,-1]/res2.y[3,-1]*res2.y[3])
            f3 = interp1d(res3.t,res3.y[0] - res3.y[1,-1]/res3.y[3,-1]*res3.y[2])
            fp3 = interp1d(res3.t,res3.y[1] - res3.y[1,-1]/res3.y[3,-1]*res3.y[3])
        else:
            f2 = 0
            fp2 = 0
            f3 = 0
            fp3 = 0
        
        
        # Required for calculation of shift due to short-range correlation.
        # For effiency, compute only once and pass the value via arguments
        correlation_length = isobars[i,7]
        correlation_strength = isobars[i,8]
        avgprob = 0
        if correlation_strength !=0:
            print(f'Nontrivial correlation selected.  {correlation_strength=}, {correlation_length=} fm')
            avgprob = quad(lambda r: r**2*WSnm(r,Rws,aws)**2, 0,np.inf)[0]*4*np.pi

#         data = np.zeros((n_configs,n_nucleons,3),dtype=np.float)
#         njobs = 60
#         njobs = -1
        data = Parallel(n_jobs=njobs)(delayed(build_nucleus)(seeds[s],n_nucleons,*isobars[i],avgprob,f2,fp2,f3,fp3) for s in range(n_configs))
#         for s in range(n_configs):
#             data[s,:,:] = build_nucleus(seeds[s],n_nucleons,*isobars[i],f2,fp2,f3,fp3)
    
        with h5py.File(out_dir+'/'+isobar_names[i]+'.hdf', 'w') as f:
            data_set = f.create_dataset(isobar_names[i],(n_configs,n_nucleons,3))
            data_set[:] = data 
            
#%%
if __name__ == "__main__":
#         cProfile.run('main()', 'stats')
        main()   
