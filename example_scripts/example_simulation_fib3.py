#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Example for Simulation (2 fiber case)
@author: Seungyong (30 Jan 2020)
"""

#%% Setup 

#%% Set working directory
import os


#path = '/Users/jiepeng/Dropbox/CRCNS_projects/FDD_estimation/codes/BJS/python/'
path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/python/'
#path = 'C:/Users/Seung/Dropbox/FDD_estimation/codes/BJS/python/'

os.chdir(path)
# import built in Packages
import numpy as np
import tqdm
# import functions: generate Spherical Harmonic basis, spherical sampling, peak detection, and FOD estimation, generating DWI data.
#from sphere_harmonics import spharmonic_eval, spharmonic, myresponse, Rmatrix
from sphere_harmonics import *
from sphere_mesh import spmesh
from FOD_peak import FOD_Peak
from fod_estimation import *
from dwi_simulation import *


#%% Simulation 
#%% Setup simulation parameters 
# True FOD setting: 3-fiber
weight = [0.33, 0.33, 0.34] #weight for each fiber 
num_fib = len(weight)

test_degree = 60 # separation angle  (60 or 90)
ratio = 10 #response function: ratio between the first eigenvalue and the avarage of the second and the third eigenvalues in the tensor. 

# Parameters for DWI generation and level of  SH basis used in the methods
J = 3 #Tessellation order: control the sampling scheme of the gradient directions on the sphere. E.g., J = 2.5 (n = 41), J = 3 (n = 91), J = 4(n = 321)     
lmax = 10 #the maximum level of real symmetrized spherical harmonic basis used in (BJS, superCSD, SHridge); for n=41 (J=2.5), use lmax=6; for n=91 (J=3), use lmax=10, for n=321, lmax=12
b = 3 #b-value (1 = 1,000mm^2/s, 3 = 3,000mm^2/s)
sigma = 0.02 #SNR = 1/sigma (sigma=0.02 (SNR = 50), sigma=0.05 (SNR = 20))


# level of SH basis used for the super-resolution updating
lmax_update = 12 #the maximum level of real symmetrized spherical harmonic basis used for the super-resolution updating in (BJS, superCSD)

# the number of replication
rep=100 


#%% Generate Design Matrix and DWI data 
#sample the gradient directions
pos, theta, phi, sampling_index = spmesh(J = J, half = True) # cartesian coordinates and spherical coordinates)on the sphere (equi_angle grid)

#generate design matrix: used for the methods (Notes: takes time to generate matrice R and SH)
SH = spharmonic(theta, phi, lmax_update) #  Phi
R = Rmatrix(b, ratio, lmax_update) # response matrix R; design matrix = Phi *R

# Generate True FOD: for data generation, and evaluation/plotting purpose 
# Option 1: Separation Angle 60
if (test_degree == 60):
    fib3_theta, fib3_phi = [np.pi/3, 2*np.pi/3, np.pi], [np.pi/2, np.pi/2, np.pi/2]
# Option 2: Separation Angle 90
if (test_degree == 90):
    fib3_theta, fib3_phi = [np.pi/2, np.pi, np.pi/2], [np.pi/2, np.pi/2, np.pi]

        
fib3_pos1 = sph2cart(fib3_theta[0], fib3_phi[0]) #cartesian coordinates of the true fiber directions
fib3_pos2 = sph2cart(fib3_theta[1], fib3_phi[1])
fib3_pos3 = sph2cart(fib3_theta[2], fib3_phi[2])

true_fod = true_fod_crossing(weight, fib3_theta, fib3_phi, lmax = 20) # projection of the FOD on a high order SH basis 


# Generate DWI observations: 
dwi_noiseless = myresponse_crossing(b, ratio, weight, fib3_theta, fib3_phi, theta, phi) #noisless signals
dwi = np.zeros((len(theta),rep))
for i in range(rep):
    dwi[:,i] = Rician_noise(dwi_noiseless, sigma, seed=i) ##add Rician noise to noiseless DWI signals


#%% Generate SH evaluation matrices and True FOD: take a bit of time
#generate SH evaluation matrix used in the super-resolution updating step
pos_dense_half, theta_dense_half, phi_dense_half, sampling_index_dense_half = spmesh(J = 5, half = True) # generate the evaluation grid points on the sphere (equi_angle grid)
SHD = spharmonic(theta_dense_half, phi_dense_half, lmax_update) # design matrix

#generate SH evaluation matrix on the whole sphere on a dense grid for plotting/evaluation purpose 
pos_dense, theta_dense, phi_dense = spmesh(J = 5, half = False) # generate the location of grid point on the sphere (equi_angle grid)
SHP = spharmonic(theta_dense, phi_dense, lmax_update) # 


#%% Parameters for methods 
# Tuning parameter grid for SHRidge: equally spaced on log scale on the coarseer grid, linearly equally spaced within the sub-intervals
lam = np.array([np.linspace(1e-06,1e-05,21)[:20],
            np.linspace(1e-05,1e-04,21)[:20],
            np.linspace(1e-04,1e-03,21)[:20],
            np.linspace(1e-03,1e-02,21)[:20],
            np.linspace(1e-02,1e-01,21)[:20],
            np.linspace(1e-01,1,21)[:20],
            np.linspace(1,10,21)[:20]])
lam = lam.reshape(1,-1)[0]


# Parameters for Peak detection
nbhd = 40  #neighborhood size
thresh = 0.4 #peak thresholding: ingore any peak < thresh * max
degree = 5 #clustering peaks within "degree" as one 
peak_cut = 4 # maximum number of peaks: only return the top "peak_cut" peaks  


#%% Auxiliary components of the methods:
# BJS: generate the associated eigenvalues of each block of the covariance matrix used in BJS estimator definition
L = int((lmax+1)*(lmax+2)/2) #the number of SH basis used by the methods 
SH_init = SH[:,:L]
R_init = R[:L,:L]
mu1, mu2, muinf = cal_mu(SH_init, R_init, lmax)

#SHRidge: generate the LB penalty matrix
P=penalty_mat(lmax)

#Peak detection: 
dis = pos_dense.T.dot(pos_dense) # pairwise distance between the grid points on the dense evaluation grid
idx = np.zeros(dis.shape, dtype=int) # for each grid point, sort other grid points in increasing distance to it; this is used as input for the peak detection  algorithm and generated once to avoid repeated calculation 
for i in range(dis.shape[0]):
    idx[i, :] = np.argsort(-dis[i, :])
    
    
#%% Apply the methods to the simulated DWI data
# arrayes to store results 
store_est_fod_bjs = np.zeros((SHP.shape[0],rep)) #For FOD estimation
store_est_fod_shridge = np.zeros((SHP.shape[0],rep))
store_est_fod_superCSD = np.zeros((SHP.shape[0],rep))

store_est_fod_coeff_shridge = np.zeros((L, rep))
store_num_fib_bjs = np.zeros(rep) #For peak detection
store_num_fib_shridge = np.zeros(rep)
store_num_fib_superCSD = np.zeros(rep)

store_est_sep1_bjs = [] #For separation angle 
store_est_sep2_bjs = []
store_est_sep3_bjs = []

store_est_sep1_shridge = []
store_est_sep2_shridge = []
store_est_sep3_shridge = []

store_est_sep1_superCSD = []
store_est_sep2_superCSD = []
store_est_sep3_superCSD = []


#% perform estimation 
for i in tqdm.tqdm(range(rep), desc="FOD estimation(BJS)"): # ith replicate 
    #BJS
    store_est_fod_bjs[:,i] = BJS(dwi[:,i], SH, SHD, SHP, R, mu1, mu2, muinf, lmax)


for i in tqdm.tqdm(range(rep), desc="FOD estimation(SHridge)"): # ith replicate 
    #SHRidge 
    store_est_fod_coeff_shridge[:,i], store_est_fod_shridge[:,i] = SH_ridge(dwi[:,i], SH, SHP, R, P, lmax, lam)    


for i in tqdm.tqdm(range(rep), desc="FOD estimation(superCSD)"): # ith replicate 
    #SCSD: take the SHRidge estimator as initial 
    store_est_fod_superCSD[:,i] = superCSD(dwi[:,i], SH, SHD, SHP, R, store_est_fod_coeff_shridge[:,i])
    
for i in tqdm.tqdm(range(rep), desc="Peak Detection:"): 
    #peak detection 
    num_fib_shridge, fib_loc_shridge = FOD_Peak(store_est_fod_shridge[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    num_fib_superCSD, fib_loc_superCSD = FOD_Peak(store_est_fod_superCSD[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    num_fib_bjs, fib_loc_bjs = FOD_Peak(store_est_fod_bjs[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)

    #record peak detection results 
    store_num_fib_shridge[i] = num_fib_shridge
    store_num_fib_superCSD[i] = num_fib_superCSD
    store_num_fib_bjs[i] = num_fib_bjs

    #get and record separation angle estimation  
    if (num_fib_shridge == num_fib):
        est_sep1_shridge, est_sep2_shridge, est_sep3_shridge = fib3_sep_angle(fib3_pos1, fib3_pos2, fib3_pos3, fib_loc_shridge[:,0], fib_loc_shridge[:,1], fib_loc_shridge[:,2])
        store_est_sep1_shridge.append(est_sep1_shridge)
        store_est_sep2_shridge.append(est_sep2_shridge)        
        store_est_sep3_shridge.append(est_sep3_shridge)
        
    if (num_fib_superCSD == num_fib):
        est_sep1_superCSD, est_sep2_superCSD, est_sep3_superCSD = fib3_sep_angle(fib3_pos1, fib3_pos2, fib3_pos3, fib_loc_superCSD[:,0], fib_loc_superCSD[:,1], fib_loc_superCSD[:,2])
        store_est_sep1_superCSD.append(est_sep1_superCSD)
        store_est_sep2_superCSD.append(est_sep2_superCSD)        
        store_est_sep3_superCSD.append(est_sep3_superCSD)
        
    if (num_fib_bjs == num_fib):
        est_sep1_bjs, est_sep2_bjs, est_sep3_bjs = fib3_sep_angle(fib3_pos1, fib3_pos2, fib3_pos3, fib_loc_bjs[:,0], fib_loc_bjs[:,1], fib_loc_bjs[:,2])
        store_est_sep1_bjs.append(est_sep1_bjs)
        store_est_sep2_bjs.append(est_sep2_bjs)        
        store_est_sep3_bjs.append(est_sep3_bjs)


#%%
#%% Evaluation 
#BJS:         
correct_bjs = len(np.where(store_num_fib_bjs == num_fib)[0]) #percentage of correct peak detection
under_bjs = len(np.where(store_num_fib_bjs < num_fib)[0])
over_bjs = len(np.where(store_num_fib_bjs > num_fib)[0])

#SHRidge
correct_shridge = len(np.where(store_num_fib_shridge == num_fib)[0])
under_shridge = len(np.where(store_num_fib_shridge < num_fib)[0])
over_shridge = len(np.where(store_num_fib_shridge > num_fib)[0])

#SCSD
correct_superCSD = len(np.where(store_num_fib_superCSD == num_fib)[0])
under_superCSD = len(np.where(store_num_fib_superCSD < num_fib)[0])
over_superCSD = len(np.where(store_num_fib_superCSD > num_fib)[0])    


#show results on screen : add an if to deal with 0% success rate; print the setting as in the tables
print("J: {}, lmax: {}, lmax_update:{}, SNR:{}, b-value:{}s/mm^2 ".format(J, lmax, lmax_update, 1/sigma, b*1000))
if (correct_bjs != 0):
    print("BJS: D.R {:.2f}, sep1_mean(s.e.) {:.3f}({:.3f}), sep2_mean(s.e.) {:.3f}({:.3f}), sep3_mean(s.e.) {:.3f}({:.3f})".format(
            correct_bjs/rep, np.mean(store_est_sep1_bjs), np.std(store_est_sep1_bjs)/np.sqrt(correct_bjs), np.mean(store_est_sep2_bjs), np.std(store_est_sep2_bjs)/np.sqrt(correct_bjs), np.mean(store_est_sep3_bjs), np.std(store_est_sep3_bjs)/np.sqrt(correct_bjs)))
else:
    print("BJS: D.R 0, sep1_mean(s.e.) - (-), sep2_mean(s.e.) - (-), sep3_mean(s.e.) - (-)")
    
if (correct_superCSD != 0):
    print("superCSD: D.R {:.2f}, sep1_mean(s.e.) {:.3f}({:.3f}), sep2_mean(s.e.) {:.3f}({:.3f}), sep3_mean(s.e.) {:.3f}({:.3f})".format(
            correct_superCSD/rep, np.mean(store_est_sep1_superCSD), np.std(store_est_sep1_superCSD)/np.sqrt(correct_superCSD), np.mean(store_est_sep2_superCSD), np.std(store_est_sep2_superCSD)/np.sqrt(correct_superCSD), np.mean(store_est_sep3_superCSD), np.std(store_est_sep3_superCSD)/np.sqrt(correct_superCSD)))
else:
    print("superCSD: D.R 0, sep1_mean(s.e.) - (-), sep2_mean(s.e.) - (-), sep3_mean(s.e.) - (-)")
    
if (correct_shridge != 0):
    print("SHridge: D.R {:.2f}, sep1_mean(s.e.) {:.3f}({:.3f}), sep2_mean(s.e.) {:.3f}({:.3f}), sep3_mean(s.e.) {:.3f}({:.3f})".format(
            correct_shridge/rep, np.mean(store_est_sep1_shridge), np.std(store_est_sep1_shridge)/np.sqrt(correct_shridge), np.mean(store_est_sep2_shridge), np.std(store_est_sep2_shridge)/np.sqrt(correct_shridge), np.mean(store_est_sep3_shridge), np.std(store_est_sep3_shridge)/np.sqrt(correct_shridge)))
else:
    print("SHridge: D.R 0, sep1_mean(s.e.) - (-), sep2_mean(s.e.) - (-), sep3_mean(s.e.) - (-)")


#%
bjs_mean =  np.mean(store_est_fod_bjs, axis=1)    
scsd_mean =  np.mean(store_est_fod_superCSD, axis=1)   
shridge_mean =  np.mean(store_est_fod_shridge, axis=1)
   
bjs_std =  np.std(store_est_fod_bjs, axis=1)    
scsd_std =  np.std(store_est_fod_superCSD, axis=1)   
shridge_std =  np.std(store_est_fod_shridge, axis=1)   

bjs_m2sd = bjs_mean +  2*bjs_std
scsd_m2sd = scsd_mean + 2*scsd_std
shridge_m2sd = shridge_mean + 2*shridge_std
#%
import scipy.io
file_name = "estfib3degree"+str(test_degree)+"J"+str(int(J))+'l'+str(lmax)+'b'+str(b)+'snr'+str(int(1/sigma))+'.mat'
scipy.io.savemat(file_name,{'degree':test_degree, 'J':J, 'b':b, 'lmax':lmax, 'lmax_update':lmax_update, 'snr' : 1/sigma, 'theta':fib3_theta,'phi':fib3_phi,'true':true_fod,'BJS_mean':bjs_mean, "SCSD_mean":scsd_mean, 'SHridge_mean':shridge_mean, 'BJS_m2sd':bjs_m2sd, 'SCSD_m2sd':scsd_m2sd, 'SHridge_m2sd':shridge_m2sd})