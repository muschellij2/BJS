#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:22:14 2020

@author: seungyong
"""

import os
#path = '/Users/jiepeng/Dropbox/CRCNS_projects/FDD_estimation/codes/BJS/python'
path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/python'
os.chdir(path)

#%%
import numpy as np
import tqdm
#For anaconda user : conda install -c conda-forge tqdm
import nibabel as nib
#Since nibabel is not the built-in library, you need to install it before executing this code.
#For anaconda user : conda install -c conda-forge nibabel
#For Python3 user : sudo pip3 install nibabel
import warnings # To prevent the error message in data import step (nibabel package problem) 
warnings.filterwarnings("ignore")

#%%
from fod_HCP_application import * #estimate response function, get mask info for ROI
from sphere_harmonics import *
from FOD_peak import *
from fod_estimation import *
from dwi_simulation import *

#%%
#data_path='/Users/jiepeng/Dropbox/CRCNS_projects/FDD_estimation/codes/BJS/data/'
data_path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/data/'


#%% Load dwi image and gradient directions and bvalues information : load image takes time
org_bvecs = np.loadtxt(data_path+'bvecs')  # gradient directions along which DWI signals are measured
org_bvals = np.loadtxt(data_path+'bvals')  # b-values

#img_data = nib.load(data_path+'data_brain_flirt.nii.gz').get_data().astype('float64')  # load DWI image: this the FLIRT registered image on the template 
#img_data = img_data[::-1]  # flip img_data_raw left-right to make the loaded image consistent with image loaded  in matlab by the NIfTI package 

mask_img = nib.load(data_path + 'Caudate(R).nii.gz').get_data().astype('float64') # get the mask for a ROI: here Caudate; mask created in fsl on the registered template 
mask_img = mask_img[::-1] # do the same flipping

#%% Prestep 1: Extract ROI from the original image
# Notes: Since the whole brain dMRI dataset requires registration of connectome DB,  here we  provide  a faked selected ROI image (original signals plus artificial small noises). 
# If you want to process a whole brain dMRI image, you need to run the commented out codes.

############################### 
## codes for processing the whole brain dMRI image 
## Get ROI info from the mask image
# coord, x_min, x_max, x_size, y_min, y_max, y_size, z_min, z_max, z_size = ROI_info(mask_img)
# part_img=img_data[coord]
###############################

part_img=nib.load(data_path+'Caudata(R)_noise.nii.gz').get_data().astype('float64')

#%% Prestep 2(Takes Time): estimate response function parameters (b_factor and ratio) and SNR estimation from DWI signals from ~a quarter of the entire brain image
#b_factor captures the scale information of the tensor in the definition of the response function: exp(-b*u^T*D*u); used in generating the Rmatrix
# ratio_response: captures the ratio between the major and minor eigenvalues od the tensor in the definition of the response function; used in generating the Rmatrix 
# Notes: b_factor, ratio_response, and SNR are estimated based on whole brain dMRI image. 

######################### 
## codes for processing the whole brain dMRI image 
#b_factor, ratio_response, SNR = est_parameters_response(img_data, org_bvecs, org_bvals, xmin=20, xmax=70, ymin=30, ymax=70, zmin=30, zmax=60)
##########################

## Here we give the estimated results 
b_factor = 0.00135 
ratio_response = 6.913
SNR = 54.593

#%% Prestep 2: Design matrix (SH matrix, R matrix)
b = 3 # b=3 corresponds to bvalue=3000
lmax = 10 #note n=90 gradient directions
lmax_update = 12
b3_loc, b3_vecs = bvecs_location(b=b*1000, bvecs = org_bvecs, bvals = org_bvals) # get the 90 gradient directions corresponding to bvalue=3000
theta, phi = cart2sph(b3_vecs[0,:], b3_vecs[1,:], b3_vecs[2,:])

#generate design matrix: used for the methods (Notes: takes time to generate matrice R and SH)
SH = spharmonic(theta, phi, lmax_update) #  Phi
R = Rmatrix(b = b * b_factor, ratio = ratio_response, lmax = lmax_update) # response matrix R; design matrix = Phi *R


#%% Prestep 3: Extract ROI from the original image
# Get ROI info from the mask image
coord, x_min, x_max, x_size, y_min, y_max, y_size, z_min, z_max, z_size = ROI_info(mask_img)
b0_loc, _ = bvecs_location(b=0, bvecs = org_bvecs, bvals = org_bvals) # the indices of the bO (S0) image 
b0_img = part_img[...,b0_loc] # DWI signals corresponding to b=0
#b0_img = b0_img[coord] #b0 image on the ROI

b3_img = part_img[...,b3_loc] # DWI signals corresponding to bvalue=3000
#b3_img = b3_img[coord] # DWI signals on the ROI

b0_indi=np.invert(b0_img.min(axis=1)>0) #row-indices where the row-wise minimum>0
S0 = b0_img.mean(axis=1) #S0 for each voxel is defined as the average b0 signals across the 18 b0 images
np.sum(b0_indi) # this is actually 0; 
S0[b0_indi] = 1 # this step is to deal with out-side-of brain voxels; for ROI, this step is not really needed

roi_img=(b3_img.T/S0.T).T #normalized the DWI signals on the ROI by their respective S0 value



#%% Generate SH evaluation matrices: take a bit of time
#generate SH evaluation matrix used in the super-resolution updating step
pos_dense_half, theta_dense_half, phi_dense_half, sampling_index_dense_half = spmesh(J = 5, half = True) # generate the evaluation grid points on the sphere (equi_angle grid)
SHD = spharmonic(theta_dense_half, phi_dense_half, lmax_update) # design matrix

#generate SH evaluation matrix on the whole sphere on a dense grid for plotting/evaluation purpose 
pos_dense, theta_dense, phi_dense = spmesh(J = 5, half = False) # generate the location of grid point on the sphere (equi_angle grid)
SHP = spharmonic(theta_dense, phi_dense, lmax_update) # 



#%% Parameters for methods 
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

#Peak detection: 
dis = pos_dense.T.dot(pos_dense) # pairwise distance between the grid points on the dense evaluation grid
idx = np.zeros(dis.shape, dtype=int) # for each grid point, sort other grid points in increasing distance to it; this is used as input for the peak detection  algorithm and generated once to avoid repeated calculation 
for i in range(dis.shape[0]):
    idx[i, :] = np.argsort(-dis[i, :])

#%%
fod_store=np.zeros(shape=(x_size,y_size,z_size,len(theta_dense))) # array to store the fod estimates 

roi_coord=coord[0]-x_min, coord[1]-y_min, coord[2]-z_min # make the ROI coordinates indices starting from zero, for recording purpose

#%% Step 1: BJS estimation
for k in tqdm.tqdm(range(roi_img.shape[0]), desc="FOD estimation (BJS)"): # loop over each voxel in the ROI
    if np.sum(roi_img[k,:]) > 0: # if there is some signal, i.e., voxel not outside of brain
        fod_store[roi_coord[0][k],roi_coord[1][k],roi_coord[2][k],:] = BJS(roi_img[k,:], SH, SHD, SHP, R, mu1, mu2, muinf, lmax) # apply BJS
        
        
#%% Step 2: peak detection and tracking; (tracking will be done in R)
peak_result = peak_detection(fod_store, idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, peak_cut)

peak_result['x_size'] = x_size #add x-coordinate box size to peak detection result: needed for the tracking algorithm
peak_result['y_size'] = y_size
peak_result['z_size'] = z_size


#%%
import scipy.io
scipy.io.savemat("HCP_peaks.mat",peak_result) #save the results into .mat matlab data, this will later be fed to the tracking algorithm



#%%