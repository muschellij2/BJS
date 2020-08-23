import os
path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/python/'
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
import scipy.io
#%%
from fod_HCP_application import * #estimate response function, get mask info for ROI
from sphere_harmonics import *
from FOD_peak import *
from fod_estimation import *
from dwi_simulation import *

#%%
data_path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/data/'

#%% Load dwi image and gradient directions and bvalues information : load image takes time
org_bvecs = np.loadtxt(data_path+'bvecs')  # gradient directions along which DWI signals are measured
org_bvals = np.loadtxt(data_path+'bvals')  # b-values

#dmri_all = nib.load(data_path+'data.nii.gz').get_data().astype('float64')  # load DWI image: this the FLIRT registered image on the template 
#dmri_all = dmri_all[::-1]  # flip img_data_raw left-right to make the loaded image consistent with image loaded in matlab by the NIfTI package 

wm_mask_all = nib.load(data_path + 'T1w_acpc_dc_restore_1.25_brain_pve_2.nii.gz').get_data().astype('float64')
wm_mask_all = wm_mask_all[::-1] # do the same flipping
wm_mask_all = (wm_mask_all>0).astype('int')

roi_mask_all = nib.load(data_path + 'SLF_L_org.nii.gz').get_data().astype('float64') # get the mask for a ROI: mask created in FSL on the native space 
roi_mask_all = roi_mask_all[::-1] # do the same flipping
roi_mask_all = (roi_mask_all>0).astype('int')

wm_roi_mask = roi_mask_all * wm_mask_all
#%% Prestep 1: Extract ROI from the original image
# Notes: Since the whole brain dMRI dataset requires registration of connectome DB,  here we  provide  a faked selected ROI image (original signals plus artificial small noises). 
# If you want to process a whole brain dMRI image, you need to run the commented out codes.

############################### 
## codes for processing the whole brain dMRI image 
## Get ROI info from the mask image
coord, xmin, xmax, xsize, ymin, ymax, ysize, zmin, zmax, zsize = ROI_info(wm_roi_mask)
x_pre, y_pre, z_pre = np.arange(xmin, xmax+1), np.arange(ymin, ymax+1), np.arange(zmin, zmax+1)
wm_roi = wm_roi_mask[np.ix_(x_pre, y_pre, z_pre)]
#img_data, wm_roi = dmri_all[np.ix_(x_pre, y_pre, z_pre)], wm_roi_mask[np.ix_(x_pre, y_pre, z_pre)]
###############################
#%%
img_data=nib.load(data_path+'SLF_L_data.nii.gz').get_data().astype('float64')
#%% Index for streamline selection masks
seed_img = nib.load(data_path + 'SLF_L_seed_org.nii.gz').get_data().astype('float64')
seed_img = seed_img[::-1]
seed_img = (seed_img>0).astype('int')

target_img = nib.load(data_path + 'SLF_L_target_org.nii.gz').get_data().astype('float64')
target_img = target_img[::-1]
target_img = (target_img>0).astype('int')

seed_img, target_img = seed_img[np.ix_(x_pre, y_pre, z_pre)], target_img[np.ix_(x_pre, y_pre, z_pre)]

seed_coord, _, _, _, _, _, _, _, _, _ = ROI_info(seed_img)
target_coord, _, _, _, _, _, _, _, _, _ = ROI_info(target_img)

seed = seed_coord[2] * xsize * ysize + seed_coord[1] * xsize + seed_coord[0] + 1
target = target_coord[2] * xsize * ysize + target_coord[1] * xsize + target_coord[0] + 1

#%% Prestep 2(Takes Time): estimate response function parameters (b_factor and ratio) and SNR estimation from DWI signals from white matter.
#b_factor captures the scale information of the tensor in the definition of the response function: exp(-b*u^T*D*u); used in generating the Rmatrix
# ratio_response: captures the ratio between the major and minor eigenvalues od the tensor in the definition of the response function; used in generating the Rmatrix 
# Notes: b_factor, ratio_response, and SNR are estimated based on whole brain dMRI image. 

######################### 
## codes for processing the whole WM dMRI image: this requires the original D-MRI image, so below we give the results 
#b_factor, ratio_response, SNR = est_parameters_response(dmri_all, wm_mask_all, org_bvecs, org_bvals)
##########################

## Here we give the estimated results 
b_factor = 0.0017482689697695731 ## the largest (major) eigenvalue of the tensor 
ratio_response = 6.8356202340182435 ## ration between the major and minor eigenvalues of the tensor
SNR = 20.373593001153086. ##S0/sigma

#%% Prestep 2: Design matrix (SH matrix, R matrix)
b = 3000 # bvalue=3000
lmax = 10 #note n=90 gradient directions
lmax_update = 12
b3_loc, b3_vecs = bvecs_location(b = b, bvecs = org_bvecs, bvals = org_bvals) # get the 90 gradient directions corresponding to bvalue=3000
theta, phi = cart2sph(b3_vecs[0,:], b3_vecs[1,:], b3_vecs[2,:])

#generate design matrix: used for the methods (Notes: takes time to generate matrice R and SH)
SH = spharmonic(theta, phi, lmax_update)  # Phi
R = Rmatrix(b = b, lambda1 = b_factor, ratio = ratio_response, lmax = lmax_update) # response matrix R; design matrix = Phi *R

#%% Prestep 3: Extract ROI from the original image
b0_loc, _ = bvecs_location(b=0, bvecs = org_bvecs, bvals = org_bvals) # the indices of the bO (S0) image 

img_b0 = img_data[...,b0_loc] # DWI signals corresponding to b=0
img_b3 = img_data[...,b3_loc] # DWI signals corresponding to bvalue=3000

# estimation of S0
S0 = img_b0.mean(axis=3)
img_b0_indi = np.invert((img_b0.min(axis=3)>0)*(wm_roi>0))  #row-indices where the row-wise minimum>0 among white matter and ROI voxels
S0[img_b0_indi] = 1  # this step is to deal with out-side-of brain voxels; for ROI, this step is not really needed
img_b3[img_b3<0] = np.min(img_b3[img_b3>0])
DWI = (img_b3.T/S0.T).T 

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
xgrid_sp, ygrid_sp, zgrid_sp = 1.25, 1.25, 1.25
n1, n2, n3, _ = DWI.shape

braingrid = np.zeros((3, n1, n2, n3))

for i in range(n1):
    for j in range(n2):
        for k in range(n3):
            braingrid[:, i, j, k] = [(i-0.5*(n1-1))*xgrid_sp, (j-0.5*(n2-1))*ygrid_sp, (k-0.5*(n3-1))*zgrid_sp]

n_fiber, rmap = np.zeros(n1*n2*n3), np.zeros(n1*n2*n3)

temp_vec={i:[] for i in range(n3)}
temp_loc={i:[] for i in range(n3)}
temp_map={i:[] for i in range(n3)}

for k in range(n3):
    vec, loc, map = np.array([[]]*3).T, np.array([[]]*3).T, np.array([])
    print('slice' + str(k) +"/"+str(n3) + " "+str(np.sum(wm_roi[:, :, k])) + " voxels")
    for j in range(n2):
        for i in range(n1):            
            ind = k*n1*n2+j*n1+i
            if wm_roi[i, j, k]>0:
                fod_temp = BJS(DWI[i, j, k,:], SH, SHD, SHP, R, mu1, mu2, muinf, lmax)
                n_fiber[ind], peak_pos = FOD_Peak(fod_temp, idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
            else: ## for non-white-matter voxels, treat them as isotropic voxels (i.e, zero direction)
                n_fiber[ind] = 0

            if n_fiber[ind] > 0:
                vec = np.vstack((vec, peak_pos.T))
                loc = np.vstack((loc, np.tile(braingrid[:, i, j, k], (int(n_fiber[ind]), 1))))
                map = np.concatenate((map, [ind+1]*int(n_fiber[ind])))
            else:
                vec = np.vstack((vec, [np.nan]*3))
                loc = np.vstack((loc, braingrid[:, i, j, k]))
                map = np.concatenate((map, [ind+1]))

    temp_vec[k] = vec
    temp_loc[k] = loc
    temp_map[k] = map
vec=np.vstack([temp_vec[i] for i in range(n3)])
loc=np.vstack([temp_loc[i] for i in range(n3)])
map=np.concatenate([temp_map[i] for i in range(n3)], 0)

rmap = np.unique(map, return_index=True)[1]+1

temp_nfib2={i:[] for i in range(n3)}
for i in range(n3):
    n_fiber2 =  np.array([])
    for j in range(n1*n2*i, n1*n2*(i+1)):
        n_fiber2 = np.concatenate((n_fiber2, [n_fiber[j]]*max(int(n_fiber[j]), 1)))
    temp_nfib2[i] = n_fiber2
n_fiber2=np.concatenate([temp_nfib2[i] for i in range(n3)], 0)



result={'braingrid':braingrid,'loc':loc,'map':map,'n.fiber':n_fiber,'n.fiber2':n_fiber2,\
        'nn1':n1,'nn2':n2,'nn3':n3,'rmap':rmap,'vec':vec, 'x_size':n1,'y_size':n2,'z_size':n3,'xgrid.sp':xgrid_sp,'ygrid.sp':ygrid_sp,'zgrid.sp':zgrid_sp,\
        'seed':seed,'target':target}
#%%%
scipy.io.savemat(os.path.join(data_path,'peak.mat'),result) #save the results into .mat matlab data, this will later be fed to the tracking algorithm