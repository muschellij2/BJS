# -*- coding: utf-8 -*-
"""
@Author: Seungyong
@Purpose: Estimate S0, sigma,  and ratio (for response function)
@Input: whole dwi at b=3000 for each subject
@Output: for each subject: estimated response function: response_est_result;
@Time: On the server: ~ 5 minutes/subject; a bit time consuming because of the nonlinear regression part (iterative), also depends on how large the brain we use (right now using the middle quarter of the whole brain)
@Update: 10/01/2019
"""

import numpy as np

#%%
# Gauss-Newton algorithm (check wiki of Levenberg–Marquardt algorithm for more details)
# objective function S(beta)=||y-exp(-X*beta)||^2=||y-exp(-bvec^T*D*bvec)||^2, where f(beta)=exp(-X*beta)
# Jacobian matrix J=\partial{exp(-X*beta)}/\partial{beta}=-X.*exp(-X*beta), 
# where .* multiplies each row of X by corresponding element in exp(-X*beta)
# increment delta satisfy (J^T*J)*delta=J^T*(y-exp(-X*beta))
# reason to re-parameterize D&bvec by beta&X is to ensure D is symmetric, 
# such that there are 6 parameters to estimate instead of 9 in D
def gauss_newton_dwi(bvec, bval, y, thresh = 1e-10):

	n = bvec.shape[1]
	X = np.zeros((n, 6))
	X[:, :3] = (bvec**2).T
	X[:, 3], X[:, 4], X[:, 5] = 2*bvec[0]*bvec[1], 2*bvec[0]*bvec[2], 2*bvec[1]*bvec[2]
	X = (X.T*bval).T

	beta = -np.linalg.solve(X.T.dot(X), X.T.dot(np.log(y)))  # initialization of beta (log(y)\approx -X*beta)
	delta = beta.copy()

	while np.linalg.norm(delta) > thresh:

		f_beta = np.exp(-X.dot(beta))
		J = -(X.T*f_beta).T  # python trick: make the trailing axes have the same dimension
		delta = np.linalg.solve(J.T.dot(J), J.T.dot(y-f_beta))
		beta += delta

	D = np.array([[beta[0], beta[3], beta[4]], [beta[3], beta[1], beta[5]], [beta[4], beta[5], beta[2]]])

	return D
#%%
# Estimate parameters for response function (b_factor, ratio) and SNR
# b_factor is a multiplicative factor used for adjusting the scale of -b*u^T*D*u in function "myresponse",
# leading to an appropriate R matrix
# in function "myresponse", D is assumed to have eigenvalues 1/ratio, 1/ratio and 1, however in practice, 
# the largest eigenvalue of D might be, say lambda. we then set b_factor=lambda, and change b value by b=b*b_factor,
# thus -b*u^T*D*u in function "myresponse" will have the correct value
def est_parameters_response(brain_image, bvecs, bvals, xmin=20, xmax=70, ymin=30, ymax=70, zmin=30, zmax=60):
    
    # Standardize b-values into 3 commonly used b-values
    bval_list = np.array([1000, 2000, 3000])
    bvals[np.abs(bvals)<100] = 0
    for b in bval_list:
    	bvals[np.abs(bvals-b)<100] = b
    
    # Separate DWI into b=0 image and b=1000, 2000, 3000 images.
    bval_indi = np.array([i for i in range(len(bvals)) if bvals[i] in bval_list])
    img_b0_all, img_data_all  = brain_image[..., bvals == 0], brain_image[..., bval_indi] 
    bvec, bval = bvecs[..., bval_indi], bvals[bval_indi]
    
    # Estimate b_factor and ratio for response function (kernel function)
    ## Set the ROI for pre-analysis (which covers the ROI analysis )
    x_pre, y_pre, z_pre = np.arange(xmin, xmax), np.arange(ymin, ymax), np.arange(zmin, zmax)
    img_b0_pre, img_data_pre = img_b0_all[np.ix_(x_pre, y_pre, z_pre)], img_data_all[np.ix_(x_pre, y_pre, z_pre)]

    ## Estimate S0 and SNR from b = 0 image
    img_b0_pre_indi = img_b0_pre.min(axis=3)>0
    S0_pre, sigma_pre = img_b0_pre.mean(axis=3)*img_b0_pre_indi, img_b0_pre.std(axis=3, ddof=1)*img_b0_pre_indi
    SNR = np.median(S0_pre[img_b0_pre_indi]/sigma_pre[img_b0_pre_indi])
    
    ## Fit the single tensor model to b=1000, 2000, 3000 images.
    ### Normalize the original image
    img_norm = (img_data_pre.T/S0_pre.T).T 
    img_norm[img_norm<=0] = np.min(img_norm[img_norm>0])
    img_norm[img_norm==float("inf")] = np.max(img_norm[img_norm<float("inf")])
    
    n_index = img_norm.shape[:-1]
    eval_pre = np.ones(n_index+(3,))  # store 3 eigenvalues of D matrix for each voxel
    index_list = list(np.ndindex(n_index))
    
    for k in index_list:
    	D_matrix = gauss_newton_dwi(bvec, bval, img_norm[k])
    	if not np.isnan(D_matrix).any():
    		eval_pre[k] = np.sort(np.linalg.eig(D_matrix)[0])
    
    FA_pre = np.sqrt(((eval_pre[..., 0]-eval_pre[..., 1])**2+(eval_pre[..., 1]-eval_pre[..., 2])**2
    	+(eval_pre[..., 2]-eval_pre[..., 0])**2)/(2*(eval_pre[..., 0]**2+eval_pre[..., 1]**2+eval_pre[..., 2]**2)))
    eval_ratio_pre = eval_pre[..., 2]*2/(eval_pre[..., 0]+eval_pre[..., 1])
    eval_pre_indi = (FA_pre<1) * (FA_pre>0.8) * ((eval_pre>0).all(axis=3)) * (eval_pre[..., 1]/eval_pre[..., 0]<1.5)
    
    eval_select_pre = list(zip(eval_pre[eval_pre_indi], eval_ratio_pre[eval_pre_indi]))
    eval_select_pre.sort(key = lambda x: x[1])
    eval_select_median_pre = eval_select_pre[len(eval_select_pre)//2]
    
    b_factor, ratio_response = eval_select_median_pre[0][2], eval_select_median_pre[1]
    
    return b_factor, ratio_response, SNR
#%%
# Input: the mask file of the ROI which can be chosen from the various ATLAS in FSL  
# Purpose: Extract the ROI information  (Location w.r.t each axis)
# Hwang et al.(2020) considered 7 subcortical regions chosen from Havard-Oxford subcortical atlas
# Considered Subcortical regions: Caudate, Accumbens, Amygdala, Hippocampus, Pallidum, Putamen, Thalamus
def ROI_info(mask_img):
    
    ROI_coord=np.where(mask_img>0)
    
    x_min, x_max = min(ROI_coord[0]), max(ROI_coord[0])
    x_size = int(x_max - x_min + 1)
    
    y_min, y_max = min(ROI_coord[1]), max(ROI_coord[1])
    y_size = int(y_max - y_min + 1)
    
    z_min, z_max = min(ROI_coord[2]), max(ROI_coord[2])
    z_size = int(z_max - z_min + 1)
    
    return ROI_coord, x_min, x_max, x_size, y_min, y_max, y_size, z_min, z_max, z_size
    #return {'coord':ROI_coord, 'x_min':x_min, 'x_max':x_max, 'x_num':x_num, 'y_min':y_min, 'y_max':y_max ,'y_num':y_num, 'z_min':z_min, 'z_max':z_max ,'z_num':z_num}    
#%%
# Input: b, bvecs, bvals
# Output: bvecs and image location in original image correspoding to the given 'b' value.
# Used for Data extraction step and Design matrix generation.
def bvecs_location(b, bvecs, bvals):
    
    img_loc = np.where(np.abs(bvals-b)<100)[0]
    return img_loc, bvecs[...,img_loc]
