#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 15:44:05 2018

@author: Seungyong, 1/30/2020
"""
#%%import built-in package
import numpy as np


#%% BJS
# Description: FOD estimation method proposed by Hwang et al. 2020 
#Input: Y: DWI signals
#       SH: Spherical Harmonic Matrix evaluated at the sampled gradient direction grid (half sphere)
#       SHD: Spherical Harmonic Matrix evaluated at the dense evaluation grid (half sphere)
#       SHP: Spherical Harmonic Matrix evalueate at the dense evaluation grid (entire sphere)
#       R: Response matrix
#       mu1, mu2, muinf: eigenvalue values from the block covariance matrices in BJS definition
#       lmax: same value as what have used in 'cal_mu'
#       t: the coefficient in the threshold formula t*(2 log (2l+1))
#       k: the SH level  for thresholding: for level <=2*k, no thresholding;
#       (e.g. k=0, thresholding is not applied to the blocks which are correspoding to l=0 )
#       (e.g. k=1, thresholding is not applied to the blocks which are correspoding to l=0,2 )
#       (e.g. k=2, thresholding is not applied to the blocks which are correspoding to l=0,2,4 )
#Output: estimated FOD 
def BJS(Y, SH, SHD, SHP, R, mu1, mu2, muinf, lmax, tau=0, t=1, k=2, SH_plot=None,plot=False):
    
    L = int((lmax+1)*(lmax+2)/2)
    
    SH_init = SH[:,:L]
    R_init = R[:L,:L]
    X_init = SH_init.dot(R_init)

    sigma2=np.sum((Y-SH_init.dot(np.linalg.inv(SH_init.T.dot(SH_init))).dot(SH_init.T).dot(Y))**2)/(len(Y)-np.linalg.matrix_rank(SH_init))  

    bar_R = precond_R(R_init,1)
    Z = bar_R.dot(np.linalg.inv(SH_init.T.dot(SH_init))).dot(SH_init.T).dot(Y)
    V = bar_R.dot(np.linalg.inv(SH_init.T.dot(SH_init))).dot(bar_R)
    
    ind_block=indicator_block(lmax)

    init_ests=[]

    for i in range(0,int(lmax/2+1)):
        
        block_loc = np.where(ind_block==i)[0]
        Zk = Z[block_loc]
        
        if (i<=k):
            init_ests.extend(Zk)
        else:
            tk = t*2*np.log(4*i+1)
            err_prop = sigma2*(mu1[i] + 2*np.sqrt(mu2[i]*tk) + 2*tk*muinf[i]) / np.sum(Zk**2) 
            shrink_eq = (1-err_prop)
            init_ests.extend(shrink_eq*(shrink_eq>0)*Zk)
    
    theta = np.array(init_ests)
    nobar_R = precond_R(R_init,0)
    fod_init = SHD[:,:L].dot(nobar_R.dot(theta))
       
    SHD_update = SHD[np.where(fod_init<0)[0],:]
    
    X_update = np.vstack((SH.dot(R),SHD_update))
    Y_update = np.concatenate((Y,np.zeros(SHD_update.shape[0])))
    
    coeff = np.linalg.inv(X_update.T.dot(X_update)).dot(X_update.T).dot(Y_update)
    fod_update = SHP.dot(coeff)

    if plot:
        return fod_stand(fod_update), fod_stand((SH_plot.dot(coeff)))
    else:
        return fod_stand(fod_update)

#%% Description: FOD estimation method SHRidge motivated by Descoteaux et al.(2006)
#Input: Y: DWI signal
#       SH: Spherical Harmonic Matrix evaluated at the sampled grid (half sphere)
#       SHP: Spherical Harmonic Matrix evalueate at the dense grid (entire sphere)
#       R: Response matrix
#       P: Penalty matrix
#       lam: a set of tunning parameter value
#Output: estimated FOD   
def SH_ridge(Y, SH, SHP, R, P, lmax, lam):
    
    n=len(Y)
    
    L = int((lmax+1)*(lmax+2)/2)
    SH = SH[:,:L]
    SHP = SHP[:,:L]
    R = R[:L, :L]
    
    #store estimators with each lambda value
    beta_ests_all=np.zeros(shape=(R.shape[1],len(lam)))
    dwi_ests_all=np.zeros(shape=(n,len(lam)))
    
    df_all=np.zeros(len(lam))
    RSS_all=np.zeros(len(lam))
    BIC_all=np.zeros(len(lam))
    
    X=SH.dot(R)
    
    for i in range(0,len(lam)):
        temp=np.linalg.inv(X.T.dot(X)+lam[i]*P).dot(X.T) 
        beta_ests_all[:,i]=temp.dot(Y)
        dwi_ests_all[:,i]=X.dot(beta_ests_all[:,i])
        df_all[i]=np.sum(np.diag(X.dot(temp)))
        RSS_all[i]=np.sum((Y-dwi_ests_all[:,i])**2)
        BIC_all[i]=n*np.log(RSS_all[i]/n)+df_all[i]*np.log(n)
        
    beta_ests=beta_ests_all[:,np.argmin(BIC_all)] #choose the coefficients which produce the lowest BIC value
    fod_est=SHP.dot(beta_ests) #Update the negative values of the estimated FOD to zero.
    
    return beta_ests, fod_stand(fod_est)  


#%% Description: FOD estimation method proposed by Tournier et al.(2010)
#Input: Y: DWI signal
#       SH: Spherical Harmonic Matrix evaluated at the sampled grid (half sphere)
#       SHD: Spherical Harmonic Matrix evaluated at the dense grid (half sphere)
#       SHP: Spherical Harmonic Matrix evalueate at the dense grid (entire sphere)
#       R: Response matrix
#       init_est: initial estimator of FOD coefficienf 
#Output: estimated FOD 
def superCSD(Y, SH, SHD, SHP, R, init_est = None):
    
    if init_est is None:
  
        X_init=SH[:,:15].dot(R[:15,:15])
        init_est = np.linalg.inv(X_init.T.dot(X_init)).dot(X_init.T).dot(Y)
    
    X = SH.dot(R)
    
    fod_csd_old = SHD[:,:len(init_est)].dot(init_est)
    tao=0.1*np.mean(fod_csd_old)
    thresh_cur=99
    
    count=0
    
    while thresh_cur>1e-04 and count<50:
        SHD_update=SHD[(fod_csd_old<tao),:]
        
        M = np.vstack((X,SHD_update))
        B = np.zeros(M.shape[0])
        B[:len(Y)] = Y
        if np.linalg.matrix_rank(M) < M.shape[1]:
            f_old_csd = np.linalg.inv(M.T.dot(M) + 1e-04*np.identity(M.shape[1])).dot(M.T).dot(B)
        else:
            f_old_csd = np.linalg.inv(M.T.dot(M)).dot(M.T).dot(B)
        
        fod_csd_cur = SHD.dot(f_old_csd)
        thresh_cur = max(abs(fod_csd_cur - fod_csd_old))
        fod_csd_old = fod_csd_cur
        count = count + 1
    
    fod_est_csd = SHP.dot(f_old_csd)
    
    return fod_stand(fod_est_csd)




#%% Auxiliary functions

# BJS: Separate design matrix into blocks based on the level of basis.
def indicator_block(lmax):
    
    indi_block=[]
    
    for i in range(0,int(lmax/2+1)):
        indi_block.extend(np.repeat(i,4*i+1))

    indi_block=np.array(indi_block)

    return(indi_block)

# BJS: Preconditioning convolution matrix R
def precond_R(R,bar,alpha=-0.5):
    '''
    R: Convolution kernel matrix
    bar: indicator of bar (0: w/o bar, 1: bar)
    '''
    r=np.diag(R)
    
    if (bar==0):
        result=np.sign(r)*(np.abs(r)**(alpha))
    
    elif (bar==1):
        result=(np.abs(r)**(alpha))    

    return np.diag(result)

# BJS: calculate eigenvalues for each block
def cal_mu(SH, R, lmax):
    
    #X = SH.dot(R)    
    #V = np.linalg.inv(X.T.dot(X))

    bar_R = precond_R(R,1)
    V = bar_R.dot(np.linalg.inv(SH.T.dot(SH))).dot(bar_R)

    k_indi=indicator_block(lmax)
    
    mu1=[]
    mu2=[]
    muinf=[]

    for i in range(0,int(lmax/2+1)):
        block_loc=np.where(k_indi==i)[0] 
        V_block=V[block_loc,:][:,block_loc]

        U_b, Lam_b, V_b = np.linalg.svd(V_block)
        
        mu1.append(np.sum(Lam_b))
        mu2.append(np.sum(Lam_b**2))
        muinf.append(max(Lam_b))
        
    return mu1, mu2, muinf

#SHRidge: Description: construct penalty matrix for Laplace-Beltrami Regularization 
#Input: lmax (maximum level of spherical harmonic)
#Output: penalty matrix
def penalty_mat(lmax):

    l = np.linspace(0,int(lmax/2),int(lmax/2)+1)*2
    
    penalty=[]
    
    for i in range(0, len(l)):
        penalty.extend(np.repeat((l[i]**2)*((l[i]+1)**2), 2*l[i]+1))
    
    return np.diag(penalty)

#Description: standardize FOD (remove negative value and make the sum to 1)
def fod_stand(fod):
	fod_st = fod.copy()
	fod_st[fod_st < 0] = 0
	fod_st /= np.sum(fod_st)
	return fod_st 


