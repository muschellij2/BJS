import numpy as np
from scipy.sparse.csgraph import connected_components
import tqdm
#For anaconda user : conda install -c anaconda progressbar2
#For Python3 user : sudo pip3 install progressbar2
#%%
# detect number of peaks and their positions in FOD
# idx: indices of nearest neighboring grid points for each grid point
# nbhd: consider only the top nbhd nearest neighboring grid points to identify local maximal peaks
# thresh: eliminate local peaks lower than thresh * highest peak
# degree: merge local peaks within degree together (use mean position as final position)
# peak_cut: largest number of detected peaks (with top peak values)
#return: num_comp: number of peaks; peak_pos_final: coordinates of the peaks 
def FOD_Peak(fod, idx, nbhd, thresh, degree, pos, sampling_index, return_peak_pos, peak_cut = float('inf')):

	available_index = np.ones(len(fod))
	peak_idx = []
	for i in sampling_index:
		if available_index[i] == 1:
			nbidx = idx[i, :nbhd]
			low_nbidx = nbidx[fod[nbidx] < fod[i]]
			available_index[low_nbidx] = 0
			if len(low_nbidx) == nbhd-1:
				peak_idx.append(i)
	peak_idx = np.array(peak_idx)

	if len(peak_idx) == 0:
		if return_peak_pos:
			return 0, np.array([[]]*3)
		else:
			return 0

	peak_idx = peak_idx[fod[peak_idx] > thresh*np.max(fod)]
	if len(peak_idx) > peak_cut:
		peak_idx = peak_idx[np.argsort(-fod[peak_idx])[:peak_cut]]
	peak_pos = pos[:, peak_idx]
	peak_value = fod[peak_idx]

	peak_dis = peak_pos.T.dot(peak_pos)
	peak_comp = (peak_dis > np.cos(degree*np.pi/180)) | (peak_dis < -np.cos(degree*np.pi/180))

	num_comp, idx_comp = connected_components(peak_comp)
	if return_peak_pos:
		peak_pos_final = np.zeros((3, num_comp))
		for i in range(num_comp):
			peak_pos_tmp = peak_pos[:, idx_comp==i].dot(peak_value[idx_comp==i])
			peak_pos_final[:, i] = peak_pos_tmp/np.linalg.norm(peak_pos_tmp)
		return num_comp, peak_pos_final
	else:
		return num_comp

#%% Peak Detection
def peak_detection(fod_est, idx, nbhd, thresh, degree, pos, sampling_index, peak_cut):
    xgrid_sp, ygrid_sp, zgrid_sp = 2, 2, 2 #After registeration
    n1, n2, n3, _ = fod_est.shape
    braingrid = np.zeros((3, n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                braingrid[:, i, j, k] = [(i-0.5*(n1-1))*xgrid_sp, (j-0.5*(n2-1))*ygrid_sp, (k-0.5*(n3-1))*zgrid_sp]
    n_fiber, rmap = np.zeros(n1*n2*n3), np.zeros(n1*n2*n3)
    vec, loc, map = np.array([[]]*3).T, np.array([[]]*3).T, np.array([])
    for k in tqdm.tqdm(range(n3),desc="Peak Detecting:"):
        for j in range(n2):
            for i in range(n1):
                ind = k*n1*n2+j*n1+i
                n_fiber[ind], peak_pos = FOD_Peak(fod_est[i, j, k], idx, nbhd, thresh, degree, pos, sampling_index, True, peak_cut)
                if n_fiber[ind] > 0:
                    vec = np.vstack((vec, peak_pos.T))
                    loc = np.vstack((loc, np.tile(braingrid[:, i, j, k], (int(n_fiber[ind]), 1))))
                    map = np.concatenate((map, [ind+1]*int(n_fiber[ind])))
                else:
                    vec = np.vstack((vec, [np.nan]*3))
                    loc = np.vstack((loc, braingrid[:, i, j, k]))
                    map = np.concatenate((map, [ind+1]))
    n_fiber2 = np.array([])
    for i in range(n1*n2*n3):
        rmap[i] = np.where(map == (i+1))[0][0]+1
        n_fiber2 = np.concatenate((n_fiber2, [n_fiber[i]]*max(int(n_fiber[i]), 1)))
    result={'braingrid':braingrid,'loc':loc,'map':map,'n.fiber':n_fiber,'n.fiber2':n_fiber2,\
            'nn1':n1,'nn2':n2,'nn3':n3,'rmap':rmap,'vec':vec,'xgrid.sp':xgrid_sp,'ygrid.sp':ygrid_sp,'zgrid.sp':zgrid_sp}
    return result




