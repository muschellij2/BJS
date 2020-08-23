##Seung Yong: 
###batch download hcp data and register dwi to a reference -- this is the first step in the real data analysis 
##Update 9/27/2019;
##Update 2/08/2020; added "ListBucketResult" in the path. (b/c AWS path is updated.)
##Update 7/29/2020; 

############################ Part I: get subject information #################################
### Find the subjects whose dMRI data is available on AMAZON database.
### If the data is not available on Amazon database, we cannot download manually on database.
library(neurohcp)
set_aws_api_key(access_key = "your_key", secret_key = "your_secret_key")
have_aws_key() #Check whether access_key and secret_key are valid

#On the website, 1027 subjects are available
#However, currently, 999 subjects data are available on Amazon database
subject_list=as.numeric(unique(hcp_1200_scanning_info$id)) 

# Find the subject whether dMRI data is availble.
# If dMRI is available, under the /T1w/Diffusion folder, there should be 11 files. (if not 6 files)
# 5 File lists: bvals, bvecs, data.nii.gz, grad_dev.nii.gz, nodif_brain_mask.nii.gz

size_info=c() # Store the data.nii.gz size information
subject_ids=c() # Store the subject id which has 11 files under the /T1w/Diffusion folder.

for(i in 1:length(subject_list)){
  
  dir_info=hcp_list_dirs(paste0("HCP/",subject_list[i],"/T1w/Diffusion"))
  
  #Contents key gives you the number of files under this folder.
  loc_contents=which(names(dir_info$parsed_result$ListBucketResult)=='Contents')
  
  #The following code is to record the maximum file size.
  if (length(loc_contents)>0){
    size_list=c()
    for (j in 1:length(loc_contents)){
      size_list=c(size_list, dir_info$parsed_result$ListBucketResult[[loc_contents[j]]]$Size)
    }
    size_info=c(size_info, max(as.numeric(unlist(size_list))))
    subject_ids=c(subject_ids, subject_list[i])
  }
  else
    size_info=c(size_info,NULL)
}

## Currently, dMRI of 435 subjects are abvailable in Amazon Database.  #length(subject_ids)
## Choose dMRI whose file size is greater than 1GB (1,000,000,000) - 90 gradient directions.
## 395 dMRI data satisfies the above criteria.

############################ Part II: download and registration through FSL: done on a server ################
## transfer the R code to shell for registration through FSL.
library(fslr) ##https://www.neuroconductor.org/package/fslr

source('R/fslr_fncts.R')  # Define the path of fslr_fncts.R

id = 100307  #Example subject ID 100307

#Specify the directory to download and process the image.
data_path = paste0('user_path', id, '/')

#Specify the directory of references (following directory is example if OS is Mac)
#Reference (MNI 2mm)
ref_brain_with_skull = "/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz"
ref_brain = "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"
ref_brain_mask = "/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil.nii.gz"
flirtconfig = '/usr/local/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf'

#The follwing code is for download and registration at the same time.

# Set the directories for downloading
hcp_indir=paste0("HCP/",id,"/T1w/Diffusion") ##DWI
hcp_infile=paste0("HCP/",id,"/T1w/T1w_acpc_dc_restore_1.25.nii.gz") ##T1
   
hcp_outdir = data_path
hcp_outfile = paste0(data_path, "T1w_acpc_dc_restore_1.25.nii.gz")
  
##download from AWS:
download_hcp_dir(hcp_indir, outdir=hcp_outdir, verbose=FALSE) ##  DWI
download_hcp_file(hcp_infile, destfile=hcp_outfile, verbose=FALSE) ## T1
  
##processing by fsl: 
# (i) extract brain by BET
bet_w_fslmaths(T1w = paste0(data_path, 'T1w_acpc_dc_restore_1.25.nii.gz'),
               mask = paste0(data_path, 'nodif_brain_mask.nii.gz'), 
               outfile = paste0(data_path,'T1w_acpc_dc_restore_1.25_brain.nii.gz')) 

# (ii) FAST segmentation (GM, WM, CSF)
fast(file = paste0(data_path,'T1w_acpc_dc_restore_1.25_brain.nii.gz'), 
     outfile = nii.stub(paste0(data_path,'T1w_acpc_dc_restore_1.25_brain.nii.gz')), 
     opts = '-N')
?fast
# (iii) Registration by FLIRT (Affine -- dof = 12); get the registration matrix.
flirt(infile = paste0(data_path,'T1w_acpc_dc_restore_1.25_brain.nii.gz'),  #Input: Brain extracted T1-weighted image
      reffile = ref_brain, #Reference brain image: MNI152_2mm_brain
      omat = paste0(data_path,'org2std.mat'),  #Registration Matrix: Need to FNIRT.
      dof = 12,
      outfile = paste0(data_path,'T1w_acpc_dc_restore_1.25_brain_flirt12.nii.gz'))
  
# (iv) Registration by FNIRT 
opt_fnirt=paste0(' --aff=',data_path,'org2std.mat',  #affine transformation matrix from the previous FLIRT
                 ' --config=',flirtconfig,
                 ' --cout=', data_path, 'org2std_coef.nii.gz',  #the spline coefficient and a coopy of the affine transform 
                 ' --fout=', data_path, 'org2std_warp.nii.gz')  #actual warp-field in the x,y,z directions/.

fnirt(infile = paste0(data_path,'T1w_acpc_dc_restore_1.25.nii.gz'),  #Input: Original T1-weighted image
      reffile = ref_brain_with_skull, #Reference brain image: MNI152_2mm
      outfile = paste0(data_path,'T1w_acpc_dc_restore_1.25_fnirt.nii.gz'),
      opts = opt_fnirt)


# (v) Inverse warp-field
invwarp(reffile=paste0(data_path,'T1w_acpc_dc_restore_1.25.nii.gz'), #Reference: Original T1-weighted image
        infile=paste0(data_path,'org2std_warp.nii.gz'), #warp-field file from FNIRT
        outfile=paste0(data_path,'std2org_warp.nii.gz')) #inverse of warp-field


# (vi) Apply transformation  any masks on MNI152 to the native space.
# (You should make mask_file)
fsl_applywarp(infile = paste0(data_path,'mask_file.nii.gz'), #mask on MNI152
              reffile = paste0(data_path,'T1w_acpc_dc_restore_1.25.nii.gz'), #Reference: Original T1-weighted image
              outfile = paste0(data_path,'mask_file_org.nii.gz'), #mask on native space
              warpfile = paste0(data_path,'std2org_warp.nii.gz')) #inverse warp-field file from FNIRT
