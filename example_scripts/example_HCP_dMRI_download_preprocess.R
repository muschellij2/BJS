##Seung Yong: 
###batch download hcp data and register dwi to a reference -- this is the first step in the real data analysis 
##Update 9/27/2019;
##Update 2/08/2020; added "ListBucketResult" in the path. (b/c AWS path is updated.)

#### loading parallel loop packages 
#library(foreach)
#library(doParallel)
#library(parallel)
#numcores=detectCores()
#makeCluster()

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
#final_subjects=subject_ids[which(size_info>1000000000)]
#length(final_subjects)

write.csv(final_subjects, "~/Desktop/final_subject.csv")

############################ Part II: download and registration through FSL: done on a server ################
## transfer the R code to shell for registration through FSL.
library(fslr) ##https://www.neuroconductor.org/package/fslr
id = final_subjects

#The follwing code is for download and registration at the same time.

for (i in 1:length(id)){
  print(id[i])
  hcp_indir=paste0("HCP/",id[i],"/T1w/Diffusion") ##DWI
  hcp_infile=paste0("HCP/",id[i],"/T1w/T1w_acpc_dc_restore_1.25.nii.gz") ##T1
   
  hcp_outdir=paste0("/scratch/syhwang/hcp/",id[i])
  hcp_outfile=paste0("/scratch/syhwang/hcp/",id[i],"/T1w.nii.gz")
  
  ##download from AWS:
  download_hcp_dir(hcp_indir, outdir=hcp_outdir, verbose=FALSE) ##  DWI
  download_hcp_file(hcp_infile, destfile=hcp_outfile, verbose=FALSE) ## T1
  
  ##processing by fsl: 
  # (i) extract brain by BET
  fsl_bet_infile1=paste0("/scratch/syhwang/hcp/",id[i],"/T1w.nii.gz")  ##T1
  fsl_bet_outfile1=paste0("/scratch/syhwang/hcp/",id[i],"/T1w_brain.nii.gz")
  
  fsl_bet_infile2=paste0("/scratch/syhwang/hcp/",id[i],"/data.nii.gz") ##DWI
  fsl_bet_outfile2=paste0("/scratch/syhwang/hcp/",id[i],"/data_brain.nii.gz")
  
  fsl_bet(infile=fsl_bet_infile1, outfile=fsl_bet_outfile1) #BET extracting brain 
  fsl_bet(infile=fsl_bet_infile2, outfile=fsl_bet_outfile2)
  
  
  #(ii) registration by FLIRT
  flirt_infile1=paste0("/scratch/syhwang/hcp/",id[i],"/T1w_brain.nii.gz") ##input brain: extracted T1 brain
  flirt_outfile1=paste0("/scratch/syhwang/hcp/",id[i],"/T1w_brain_flirt.nii.gz") ## output brain 
  flirt_omat=paste0("/scratch/syhwang/hcp/",id[i],"/T1w_flirt.mat") ## registration matrix: needed to register the DTI brain
  
  
  #(a)FLIRT AFFINE(dof=12) registration based on T1:  get the registration matrix 
  flirt(infile=flirt_infile1, outfile=flirt_outfile1, omat=flirt_omat,
        dof=12, reffile="/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz") ## the last argument gives the reference brain 
  
  #(b) register the DWI brain using the registration matrix from (a)
  flirtap_infile1=paste0("/scratch/syhwang/hcp/",id[i],"/data_brain.nii.gz")
  flirtap_outfile1=paste0("/scratch/syhwang/hcp/",id[i],"/data_brain_flirt.nii.gz")
  
  flirt_apply(infile=flirtap_infile1, outfile=flirtap_outfile1,
              reffile="/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz",
              initmat=flirt_omat) 
  print(i)
}
