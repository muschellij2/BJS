###HCP data analysis: Step 2 (cont'd):  tractography
#take the peak detection results as input

##need to first install Rpackage: dmri.tracking_1.01.tar.gz in fold /dmri.tracking-r

##load packages
library(R.matlab)
library(dmri.tracking)
library(rgl)

data_path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/data/'
##load peak detection results from "example_HCP.py"
file_name= paste0(data_path,"peak.mat")
peak_result= readMat(file_name) #read matlab data into R

##format the peak detection results for the tracking algorithm
form_tractography<-function(result){
  temp<-NULL
  temp$braingrid<-result$braingrid
  temp$loc<-result$loc
  temp$map<-c(result$map)
  temp$n.fiber<-c(result$n.fiber)
  temp$n.fiber2<-c(result$n.fiber2)
  temp$nn1<-c(result$nn1)
  temp$nn2<-c(result$nn2)
  temp$nn3<-c(result$nn3)
  temp$rmap<-c(result$rmap)
  temp$vec<-result$vec
  temp$xgrid.sp<-c(result$xgrid.sp)
  temp$ygrid.sp<-c(result$ygrid.sp)
  temp$zgrid.sp<-c(result$zgrid.sp)
  
  return(temp)
}

temp = form_tractography(peak_result)
## specify region to draw the tracking results
x_subr = 1:temp$nn1
y_subr = 1:temp$nn2
z_subr = 1:temp$nn3

## Apply Tracking Algorithm  
# It takaes 20 min on 2020 mac 16inch
# If you want to skip this step, you can load the attached Rdata file (tracts_SLF_L.Rdata)

nproj = 1  ## skip nproj voxles before termination
tracts <- v.track(v.obj=temp, xgrid.sp=temp$xgrid.sp, ygrid.sp=temp$ygrid.sp,
                     zgrid.sp=temp$zgrid.sp, braingrid=array(temp$braingrid,dim=c(3,length(x_subr),length(y_subr),length(z_subr))), elim=T, nproj=nproj,
                     vorient=c(1,1,1), elim.thres=10, max.line=500) 
# elim.tresh:  return indices of tracks of at least elim.thres length: use this information for quicker plotting

save(tracts, file=paste0(data_path,'tracts_SLF_L.Rdata'))
#load(paste0(data_path,'tracts_SLF_L.Rdata'))

###Streamline Selection based on predefined streamline selection masks

rmap = temp$rmap

seed = rmap[as.vector(peak_result$seed)]
target = rmap[as.vector(peak_result$target)]

iind_store = c()
for(iind in (tracts$sorted.iinds[tracts$sorted.update.ind])){
  cond1 = (sum(tracts$tracks1[[iind]]$iinds %in% seed) + sum(tracts$tracks2[[iind]]$iinds %in% seed)>0)
  cond2 = (sum(tracts$tracks1[[iind]]$iinds %in% target) + sum(tracts$tracks2[[iind]]$iinds %in% target)>0)
  
  if (cond1*cond2 > 0){
    print(iind)
    iind_store<-c(iind_store, iind)
  }
}


## plot the tractography results (the selected streamlines) 
open3d()
for (iind in iind_store){
  cat(iind,"\n")
  # plot
  tractography(tracts$tracks1[[iind]]$inloc, tracts$tracks1[[iind]]$dir)
  tractography(tracts$tracks2[[iind]]$inloc, tracts$tracks2[[iind]]$dir)
}
par3d(windowRect = c(0, 0, 700, 700))
load(paste0(data_path,'view_left_slf.Rdata'))
rgl.viewpoint(scale=c(1,1,1),zoom=0.7,userMatrix = view_M)
rgl.snapshot(paste0(data_path,'slf_l'), fmt='png')

## Feature extraction (e.g. number of fiber longer than 10mm)
length(iind_store) # number of streamlines
summary(tracts$lens[iind_store])  # Summary of tracts lengths for the selected streamlines
