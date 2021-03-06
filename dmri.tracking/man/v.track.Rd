% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dmri.tracking.R
\name{v.track}
\alias{v.track}
\title{Apply Tracking Algorithm}
\usage{
v.track(
  v.obj,
  xgrid.sp,
  ygrid.sp,
  zgrid.sp,
  braingrid,
  max.line = 100,
  nproj = 1,
  elim = T,
  elim.thres = 1,
  thres.ang = 0.5235988,
  vorient = rep(1, 1, 1)
)
}
\arguments{
\item{v.obj}{vobj}

\item{xgrid.sp}{xgird.sp}

\item{ygrid.sp}{ygrid.sp}

\item{zgrid.sp}{zgrid.sp}

\item{braingrid}{braingrid}

\item{max.line}{max.line}

\item{nproj}{nprog}

\item{elim}{elim}

\item{elim.thres}{elim.thres}

\item{thres.ang}{thres.ang}

\item{vorient}{vorient}
}
\value{
object for tractography
}
\description{
v.track function will track the fiber based on the obtained peaks in each voxel.
}
