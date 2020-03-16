clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set path
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
module_path='/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/matlab/fod_plotting';
data_path='/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/data/';

%%% add path of folders (set of fod plotting functions)
addpath(module_path);
addpath(data_path);
%% Set the options for plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.base_mesh = 'ico';
options.relaxation = 1;
options.keep_subdivision = 1;

%%%plotting options 
options.spherical = 1;

% options for fitted FOD display
options.use_color = 1;
options.color = 'wavelets';
options.use_elevation = 2;
options.rho = 0.5;
options.scaling = 1.5;

% options for true fiber
plot_rho = 0.5;%options.rho;
plot_scale = 1;%options.scaling;

% load pre-generated vertex and face (J=5)
load("vertex_face_plotting.mat");

%% Load Estimated FOD
est_fod = load("est_result.mat");

%% Draw Figures
figure 
subplot(1,4,1)
true =squeeze(est_fod.true);
true(true<1e-07)=1e-07;
plot_spherical_function(v_plot, f_plot, true ,options);
hold on;
draw_fiber(est_fod.theta,est_fod.phi,plot_scale,plot_rho*max(true));
title("True")
view([-1 0 0])
camlight('left')

subplot(1,4,2)
bjs_mean = squeeze(est_fod.BJS_mean);
bjs_mean(bjs_mean<1e-07)=1e-07;
bjs_m2sd = squeeze(est_fod.BJS_m2sd);
bjs_m2sd(bjs_m2sd<1e-07)=1e-07;
hold all;
plot_spherical_function(v_plot,f_plot,est_fod.BJS_m2sd,options);
alpha(0.25)
hold on;
plot_spherical_function(v_plot,f_plot,est_fod.BJS_mean,options);
draw_fiber(est_fod.theta,est_fod.phi,plot_scale,plot_rho*max(bjs_m2sd));
title("BJS")
view([-1 0 0])
camlight('left')

subplot(1,4,3)
scsd_mean = squeeze(est_fod.SCSD_mean);
scsd_mean(scsd_mean<1e-07)=1e-07;
scsd_m2sd = squeeze(est_fod.SCSD_m2sd);
scsd_m2sd(scsd_m2sd<1e-07)=1e-07;
hold all;
plot_spherical_function(v_plot,f_plot,est_fod.SCSD_m2sd,options);
alpha(0.25)
hold on;
plot_spherical_function(v_plot,f_plot,est_fod.SCSD_mean,options);
draw_fiber(est_fod.theta,est_fod.phi,plot_scale,plot_rho*max(scsd_m2sd));
title("SCSD")
view([-1 0 0])
camlight('left')

subplot(1,4,4)
shridge_mean = squeeze(est_fod.SHridge_mean);
shridge_mean(shridge_mean<1e-07)=1e-07;
shridge_m2sd = squeeze(est_fod.SHridge_m2sd);
shridge_m2sd(shridge_m2sd<1e-07)=1e-07;
hold all;
plot_spherical_function(v_plot,f_plot,est_fod.SHridge_m2sd,options);
alpha(0.25)
hold on;
plot_spherical_function(v_plot,f_plot,est_fod.SHridge_mean,options);
draw_fiber(est_fod.theta,est_fod.phi,plot_scale,plot_rho*max(shridge_m2sd));
title("SHridge")
view([-1 0 0])
camlight('left')
