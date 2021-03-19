% code for running bath analysis of resting state data in SPM12

clear all

root_dir = pwd;
SPM_PATH = fullfile(root_dir, 'spm12');
addpath(SPM_PATH)

%% Initialize SPM

spm('Defaults','fMRI');
spm_jobman('initcfg');

%spm('CreateIntWin','on');
%spm_figure('Create','Graphics','Graphics','on');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% definition according to the specific data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

nd=5; % no of dummy scans
tr = 0.720; % in seconds
% for smoothing
%fwhm=[4 4 4]; % the thumb of rules says fwhm should be twice the voxel dimension


% Get a list of all files and folders in func folder.
files = dir('func');
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
subFolders(ismember( {subFolders.name}, {'.', '..'})) = [];  %remove . and ..
for k = 1 : length(subFolders)
subNames{1,k}=subFolders(k).name;
end
%subNames= {'101309','102109','102715','105923','107422','109325','110007','110411','110613','111312','111514','112112','112819','114217','114419','115017','115724','117324','117930','118023'};

clear k subFolders 
for sI = 1: length(subNames)

    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% directories 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
% define directories    

func_dir = fullfile(root_dir,'func', subNames{sI});


% file select
f_or = spm_select('FPList',func_dir,'^rfMRI.*\.nii$'); % original functional images


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% create directory GLM & GLM2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
GLM_dir=fullfile(root_dir,'GLM');
mkdir(GLM_dir)
glm_dir=fullfile(GLM_dir, subNames{sI});
mkdir(glm_dir)

GLM2_dir=fullfile(root_dir,'GLM2');
mkdir(GLM2_dir)
glm2_dir=fullfile(GLM2_dir, subNames{sI});
mkdir(glm2_dir)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONVERT FUNCTIONAL SCANS FROM 4D TO 3D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spm_file_split(f_or);



clear matlabbatch

matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(func_dir);
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'original';

matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_move.files = cellstr(f_or); % remove the original functional file
matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_move.action.moveto = cellstr(fullfile(func_dir,'original'));

spm_jobman('run',matlabbatch);

f = spm_select('FPList',func_dir,'^rfMRI.*\.nii$'); % functional images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DUMMY SCANS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear matlabbatch

matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = cellstr(func_dir);
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'dummy';

matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_move.files = cellstr(f(1:nd,:)); % remove first 5 scans to allow for magnetization to be stable
matlabbatch{2}.cfg_basicio.file_dir.file_ops.file_move.action.moveto = cellstr(fullfile(func_dir,'dummy'));

spm_jobman('run',matlabbatch);


f = spm_select('FPList',func_dir,'^rfMRI.*\.nii$'); % functional images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% creating friston24 regressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rp = spm_select('FPList',func_dir,'^Movement.*\.txt$'); % 6 head motion parameters and 6 temporal derivatives
rp=readmatrix(rp);
rp=rp(:,1:6);%select only 6 motion parameters
[r,c]=size(rp);
rp1=vertcat(zeros(1,6),rp); % 6 head motion parameters from the previous time point
rp1(r+1,:)=[];
sq=rp.^2; % squared head motion parameters

clear matlabbatch

% %% Smoothing
% matlabbatch{1}.spm.spatial.smooth.data =cellstr(f);
% matlabbatch{1}.spm.spatial.smooth.fwhm = fwhm; 
% matlabbatch{1}.spm.spatial.smooth.dtype = 0; 
% matlabbatch{1}.spm.spatial.smooth.im = 0; 
% matlabbatch{1}.spm.spatial.smooth.prefix = 's'; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLM SPECIFICATION, ESTIMATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Model specification
matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(glm_dir);
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans'; 
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = tr; 
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16; 
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8; 
%matlabbatch{2}.spm.stats.fmri_spec.sess.scans = cfg_dep('Smooth: Smoothed Images', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files')); 
matlabbatch{1}.spm.stats.fmri_spec.sess.scans= cellstr(f);
matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {}); 
matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''}; 
matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {}); 
matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''}; 
matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128; 
matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {}); 
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0]; 
matlabbatch{1}.spm.stats.fmri_spec.volt = 1; 
matlabbatch{1}.spm.stats.fmri_spec.global = 'None'; 
matlabbatch{1}.spm.stats.fmri_spec.mthresh = -Inf; 
matlabbatch{1}.spm.stats.fmri_spec.mask = {''}; 
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)'; 


%% Estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(fullfile(glm_dir,'SPM.mat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0; 
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1; 
spm_jobman('run', matlabbatch)
%%

% file select
mask = spm_select('FPList',glm_dir,'^mask.*\.nii$'); % whole brain mask


clear matlabbatch


%% Creation of  white matter and CSF regressor
matlabbatch{1}.spm.util.voi.spmmat = cellstr(fullfile(glm_dir,'SPM.mat'));
matlabbatch{1}.spm.util.voi.adjust = NaN; 
matlabbatch{1}.spm.util.voi.session = 1; 
matlabbatch{1}.spm.util.voi.name = 'WM'; 
matlabbatch{1}.spm.util.voi.roi{1}.sphere.centre = [0 -24 -33]; 
matlabbatch{1}.spm.util.voi.roi{1}.sphere.radius = 6; 
matlabbatch{1}.spm.util.voi.roi{1}.sphere.move.fixed = 1; 
matlabbatch{1}.spm.util.voi.roi{2}.mask.image(1) = cellstr(mask);
matlabbatch{1}.spm.util.voi.roi{2}.mask.threshold = 0.5; 
matlabbatch{1}.spm.util.voi.expression = 'i1&i2'; 
spm_jobman('run',matlabbatch);

wm=Y;

clear matlabbatch


matlabbatch{1}.spm.util.voi.spmmat = cellstr(fullfile(glm_dir,'SPM.mat'));
matlabbatch{1}.spm.util.voi.adjust = NaN; 
matlabbatch{1}.spm.util.voi.session = 1; 
matlabbatch{1}.spm.util.voi.name = 'CSF'; 
matlabbatch{1}.spm.util.voi.roi{1}.sphere.centre = [0 -40 -5]; 
matlabbatch{1}.spm.util.voi.roi{1}.sphere.radius = 6; 
matlabbatch{1}.spm.util.voi.roi{1}.sphere.move.fixed = 1; 
matlabbatch{1}.spm.util.voi.roi{2}.mask.image(1) = cellstr(mask); 
matlabbatch{1}.spm.util.voi.roi{2}.mask.threshold = 0.5; 
matlabbatch{1}.spm.util.voi.expression = 'i1&i2';
spm_jobman('run', matlabbatch)


csf=Y;

clear matlabbatch


nuis_reg= horzcat(rp(6:end,:),rp1(6:end,:),sq(6:end,:),wm,csf); % friston regressor plus time series from wm and csf as nuissance regressor
save(fullfile(glm_dir,'nuis_reg.txt' ), 'nuis_reg','-ascii');

% file select
%smooth= spm_select('FPList',func_dir,'^s.*\.nii$'); % select smoothed files

%% Model specification
matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(glm2_dir);
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans'; 
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = tr; 
%matlabbatch{1}.spm.stats.fmri_spec.sess.scans=cellstr(smooth);
matlabbatch{1}.spm.stats.fmri_spec.sess.scans=cellstr(f);
matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {}); 
matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''}; 
matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {}); 
matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {fullfile(glm_dir,'nuis_reg.txt' )};
matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128; 
matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {}); 
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0]; 
matlabbatch{1}.spm.stats.fmri_spec.volt = 1; 
matlabbatch{1}.spm.stats.fmri_spec.global = 'None'; 
matlabbatch{1}.spm.stats.fmri_spec.mthresh = -Inf; 
matlabbatch{1}.spm.stats.fmri_spec.mask = {''}; 
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)'; 


%% Estimation
matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr(fullfile(glm2_dir,'SPM.mat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0; 
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1; 
%% 


spm_jobman('run', matlabbatch)


end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Spectral DCM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% extract voi


maskNames={'lFp1.nii','lSSC.nii','lV1.nii','lA1.nii','lSMA.nii','lMC.nii','laIns.nii','lpIns.nii','lThal.nii','lBroca.nii','rFp1.nii','rSSC.nii','rV1.nii','rA1.nii','rSMA.nii','rMC.nii','raIns.nii','rpIns.nii'};
voiNames={'lFp1','lSSC','lV1','lA1','lSMA','lMC','laIns','lpIns','lThal','lBroca','rFp1','rSSC','rV1','rA1','rSMA','rMC','raIns','rpIns'};



for sI = 1: length(subNames)

spm_dir=fullfile(root_dir,'GLM2', subNames{sI},'SPM.mat');
brain_mask=fullfile(root_dir,'GLM', subNames{sI},'mask.nii');

for vI = 1: length(voiNames)

mask_dir= fullfile (root_dir,'masks', maskNames{vI})
 
clear matlabbatch;
matlabbatch{1}.spm.util.voi.spmmat = cellstr(spm_dir); % directory to spm.mat file
matlabbatch{1}.spm.util.voi.adjust = NaN;
matlabbatch{1}.spm.util.voi.session = 1; % Session index
matlabbatch{1}.spm.util.voi.name = voiNames{vI}; % name you want to give 
matlabbatch{1}.spm.util.voi.roi{1}.mask.image = cellstr(mask_dir); % directory to mask image
%matlabbatch{1}.spm.util.voi.roi{1}.mask.threshold = 1;
%matlabbatch{1}.spm.util.voi.roi{1}.spm.mask.mtype = 0; % inclusion

matlabbatch{1}.spm.util.voi.roi{2}.mask.image = cellstr(brain_mask); % directory to brain mask

matlabbatch{1}.spm.util.voi.expression = 'i1 & i2';
spm_jobman('run',matlabbatch);



end


end




%% define DCMs

voiNamesL={'VOI_lSMA_1.mat', 'VOI_lMC_1.mat'};
voiNamesR={'VOI_rSMA_1.mat', 'VOI_rMC_1.mat'};   


for sI = 1: length(subNames)

cd(fullfile(root_dir,'GLM2', subNames{sI}));

model_name = 'L_MOT';

xY         = voiNamesL;

SPM        = 'SPM.mat';

n   = 2;    % number of regions

nu  = 1;    % number of inputs. For DCM for CSD we have one input: null

TR  = 0.72;    % volume repetition time (seconds)

TE  = 0.0331; % echo time (seconds)

 

% Connectivity matrices

a  = ones(n,n);


b  = zeros(n,n,nu);

c  = zeros(n,nu);

d  = zeros(n,n,0);

 

% Specify DCM

s = struct();

s.name       = model_name;

s.u          = [];

s.delays     = repmat(TR/2, 1, n)';

s.TE         = TE;

s.nonlinear  = false;

s.two_state  = false;

s.stochastic = false;

s.centre     = false;

s.induced    = 1;       % indicates DCM for CSD

s.a          = a;

s.b          = b;

s.c          = c;

s.d          = d;

 

DCM = spm_dcm_specify(SPM,xY,s);



end

clear DCM


for sI = 1: length(subNames)

cd(fullfile(root_dir, 'GLM2', subNames{sI}));

model_name = 'R_MOT';

xY         = voiNamesR;

SPM        = 'SPM.mat';

n   = 2;    % number of regions

nu  = 1;    % number of inputs. For DCM for CSD we have one input: null

TR  = 0.72;    % volume repetition time (seconds)

TE  = 0.0331; % echo time (seconds)

 

% Connectivity matrices

a  = ones(n,n);


b  = zeros(n,n,nu);

c  = zeros(n,nu);

d  = zeros(n,n,0);

 

% Specify DCM

s = struct();

s.name       = model_name;

s.u          = [];

s.delays     = repmat(TR/2, 1, n)';

s.TE         = TE;

s.nonlinear  = false;

s.two_state  = false;

s.stochastic = false;

s.centre     = false;

s.induced    = 1;       % indicates DCM for CSD

s.a          = a;

s.b          = b;

s.c          = c;

s.d          = d;

 

DCM = spm_dcm_specify(SPM,xY,s);



end

clear DCM
%%estimate DCMs



 for h=1: length(subNames) 
   
 GCM_L_MOT(h,1) = {fullfile(root_dir, 'GLM2', subNames{h},'DCM_L_MOT.mat')}; 
 
 end
  
 
 for h=1: length(subNames) 
   
 GCM_R_MOT(h,1) = {fullfile(root_dir, 'GLM2', subNames{h},'DCM_R_MOT.mat')}; 
 
 end
  
cd(root_dir);

use_parfor = true ;
GCM_L_MOT = spm_dcm_fit(GCM_L_MOT);
save('GCM_L_MOT.mat','GCM_L_MOT');


GCM_R_MOT = spm_dcm_fit(GCM_R_MOT);
save('GCM_R_MOT.mat','GCM_R_MOT');

voiNamesL={'VOI_laIns_1.mat','VOI_lpIns_1.mat'};
voiNamesR={'VOI_raIns_1.mat','VOI_rpIns_1.mat'};

for sI = 1: length(subNames)

cd(fullfile(root_dir,'GLM2', subNames{sI}));

model_name = 'L_INTERO';

xY         = voiNamesL;

SPM        = 'SPM.mat';

n   = 2;    % number of regions

nu  = 1;    % number of inputs. For DCM for CSD we have one input: null

TR  = 0.72;    % volume repetition time (seconds)

TE  = 0.0331; % echo time (seconds)

 

% Connectivity matrices

a  = ones(n,n);


b  = zeros(n,n,nu);

c  = zeros(n,nu);

d  = zeros(n,n,0);

 

% Specify DCM

s = struct();

s.name       = model_name;

s.u          = [];

s.delays     = repmat(TR/2, 1, n)';

s.TE         = TE;

s.nonlinear  = false;

s.two_state  = false;

s.stochastic = false;

s.centre     = false;

s.induced    = 1;       % indicates DCM for CSD

s.a          = a;

s.b          = b;

s.c          = c;

s.d          = d;

 

DCM = spm_dcm_specify(SPM,xY,s);



end

clear DCM


for sI = 1: length(subNames)

cd(fullfile(root_dir, 'GLM2', subNames{sI}));

model_name = 'R_INTERO';

xY         = voiNamesR;

SPM        = 'SPM.mat';

n   = 2;    % number of regions

nu  = 1;    % number of inputs. For DCM for CSD we have one input: null

TR  = 0.72;    % volume repetition time (seconds)

TE  = 0.0331; % echo time (seconds)

 

% Connectivity matrices

a  = ones(n,n);


b  = zeros(n,n,nu);

c  = zeros(n,nu);

d  = zeros(n,n,0);

 

% Specify DCM

s = struct();

s.name       = model_name;

s.u          = [];

s.delays     = repmat(TR/2, 1, n)';

s.TE         = TE;

s.nonlinear  = false;

s.two_state  = false;

s.stochastic = false;

s.centre     = false;

s.induced    = 1;       % indicates DCM for CSD

s.a          = a;

s.b          = b;

s.c          = c;

s.d          = d;

 

DCM = spm_dcm_specify(SPM,xY,s);



end

clear DCM
%%estimate DCMs



 for h=1: length(subNames) 
   
 GCM_L_INTERO(h,1) = {fullfile(root_dir, 'GLM2', subNames{h},'DCM_L_INTERO.mat')}; 
 
 end
  
 
 for h=1: length(subNames) 
   
 GCM_R_INTERO(h,1) = {fullfile(root_dir, 'GLM2', subNames{h},'DCM_R_INTERO.mat')}; 
 
 end
  
cd(root_dir);

use_parfor = true ;
GCM_L_INTERO = spm_dcm_fit(GCM_L_INTERO);
save('GCM_L_INTERO.mat','GCM_L_INTERO');


GCM_R_INTERO = spm_dcm_fit(GCM_R_INTERO);
save('GCM_R_INTERO.mat','GCM_R_INTERO');


voiNamesL={'VOI_lFp1_1.mat','VOI_lSSC_1.mat','VOI_lV1_1.mat','VOI_lA1_1.mat'};
voiNamesR={'VOI_rFp1_1.mat','VOI_rSSC_1.mat','VOI_rV1_1.mat','VOI_rA1_1.mat'};

for sI = 1: length(subNames)

cd(fullfile(root_dir,'GLM2', subNames{sI}));

model_name = 'L_EXTERO';

xY         = voiNamesL;

SPM        = 'SPM.mat';

n   = 4;    % number of regions

nu  = 1;    % number of inputs. For DCM for CSD we have one input: null

TR  = 0.72;    % volume repetition time (seconds)

TE  = 0.0331; % echo time (seconds)

 

% Connectivity matrices

a  = ones(n,n);


b  = zeros(n,n,nu);

c  = zeros(n,nu);

d  = zeros(n,n,0);

 

% Specify DCM

s = struct();

s.name       = model_name;

s.u          = [];

s.delays     = repmat(TR/2, 1, n)';

s.TE         = TE;

s.nonlinear  = false;

s.two_state  = false;

s.stochastic = false;

s.centre     = false;

s.induced    = 1;       % indicates DCM for CSD

s.a          = a;

s.b          = b;

s.c          = c;

s.d          = d;

 

DCM = spm_dcm_specify(SPM,xY,s);



end

clear DCM


for sI = 1: length(subNames)

cd(fullfile(root_dir, 'GLM2', subNames{sI}));

model_name = 'R_EXTERO';

xY         = voiNamesR;

SPM        = 'SPM.mat';

n   = 4;    % number of regions

nu  = 1;    % number of inputs. For DCM for CSD we have one input: null

TR  = 0.72;    % volume repetition time (seconds)

TE  = 0.0331; % echo time (seconds)

 

% Connectivity matrices

a  = ones(n,n);


b  = zeros(n,n,nu);

c  = zeros(n,nu);

d  = zeros(n,n,0);

 

% Specify DCM

s = struct();

s.name       = model_name;

s.u          = [];

s.delays     = repmat(TR/2, 1, n)';

s.TE         = TE;

s.nonlinear  = false;

s.two_state  = false;

s.stochastic = false;

s.centre     = false;

s.induced    = 1;       % indicates DCM for CSD

s.a          = a;

s.b          = b;

s.c          = c;

s.d          = d;

 

DCM = spm_dcm_specify(SPM,xY,s);



end

clear DCM
%%estimate DCMs


 for h=1: length(subNames) 
   
 GCM_L_EXTERO(h,1) = {fullfile(root_dir, 'GLM2', subNames{h},'DCM_L_EXTERO.mat')}; 
 
 end
  
 
 for h=1: length(subNames) 
   
 GCM_R_EXTERO(h,1) = {fullfile(root_dir, 'GLM2', subNames{h},'DCM_R_EXTERO.mat')}; 
 
 end
  
cd(root_dir);

use_parfor = true ;
GCM_L_EXTERO = spm_dcm_fit(GCM_L_EXTERO);
save('GCM_L_EXTERO.mat','GCM_L_EXTERO');


GCM_R_EXTERO = spm_dcm_fit(GCM_R_EXTERO);
save('GCM_R_EXTERO.mat','GCM_R_EXTERO');

rmdir('func','s')  
rmdir('spm12','s')
rmdir('masks','s')
rmdir('GLM','s')
rmdir('GLM2','s')
