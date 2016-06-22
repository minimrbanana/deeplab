%% set path and parameters
close all; clc;
addpath(genpath('/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/densecrf/'));
addpath(genpath('/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/matlab/my_script/'));
SetupEnv;
config.imageset = 'test';
config.cmap= 'pascal_seg_colormap.mat';
config.gpuNum =1;
config.Path.CNN.caffe_root = '/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/';
config.save_root = fullfile('/BS/joint-multicut-2/work/FBMS-fasterRCNN/deeplab/', dataset, feature_name, model_name, testset);
model_name='deeplab_largeFOV';
config.write_file = 1;
config.Path.CNN.script_path = fullfile('/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/deeplab/exper/', dataset);
config.Path.CNN.model_data = [config.Path.CNN.script_path '/model/' model_name '/train_iter_8000.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/config/' model_name '/deploy.prototxt'];
config.im_sz = 513;%657;?

%% initialization
load(config.cmap);

%% initialize caffe
% reboot MATLAB every time?
addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('initializing caffe..\n');
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data);
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
caffe('set_phase_test');
fprintf('done\n');

%% initialize paths
% save_res_dir = [config.save_root, '/test1'];
% save_res_path = [save_res_dir, '/%s.png'];
% 
% if config.write_file
%     if ~exist(save_res_dir,'dir')
%         mkdir(save_res_dir)
%     end
% end