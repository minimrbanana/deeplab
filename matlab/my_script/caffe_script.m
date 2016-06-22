clear all; close all; clc;
addpath(genpath('/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/densecrf/'));
addpath(genpath('/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/matlab/my_script/'));
SetupEnv;
config.imageset = 'test';
config.cmap= 'pascal_seg_colormap.mat';
config.gpuNum =0;
config.Path.CNN.caffe_root = '/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/';
config.save_root = fullfile('/BS/joint-multicut-2/work/FBMS-fasterRCNN/deeplab/', dataset, feature_name, model_name, testset);
model_name='deeplab_largeFOV';
config.write_file = 1;
config.Path.CNN.script_path = fullfile('/home/zhongjie/CodesDown/jointmulticut/joint-multicut/deeplab/deeplab/exper/', dataset);
config.Path.CNN.model_data = [config.Path.CNN.script_path '/model/' model_name '/train_iter_8000.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/config/' model_name '/deploy.prototxt'];
config.im_sz = 513;%657;?

%%
% VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
% COCO.imgsetpath='//BS/vidsegmentHOrder2/work/DeepLab/coco/list/'
% COCO.testset='test_id'
% COCO.nclasses=21;%90;
% COCO.segpath=fullfile('//BS/vidsegmentHOrder2/work/coco/segm_data', 'segm_class_raw_correct',testset);
%% initialization
load(config.cmap);

%% initialize caffe
addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('initializing caffe..\n');
% if caffe('is_initialized')
%     caffe('release')
% end
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data);
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
caffe('set_phase_test');
fprintf('done\n');

%% initialize paths
save_res_dir = [config.save_root, '/test1'];
save_res_path = [save_res_dir, '/%s.png'];

if config.write_file
    if ~exist(save_res_dir,'dir')
        mkdir(save_res_dir)
    end
end

% read image
dataDir = '/BS/joint-multicut/work/FBMS59/Testset/';
Type = '*.ppm';
I = imread([dataDir 'cars1/cars1_12.ppm']);
%I = imread('21.jpg');
[img_height, img_width, ~] = size(I);
boxDir = '/BS/joint-multicut-2/work/FBMS-fasterRCNN/Testset03/';
temp = load([boxDir 'cars1/cars1_12.mat']);
Bbox = temp.Boxes2save;
for box_index=1:size(Bbox,1)
    cur_box = Bbox(box_index,:);
    rect = [cur_box(1) cur_box(2) cur_box(3)-cur_box(1) cur_box(4)-cur_box(2)];
    img = I(round(cur_box(2)):round(cur_box(4)),round(cur_box(1)):round(cur_box(3)),:);
    input_data = preprocess_image(double(img), config.im_sz);
    cnn_output = caffe('forward', input_data);
    segImg = permute(cnn_output{1}, [2, 1, 3]);

    result = segImg;%(1:img_height, 1:img_width, :);
    [~, result] = max(result, [], 3);
    result = uint8(result) - 1; 
    figure(1);
    imagesc(result);
    figure(2);
    imshow(img);
end







