clear all; close all; clc;
cd /BS/vidsegmentHOrder2/work/DeepLab/deeplab/matlab/caffe
addpath(genpath('/BS/vidsegmentHOrder2/work/DeepLab/deeplab/densecrf'));
addpath(genpath('/BS/vidsegmentHOrder2/work/DeepLab/deeplab/matlab/my_script'));
SetupEnv;
config.imageset = 'test';
config.cmap= 'pascal_seg_colormap.mat';
config.gpuNum =0;
config.Path.CNN.caffe_root = '//BS/vidsegmentHOrder2/work/DeepLab/deeplab/caffe';
config.save_root = fullfile('//BS/vidsegmentHOrder2/work/DeepLab/', dataset, feature_name, model_name, testset);
% model_name='DeepLab-MSc'
config.write_file = 1;
config.Path.CNN.script_path = fullfile('//BS/vidsegmentHOrder2/work/DeepLab/', dataset);
config.Path.CNN.model_data = [config.Path.CNN.script_path '/model/' model_name '/train10_rect_box_region_iter_6000.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/config/' model_name '/deploy.prototxt'];
config.im_sz = 513;%657;

%%
VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
COCO.imgsetpath='//BS/vidsegmentHOrder2/work/DeepLab/coco/list/'
COCO.testset='test_id'
COCO.nclasses=21;%90;
COCO.segpath=fullfile('//BS/vidsegmentHOrder2/work/coco/segm_data', 'segm_class_raw_correct',testset);
%% initialization
load(config.cmap);


%% initialize caffe
addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('initializing caffe..\n');
% if caffe('is_initialized')
%     caffe('release')
% end
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data)
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
caffe('set_phase_test');
fprintf('done\n');


%% initialize paths
save_res_dir = [config.save_root, '/train10_rect_box_region_iter_6000_crf2'];
save_res_path = [save_res_dir, '/%s.png'];

%% create directory
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end


fprintf('start generating result\n');
fprintf('caffe model: %s\n', config.Path.CNN.model_proto);
fprintf('caffe weight: %s\n', config.Path.CNN.model_data);


%% read VOC2012 TEST image set
ids=textread([COCO.imgsetpath,COCO.testset,'.txt'],'%s');

%%

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
%           fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
%    
%     
%     % read image
%       I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
%       [img_height, img_width, ~] = size(I);
%     
%     
%       img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      result = segImg(1:img_row, 1:img_col, :);
   
      [~, result] = max(result, [], 3);
       result = uint8(result) - 1;     
      
            
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');

%% Bilateral solver
mkdir('tmp');
for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i}); 
   
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      result = segImg(1:img_row, 1:img_col, :);
   
      [~, result] = max(result, [], 3);
%       result = uint8(result) - 1;
      
    ens_score = segImg(1:img_row, 1:img_col, :);
    prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
    prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
    new_chan_img=zeros(size(prob_map));
    labels=unique(result);
    imwrite(img,'tmp/img.png');
    setenv('LD_LIBRARY_PATH', '');
    parfor k=1:numel(labels)%size(prob_map,3)
        chan_img=prob_map(:,:,labels(k));
        chan_img=uint8(255*chan_img);
        chan_img=cat(3,chan_img,chan_img,chan_img);
        imwrite(chan_img,['tmp/chan_img_',num2str(labels(k)),'.png']);
         for h=1:25  
          cmd = ['python bil_sover_for_matlab.py tmp/img.png tmp/chan_img_',num2str(labels(k)),'.png'];       
          system(cmd);
          dum_img=imread(['tmp/chan_img_',num2str(labels(k)),'.png_bs.png']);
          dum_img = RF(dum_img, 16, 16,3);
          imwrite(dum_img,['tmp/chan_img_',num2str(labels(k)),'.png']);
         end      
    end
    
    for k=1:numel(labels)
       dum_img=imread(['tmp/chan_img_',num2str(labels(k)),'.png']);
       new_chan_img(:,:,labels(k))=dum_img(:,:,1);
    end
    
 [~, result_new] = max(new_chan_img, [], 3);
 
 
 imwrite(result_new, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');
%% gt modif
save_res_dir = '//BS/vidsegmentHOrder2/work/DeepLab/coco/features/DeepLab-LargeFOV/val/train2_iter_6000_crf_gt_bb_clean';
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end

gt_bb_dir='/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/Annotations';
classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    gtFile=fullfile(gt_bb_dir,[ids{i} '.xml']);     
    rec=VOCreadxml(gtFile);
    gt=rec.annotation.object;%load(gtFile,'groundTruth');
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
      ens_score_new=zeros(size(ens_score));
      prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
      prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
      labs=numel(gt);
      ens_score_new(:,:,1)= 1;
      for gg=1:labs    
          ggt=rec.annotation.object(gg).bndbox;
          class=rec.annotation.object(gg).name;
          [~,se]=ismember(class,classes);

          topM = str2num(ggt.ymin);
          bottomM = str2num(ggt.ymax);
          leftN = str2num(ggt.xmin);
          rightN =str2num(ggt.xmax);
          ens_score_new(topM:bottomM, leftN:rightN,se+1)=prob_map(topM:bottomM, leftN:rightN,se+1);
          ens_score_new(topM:bottomM, leftN:rightN,1)=prob_map(topM:bottomM, leftN:rightN,1);
      end
%       [~, result] = max(ens_score_new, [], 3);
%       result = uint8(result) - 1;
%       
     fprintf('[densecrf.. ');
  
    unary = -log(ens_score_new);

    D = Densecrf(img,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 50;
    D.bilateral_y_stddev = 50;
    D.bilateral_r_stddev = 3;
    D.bilateral_g_stddev = 3;
    D.bilateral_b_stddev = 3;
    D.bilateral_weight = 5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    result = uint8(segmask-1);    
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');
%% %% gt modif fast rcnn + sese
% gt_bb_dir='/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/Annotations';
bsdsDir=['/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/data'];
gt_bb_dir=[bsdsDir, '/fast_rcnn_bb/voc12/val_all'];

classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    gtFile=fullfile(gt_bb_dir,[ids{i} '.mat']);
     gt_bb=load(gtFile);
     
     
     boxes_cell=gt_bb.boxes_cell;
     c=1;
     for h=1:numel(boxes_cell);
      I = boxes_cell{h}(:, 5) >= .8;
      boxes_cell{h} = boxes_cell{h}(I, :);
      if ~isempty(boxes_cell{h})
          for hh=1:size(boxes_cell{h},1)
              bboxes_gt{c}=[boxes_cell{h}(hh,:),h];
              c=c+1;
          end
      end          
     end
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
      ens_score_new=zeros(size(ens_score));
      prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
      prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
      labs=numel(bboxes_gt);
      ens_score_new(:,:,1)= 1;
      for gg=1:labs    
          se=bboxes_gt{gg}(6);
          topM = max(round(bboxes_gt{gg}(2)),1);
          bottomM =  min(round(bboxes_gt{gg}(4)),size(img,1));
          leftN =  max(round(bboxes_gt{gg}(1)),1);
          rightN = min(round(bboxes_gt{gg}(3)),size(img,2));
        
          ens_score_new(topM:bottomM, leftN:rightN,se+1)=prob_map(topM:bottomM, leftN:rightN,se+1);
          ens_score_new(topM:bottomM, leftN:rightN,1)=prob_map(topM:bottomM, leftN:rightN,1);
      end
%       [~, result] = max(ens_score_new, [], 3);
%       result = uint8(result) - 1;
%       
     fprintf('[densecrf.. ');
  
    unary = -log(ens_score_new);

    D = Densecrf(img,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 50;
    D.bilateral_y_stddev = 50;
    D.bilateral_r_stddev = 3;
    D.bilateral_g_stddev = 3;
    D.bilateral_b_stddev = 3;
    D.bilateral_weight = 5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    result = uint8(segmask-1);    
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');

%% %% train gt modif
save_res_dir = '//BS/vidsegmentHOrder2/work/DeepLab/coco/features/DeepLab-Weak-Bbox-Rect/train/train2_iter_6000_gt_bb_clean';
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end

output_folder='//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG/';
im_folder2=[output_folder,'images'];
input_dir = dir(fullfile(im_folder2, '*.png'));

gt_bb_dir='/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/Annotations';
classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i=1:length(input_dir)
  fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
   
       
    gtFile=fullfile(gt_bb_dir,[input_dir(i).name(1:end-4) '.xml']);     
    rec=VOCreadxml(gtFile);
    gt=rec.annotation.object;%load(gtFile,'groundTruth');
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
      ens_score_new=zeros(size(ens_score));
      prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
      prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
      labs=numel(gt);
      ens_score_new(:,:,1)= 1;
      for gg=1:labs    
          ggt=rec.annotation.object(gg).bndbox;
          class=rec.annotation.object(gg).name;
          [~,se]=ismember(class,classes);

          topM = str2num(ggt.ymin);
          bottomM = str2num(ggt.ymax);
          leftN = str2num(ggt.xmin);
          rightN =str2num(ggt.xmax);
          ens_score_new(topM:bottomM, leftN:rightN,se+1)=prob_map(topM:bottomM, leftN:rightN,se+1);
          ens_score_new(topM:bottomM, leftN:rightN,1)=prob_map(topM:bottomM, leftN:rightN,1);
      end
%       [~, result] = max(ens_score_new, [], 3);
%       result = uint8(result) - 1;
%       
     fprintf('[densecrf.. ');
  
    unary = -log(ens_score_new);

    D = Densecrf(img,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 50;
    D.bilateral_y_stddev = 50;
    D.bilateral_r_stddev = 3;
    D.bilateral_g_stddev = 3;
    D.bilateral_b_stddev = 3;
    D.bilateral_weight = 5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    result = uint8(segmask-1);    
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [input_dir(i).name(1:end-4), '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
 

%% train set
save_res_dir = '//BS/vidsegmentHOrder2/work/DeepLab/coco/features/DeepLab-LargeFOV/train/train2_iter_6000_crf';
mkdir(save_res_dir);
% 
COCO.testset='train_id'

% read VOC2012 TEST image set
ids=textread([COCO.imgsetpath,COCO.testset,'.txt'],'%s');
output_folder='//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG/';
im_folder2=[output_folder,'images'];
input_dir = dir(fullfile(im_folder2, '*.png'));

for i=1:length(input_dir)
    fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
   
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      result = segImg(1:img_row, 1:img_col, :);
   
      [~, result] = max(result, [], 3);
      result = uint8(result) - 1;
      
 imwrite(result, colormap, fullfile(save_res_dir, [input_dir(i).name(1:end-4), '.png']));
  
end

% %% crf
% save_res_dir=[save_res_dir,'_crf'];
% mkdir(save_res_dir);
    COCO.testset='train_id'

% read VOC2012 TEST image set
ids=textread([COCO.imgsetpath,COCO.testset,'.txt'],'%s');
output_folder='//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG/';
im_folder2=[output_folder,'images'];
input_dir = dir(fullfile(im_folder2, '*.png'));
for i=1:length(input_dir)
%     fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
%    
%     
%     % read image
%       I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
%       [img_height, img_width, ~] = size(I);
%     
%     
%       img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
          fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
   
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
      
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
   
%       [~, result] = max(result, [], 3);
%       result = uint8(result) - 1;
%       
      
     fprintf('[densecrf.. ');
    prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
    prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
    unary = -log(prob_map);

    D = Densecrf(img,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 50;
    D.bilateral_y_stddev = 50;
    D.bilateral_r_stddev = 3;
    D.bilateral_g_stddev = 3;
    D.bilateral_b_stddev = 3;
    D.bilateral_weight = 5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    result = uint8(segmask-1);    
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [input_dir(i).name(1:end-4), '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
   

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
%           fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
%    
%     
%     % read image
%       I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
%       [img_height, img_width, ~] = size(I);
%     
%     
%       img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
   
%       [~, result] = max(result, [], 3);
%       result = uint8(result) - 1;
%       
      
     fprintf('[densecrf.. ');
    prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
    prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
    unary = -log(prob_map);

    D = Densecrf(img,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 30;%50;
    D.bilateral_y_stddev = 30;%50;
    D.bilateral_r_stddev = 3;%3;
    D.bilateral_g_stddev =3;%3;
    D.bilateral_b_stddev = 3;%3;
    D.bilateral_weight = 5;%5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    result = uint8(segmask-1);    
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');


for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
%           fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
%    
%     
%     % read image
%       I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
%       [img_height, img_width, ~] = size(I);
%     
%     
%       img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
   
%       [~, result] = max(result, [], 3);
%       result = uint8(result) - 1;
%       
      
     fprintf('[densecrf.. ');
    prob_map = exp(ens_score - repmat(max(ens_score, [], 3), [1,1,size(ens_score,3)]));
    prob_map = prob_map ./ repmat(sum(prob_map, 3), [1,1, size(prob_map,3)]);
    unary = -log(prob_map);

    D = Densecrf(img,single(unary));

    % Some settings.
    D.gaussian_x_stddev = 3;
    D.gaussian_y_stddev = 3;
    D.gaussian_weight = 3; 

    D.bilateral_x_stddev = 50;
    D.bilateral_y_stddev = 50;
    D.bilateral_r_stddev = 3;
    D.bilateral_g_stddev =3;
    D.bilateral_b_stddev = 3;
    D.bilateral_weight = 5;     
   
    D.iterations = 10;

    D.mean_field;
    segmask = D.segmentation;
    result = uint8(segmask-1);    
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');

%%new crf test
for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    
    % read image
%       I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
%       [img_height, img_width, ~] = size(I);
%     
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages_test', [ids{i}, '.jpg']));
%           fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
%    
%     
%     % read image
%       I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
%       [img_height, img_width, ~] = size(I);
%     
%     
%       img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      I=uint8(zeros(513,513,3));
      I(1:img_row, 1:img_col,:)=img;
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
      result=uint8(dencecrf10(img,cnn_output{1}));
     
%     result = uint8(segmask-1);    
    fprintf('done] ');
 if  ~isempty( result)    
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 end
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');
%% new crf train
output_folder='//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG/';
im_folder2=[output_folder,'images'];
input_dir = dir(fullfile(im_folder2, '*.png'));
for i=1:length(input_dir)
  fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
   
       
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
      result=uint8(dencecrf16(img,cnn_output{1}));
     
%     result = uint8(segmask-1);    
    fprintf('done] ');
 if  ~isempty( result)    
 imwrite(result, colormap, fullfile(save_res_dir, [input_dir(i).name(1:end-4), '.png']));
 end
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');
%% new crf gt modif train
output_folder='//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG/';
im_folder2=[output_folder,'images'];
input_dir = dir(fullfile(im_folder2, '*.png'));

gt_bb_dir='/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/Annotations';
classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i=1:length(input_dir)
  fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
   
       
    gtFile=fullfile(gt_bb_dir,[input_dir(i).name(1:end-4) '.xml']);     
    rec=VOCreadxml(gtFile);
    gt=rec.annotation.object;%load(gtFile,'groundTruth');
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
      ens_score_new=zeros(size(ens_score));
         
    %Transform data to probability
    data = exp(ens_score);
    prob_map = bsxfun(@rdivide, data, sum(data, 3));
    
      labs=numel(gt);
      ens_score_new(:,:,1)= 1;
      for gg=1:labs    
          ggt=rec.annotation.object(gg).bndbox;
          class=rec.annotation.object(gg).name;
          [~,se]=ismember(class,classes);

          topM = str2num(ggt.ymin);
          bottomM = str2num(ggt.ymax);
          leftN = str2num(ggt.xmin);
          rightN =str2num(ggt.xmax);
          ens_score_new(topM:bottomM, leftN:rightN,se+1)=prob_map(topM:bottomM, leftN:rightN,se+1);
          ens_score_new(topM:bottomM, leftN:rightN,1)=prob_map(topM:bottomM, leftN:rightN,1);
      end
%       [~, result] = max(ens_score_new, [], 3);
%       result = uint8(result) - 1;
%       
    result=uint8(dencecrf_new4(img,ens_score_new));
    fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [input_dir(i).name(1:end-4), '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
 %%
 gt_bb_dir='/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/Annotations';
classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i=1:length(ids)
    fprintf('progress: %d/%d [%s]...\n', i, length(ids), ids{i});  
   
    gtFile=fullfile(gt_bb_dir,[ids{i} '.xml']);     
    rec=VOCreadxml(gtFile);
    gt=rec.annotation.object;%load(gtFile,'groundTruth');
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [ids{i}, '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [ids{i}, '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
      ens_score_new=zeros(size(ens_score));
         
    %Transform data to probability
    data = exp(ens_score);
    prob_map = bsxfun(@rdivide, data, sum(data, 3));
    
      labs=numel(gt);
      ens_score_new(:,:,1)= 1;
      for gg=1:labs    
          ggt=rec.annotation.object(gg).bndbox;
          class=rec.annotation.object(gg).name;
          [~,se]=ismember(class,classes);

          topM = str2num(ggt.ymin);
          bottomM = str2num(ggt.ymax);
          leftN = str2num(ggt.xmin);
          rightN =str2num(ggt.xmax);
          ens_score_new(topM:bottomM, leftN:rightN,se+1)=prob_map(topM:bottomM, leftN:rightN,se+1);
          ens_score_new(topM:bottomM, leftN:rightN,1)=prob_map(topM:bottomM, leftN:rightN,1);
      end
%       [~, result] = max(ens_score_new, [], 3);
%       result = uint8(result) - 1;
%       
    result=uint8(dencecrf_new10(img,ens_score_new));
%       
       fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [ids{i}, '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
    VOC_root_folder = '//BS/vidsegmentHOrder2/work/VOCdevkit';
  seg_res_dir = [save_res_dir '/results/VOC2012/'];
  seg_root = fullfile(VOC_root_folder, 'VOC2012');
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, 'VOC2012');
  VOCopts.seg.clsresdir=save_res_dir;
VOCopts.seg.clsrespath=[VOCopts.seg.clsresdir '/%s.png'];
id='';
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

save([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');

%%
output_folder='//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG/';
im_folder2=[output_folder,'images'];
input_dir = dir(fullfile(im_folder2, '*.png'));

gt_bb_dir='/BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/Annotations';
classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};

for i=1:length(input_dir)
  fprintf('progress: %d/%d [%s]...\n', i, length(input_dir), input_dir(i).name(1:end-4)); 
   
       
    gtFile=fullfile(gt_bb_dir,[input_dir(i).name(1:end-4) '.xml']);     
    rec=VOCreadxml(gtFile);
    gt=rec.annotation.object;%load(gtFile,'groundTruth');
    
    % read image
      I = imread(fullfile('//BS/vidsegmentHOrder2/work/DeepLab/VOC2012_SEG_AUG', 'images', [input_dir(i).name(1:end-4), '.png']));
      [img_height, img_width, ~] = size(I);
    
    
      img = imread(fullfile('//BS/vidsegmentHOrder2/work/VOCdevkit/VOC2012/JPEGImages', [input_dir(i).name(1:end-4), '.jpg']));
%       
      img_row = size(img, 1);
      img_col = size(img, 2);
    
      input_data = preprocess_image(double(I), config.im_sz); 
      cnn_output = caffe('forward', input_data);
        
      segImg = permute(cnn_output{1}, [2, 1, 3]);
        
      ens_score = segImg(1:img_row, 1:img_col, :);
      ens_score_new=zeros(size(ens_score));
         
    %Transform data to probability
    data = exp(ens_score);
    prob_map = bsxfun(@rdivide, data, sum(data, 3));
    
      labs=numel(gt);
      ens_score_new(:,:,1)= 1;
      for gg=1:labs    
          ggt=rec.annotation.object(gg).bndbox;
          class=rec.annotation.object(gg).name;
          [~,se]=ismember(class,classes);

          topM = str2num(ggt.ymin);
          bottomM = str2num(ggt.ymax);
          leftN = str2num(ggt.xmin);
          rightN =str2num(ggt.xmax);
          ens_score_new(topM:bottomM, leftN:rightN,se+1)=prob_map(topM:bottomM, leftN:rightN,se+1);
          ens_score_new(topM:bottomM, leftN:rightN,1)=prob_map(topM:bottomM, leftN:rightN,1);
      end
      [~, result] = max(ens_score_new, [], 3);
      result = uint8(result) - 1;
%       
%     result=uint8(dencecrf_new16(img,ens_score_new));
%     fprintf('done] ');
         
 imwrite(result, colormap, fullfile(save_res_dir, [input_dir(i).name(1:end-4), '.png']));
 if 0
    imshow(img);
    hold on,
    h = imshow(ind2rgb(result, colormap)); 
     alpha = (result~=0) .* (0.75*ones(size(img,1), size(img,2)));
     set(h, 'AlphaData', alpha)
 end
 
end
% 
% 

% 
% %% plot 
% load([save_res_dir,'/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');
% data2 = load(['//BS/vidsegmentHOrder2/work/DeepLab/coco/features/DeepLab-COCO-Msc/val/train2_bal_correct_v2_iter_26000/results.mat'],'accuracies', 'avacc', 'conf', 'rawcounts');
% load('//BS/vidsegmentHOrder2/work/coco_train_stat_correct.mat','class_stat','count_class_im');
% load('//BS/vidsegmentHOrder2/work/coco_train_stat_size.mat','count_class_size','count_class_im');
% load('//BS/vidsegmentHOrder2/work/coco_train_stat_inst.mat','cat_id','h');
% 
% [h_s2, idx2]=sort(h,'descend');
% 
% accuracies(1)=[];
% accuracies(count_class_im==0)=[];
% data2.accuracies(1)=[];
% data2.accuracies(count_class_im==0)=[];
% count_class_size(count_class_size==0)=[];
% count_class_im(count_class_im==0)=[];
% 
% 
% % ss=h(idx2);
% % ss(72)=ss(72)+1;
% % names=nms(idx);
% % figure()
% % hold on
% % bar(log(ss),accuracies(idx2));
% % ax=gca;
% % %ax.XTick=[1:80];
% % ax.XTickLabel=nms(idx2);
% % ax.XTickLabelRotation=90;
% % ax.XTickLabelMode='manual';
% % 
% % figure()
% % hold on
% % X = [log(h(idx2)),log(count_class_im(idx2)),accuracies(idx2)/10];
% % B=bar3(X);
% % ax=gca;
% % ax.YTick=[1:80];
% % ax.YTickLabel=nms(idx2);
% % view(3)
% 
% figure(1)
% hold on
% scatter(log(h(idx2)),accuracies(idx2),20, 'MarkerEdgeColor','k','MarkerFaceColor',[1 0 0]);
% text(log(h(idx2))+.03,accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% 
% scatter(log(h(idx2)),data2.accuracies(idx2),20, 'MarkerEdgeColor','k','MarkerFaceColor',[0 1 0]);
% text(log(h(idx2))+.03,data2.accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% xlabel('Number of instances per class in log scale');
% ylabel('Mean IoU per class');
% ax=gca;
% legend('train on balanced data - 80%, 26k iterations','train on balanced data - 100%, 26k iterations')
% %ax.XLim=[4.7 12.7];
% 
% figure(2)
% hold on
% scatter(log(count_class_size(idx2)),accuracies(idx2),20, 'MarkerEdgeColor','k','MarkerFaceColor',[1 0 0]);
% text(log(count_class_size(idx2))+.03,accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% 
% scatter(log(count_class_size(idx2)),data2.accuracies(idx2),20, 'MarkerEdgeColor','k','MarkerFaceColor',[0 1 0]);
% text(log(count_class_size(idx2))+.03,data2.accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% 
% xlabel('Number of pixels per class in log scale');
% ylabel('Mean IoU per class');
% ax=gca;
% legend('train on balanced data - 80%, 26k iterations','train on balanced data - 100%, 26k iterations')
% 
% %ax.XLim=[4.7 11];
% 
% figure(3)
% hold on
% scatter(log(count_class_size(idx2)./h(idx2)),accuracies(idx2),20, 'MarkerEdgeColor','k','MarkerFaceColor',[1 0 0]);
% text(log(count_class_size(idx2)./h(idx2))+.03,accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% scatter(log(count_class_size(idx2)./h(idx2)),data2.accuracies(idx2),20, 'MarkerEdgeColor','k','MarkerFaceColor',[0 1 0]);
% text(log(count_class_size(idx2)./h(idx2))+.03,data2.accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% 
% xlabel('Average size');
% ylabel('Mean IoU per class');
% ax=gca;
% legend('train on balanced data - 80%, 26k iterations','train on balanced data - 100%, 26k iterations')
% 
% %ax.XLim=[4.7 11];
% %% delta 
% 
% figure(1)
% hold on
% s=data2.accuracies-accuracies;
% s(isnan(s))=0;
% c = linspace(1,10,length(s));
% 
% [hhh,idx2]=sort(s);
% 
% scatter(log(h(idx2)),data2.accuracies(idx2),[],hhh,'filled'); colorbar;
% text(log(h(idx2))+.03,data2.accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% 
% xlabel('Number of instances per class in log scale');
% ylabel('Mean IoU per class');
% ax=gca;
% legend('train on balanced data - 100%, 26k iterations')
% %ax.XLim=[4.7 12.7];
% 
% figure(2)
% hold on
% scatter(log(count_class_size(idx2)),data2.accuracies(idx2),[],hhh,'filled');colorbar;
% text(log(count_class_size(idx2))+.03,data2.accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% 
% xlabel('Number of pixels per class in log scale');
% ylabel('Mean IoU per class');
% ax=gca;
% legend('train on balanced data - 100%, 26k iterations')
% 
% %ax.XLim=[4.7 11];
% 
% figure(3)
% hold on
% scatter(log(count_class_size(idx2)./h(idx2)),data2.accuracies(idx2),[],hhh,'filled');colorbar;
% text(log(count_class_size(idx2)./h(idx2))+.03,data2.accuracies(idx2)-.01,nms(idx2),'color',[0,0,0],'FontSize',7);
% xlabel('Average size');
% ylabel('Mean IoU per class');
% ax=gca;
% legend('train on balanced data - 100%, 26k iterations')
% 
% 
% 
% % scatter3(log(h(idx2)),log(count_class_im(idx2)),accuracies(idx2),30, 'MarkerEdgeColor','k','MarkerFaceColor',[0 .75 .75]);
% % text(log(h(idx2)),log(count_class_im(idx2)),accuracies(idx2),nms(idx2),'color',[0,0,0],'FontSize',8);
% % view([10,20,10])
% % 
% % for k = 1:length(B)
% %    % zdata = B(k).ZData;
% %     %B(k).CData = zdata;
% %    % B(k).FaceColor = 'interp';
% %      B(k).FaceAlpha = .65;
% % end
% % 
% % xlabel('MPG'); ylabel('Weight');
% % set(gcf,'renderer','opengl');
% % set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
