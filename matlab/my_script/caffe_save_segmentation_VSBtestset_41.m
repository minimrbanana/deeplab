%% script for saving segmentations using deeplab
% initialize caffe first
fprintf('Please init caffe\n');
%pause;
% set data path
dataChoice = 'Testset';
dataDir = '/BS/siyu-project/work/MulticutMotionTracking/dataset/VSB100/';
Type = '*.png';
inMat  = ['/BS/joint-multicut-2/work/VSB-fasterRCNN/' dataChoice '/'];
outMat = ['/BS/joint-multicut-2/work/VSB-fasterRCNN/deeplab/' dataChoice '/'];
folder = dir(dataDir);
folder(1:2)=[];


% loop for each image folder
for img_index=53%:size(folder,1)
    imgDir = [folder(img_index).name '/'];
    Files=dir([dataDir imgDir Type]);
    LengthFiles = length(Files);
    boxList = dir([inMat imgDir '*.mat']);
    if(~exist([outMat imgDir],'dir'))
        mkdir([outMat imgDir]) ;
    end
    for imgnum=1:LengthFiles
        % setting path
        imgName = Files(imgnum).name;
        filename = [dataDir imgDir imgName];
        boxName = boxList(imgnum).name;
        matName = imgName; matName(end-3:end)='.mat';
        outbox = [outMat imgDir matName];
        % based on deeplab 
        I = imread(filename);
        [img_height, img_width, ~] = size(I);
        temp = load([inMat imgDir boxName]);
        Bbox = temp.Boxes2save;
        %get segmentation for each Bbox
        segment = cell(1,size(Bbox,1));
        for box_index=1:size(Bbox,1)
            original_box = Bbox(box_index,:);
            cur_box = zoom_box(original_box,img_height,img_width);
            %rect = [cur_box(1) cur_box(2) cur_box(3)-cur_box(1) cur_box(4)-cur_box(2)];
            img = I(round(cur_box(2)):round(cur_box(4)),round(cur_box(1)):round(cur_box(3)),:);
            input_data = preprocess_image(double(img), config.im_sz);
            cnn_output = caffe('forward', input_data);
            result = permute(cnn_output{1}, [2, 1, 3]);
            [~, result] = max(result, [], 3);
            result = uint8(result) - 1;
            segImg = back_image(result, img_height, img_width, cur_box,original_box);
            segment{1,box_index} = segImg;
        end
        save(outbox,'segment');
        % figure out every 20 frame
        if mod(imgnum-1,20)==0 && size(Bbox,1)
            for j=1:size(Bbox,1)
            h = figure(imgnum);clf;
            set(h,'Visible','off');
            imagesc(segment{1,j});
            hold on;
            rectangle('Position', [Bbox(j,1),Bbox(j,2),Bbox(j,3)-Bbox(j,1),Bbox(j,4)-Bbox(j,2)]);
            id = sprintf('_%.2d',j);
            img_save = [outMat imgDir imgName(1:end-4) id '.ppm'];
            fprintf('saving %s\n', img_save);
            print(h,'-dpng', img_save);
            close(h);
            end
        end
        fprintf('frame%d processed\n',imgnum);
    end
end





