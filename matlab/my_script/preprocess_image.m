function preprocessed_img= preprocess_image(img, img_sz)
new_img=zeros(img_sz,img_sz,3);
[img_height, img_width, ~] = size(img);
[Maxsize,ind] = max([img_height, img_width]);
if Maxsize>img_sz
    %image larger than the input size
    switch ind
        case 1
            scale = [img_sz img_sz*img_width/img_height];
        case 2
            scale = [img_sz*img_height/img_width img_sz];
    end
    img = imresize(img, scale, 'bilinear');
    new_img(1:size(img,1),1:size(img,2),:)=img;
else
    new_img(1:img_height,1:img_width,:)=img;
end


meanImg = [104.008, 116.669, 122.675]; % order = bgr
meanImg = repmat(meanImg, [img_sz^2,1]);
meanImg = reshape(meanImg, [img_sz, img_sz, 3]); 

%crop = imresize(img, [img_sz img_sz], 'bilinear'); % resize cropped image
crop = new_img(:,:,[3 2 1]) - meanImg; % convert color channer rgb->bgr and subtract mean 
preprocessed_img = {single(permute(crop, [2 1 3]))}; 

end