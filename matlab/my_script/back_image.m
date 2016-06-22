function segImg = back_image(result, img_height, img_width, cur_box,original_box)
result = boolean(result);
segImg = zeros(img_height, img_width);
x1 = round(cur_box(1));
y1 = round(cur_box(2));
x2 = round(cur_box(3));
y2 = round(cur_box(4));
W = x2-x1+1;
H = y2-y1+1;
[Maxsize,ind]=max([H W]);
if Maxsize<=size(result,1)
    % the box is smaller than result size, extract directly
    seg = result(1:H,1:W);
else
    % the box is larger than result size, resize the result img
    switch ind
        case 1
            seg = imresize(result, H/size(result,1), 'bilinear');
        case 2
            seg = imresize(result, W/size(result,2), 'bilinear');
    end
    seg = seg(1:H,1:W);
end
segImg(y1:y2,x1:x2)=seg;
% shrink box to the size before zoom
box_template = zeros(img_height, img_width);
% ceil and floor to ensure that the seg is completely in the box
box_template(ceil(original_box(2)):floor(original_box(4)),ceil(original_box(1)):floor(original_box(3)))=1;
segImg = segImg&box_template;