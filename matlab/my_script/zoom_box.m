function cur_box = zoom_box(original_box,img_height,img_width)
% enlarge the crop of image with 1.2
zoom_factor = 1.2;
% input coordinate
x1 = original_box(1);
y1 = original_box(2);
x2 = original_box(3);
y2 = original_box(4);
w = x2-x1;
h = y2-y1;
% output coordinate
dh = h*(zoom_factor-1)/2;
dw = w*(zoom_factor-1)/2;
X1 = max(x1-dw,1);
X2 = min(x2+dw,img_width);
Y1 = max(y1-dh,1);
Y2 = min(y2+dh,img_height);

cur_box=[X1,Y1,X2,Y2];
end