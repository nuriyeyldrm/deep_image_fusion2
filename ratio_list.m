ratioList = [];

for i=1:21
    index = i

    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];
    fuse_path = ['./fused',num2str(index),'_VGG_multiLayers.png'];

    image1 = imread(path1);
    image2 = imread(path2);
    image1 = im2double(image1);
    image2 = im2double(image2);
    
    
    % Highpass filter test image
    ratio = find_best_ratio(image1, image2);
    ratioList(end+1) = ratio;
end

writematrix(ratioList)
type 'ratioList.txt'
disp(ratioList);
