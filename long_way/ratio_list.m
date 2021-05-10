fileID = fopen('ratios.txt','w+t');

n = 21;
time = zeros(n,1);
for i=1:1
    index = i

    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];
    fuse_path = ['./fused_ir_vis/fused',num2str(index),'.png'];

    image1 = imread(path1);
    image2 = imread(path2);
    image1 = im2double(image1);
    image2 = im2double(image2);
    
    tic;
    [ratio_pca, ratio_dwt, nabf_pca, nabf_dwt, fused_wpca, fused_wdwt] =...
        find_best_ratio(image1, image2);
    
    time(i) = toc;
    if nabf_dwt < nabf_pca
        fprintf(fileID,'%3.1f, %3s\n', ratio_dwt, 'DWT');
        figure;imshow(fused_wdwt);
        imwrite(fused_wdwt, fuse_path, 'png');
    else
        fprintf(fileID,'%3.1f, %3s\n', ratio_pca, 'PCA');
        figure;imshow(fused_wpca);
        imwrite(fused_wpca, fuse_path, 'png');
    end   
end
 
fclose(fileID);



