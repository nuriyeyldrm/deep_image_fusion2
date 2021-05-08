fileID = fopen('ratio.txt','w+t');

for i=1:21
    index = i

    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];

    image1 = imread(path1);
    image2 = imread(path2);
    image1 = im2double(image1);
    image2 = im2double(image2);
    
    % Highpass filter test image
    [ratio_pca, nabf_pca] = find_best_ratio_pca(image1, image2);
    [ratio_dwt, nabf_dwt] = find_best_ratio_dwt(image1, image2);
    
    if nabf_dwt < nabf_pca
        fprintf(fileID,'%3.1f, %3s\n', ratio_dwt, 'DWT');
    else
        fprintf(fileID,'%3.1f, %3s\n', ratio_pca, 'PCA');
    end   
end
 
fclose(fileID);



