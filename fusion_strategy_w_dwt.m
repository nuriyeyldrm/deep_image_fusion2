function [gen, ave1, ave2] = fusion_strategy_w_dwt(features_a, features_b, source_a, source_b, img, img_d, unit)

[m,n] = size(features_a);
[m1,n1] = size(source_a);
ave_temp1 = zeros(m1,n1);
ave_temp2 = zeros(m1,n1);
ave_temp3 = zeros(m1,n1);

weight_ave_temp1 = zeros(m1,n1);
weight_ave_temp2 = zeros(m1,n1);
weight_ave_temp3 = zeros(m1,n1);

for i=2:m-1
    for j=2:n-1
        A1 = sum(sum(features_a(i-1:i+1,j-1:j+1)))/9;
        A2 = sum(sum(features_b(i-1:i+1,j-1:j+1)))/9;
        A3 = sum(sum(img(i-1:i+1,j-1:j+1)))/9;
        
        % weight average
        weight_ave_temp1(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A1/(A1+A2+A3);
        weight_ave_temp2(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A2/(A1+A2+A3);
        weight_ave_temp3(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A3/(A1+A2+A3);

        ave_temp1(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A1;
        ave_temp2(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A2;
        ave_temp3(((i-2)*unit+1):((i-1)*unit),((j-2)*unit+1):((j-1)*unit)) = A3;
    end
end
% figure;imshow(temp_mask);
weight_ave_temp1 = weight_ave_temp1(1:m1,1:n1);
weight_ave_temp2 = weight_ave_temp2(1:m1,1:n1);
weight_ave_temp3 = weight_ave_temp3(1:m1,1:n1);

gen = source_a.*weight_ave_temp1 + source_b.*weight_ave_temp2 + img_d.*weight_ave_temp3;

ave1 = ave_temp1;
ave2 = ave_temp2;
ave3 = ave_temp3;
end