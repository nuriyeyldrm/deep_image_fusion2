%load vgg19
net = load('imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

n = 21;
time = zeros(n,1);
for i=1:1
    index = i

    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];
    fuse_path = ['./fused_ir_vis/fused',num2str(index),'.png'];

    image1 = imread(path1);
    image2 = imread(path2);
    figure;imshow(image1);
    figure;imshow(image2);
    image1 = im2double(image1);
    image2 = im2double(image2);

    tic;

    fileID = fopen('ratio.txt');
    C = textscan(fileID,'%f, %s',1,'delimiter','\n', 'headerlines',i-1);
    fclose(fileID);
    
    ratio = C{1}
    fcriteria = strjoin(C{2})
    
    npd = 16;
    fltlmbd = 5;
    
    [I_lrr1, I_saliency1] = lowpass(image1, fltlmbd, npd, ratio);
    [I_lrr2, I_saliency2] = lowpass(image2, fltlmbd, npd, ratio);

    img1 = fusion_PCA(image1, I_saliency1);
    img2 = fusion_PCA(I_saliency2, image2);
    img = fusion_DWT_db2(image2, I_saliency1, 5);

    %% fuison lrr parts
    F_lrr = (I_lrr1+I_lrr2)/2;

    %% fuison saliency parts use VGG19
    disp('VGG19-saliency');
    saliency_a = make_3c(I_saliency1);
    saliency_b = make_3c(I_saliency2);
    saliency_a = single(saliency_a) ; % note: 255 range
    saliency_b = single(saliency_b) ; % note: 255 range

    res_a = vl_simplenn(net, saliency_a);
    res_b = vl_simplenn(net, saliency_b);

    %% relu1_1
    out_relu1_1_a = res_a(2).x;
    out_relu1_1_b = res_b(2).x;
    unit_relu1_1 = 1;

    l1_features_relu1_a = extract_l1_feature(out_relu1_1_a);
    l1_features_relu1_b = extract_l1_feature(out_relu1_1_b);
    resizes_img1 = resize(img1);
    resizes_img2 = resize(img2);
    resizes_img = resize(img);
    
    if fcriteria == 'PCA'
        [F_saliency_relu1, l1_features_relu1_ave_a, l1_features_relu1_ave_b,...
            resizes1_img1_ave_c, resizes1_img2_ave_d] = ...
            fusion_strategy_w_pca(l1_features_relu1_a, l1_features_relu1_b,...
            I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
            img1, img2, unit_relu1_1);
    else
        [F_saliency_relu1, l1_features_relu1_ave_a, l1_features_relu1_ave_b,...
            resizes1_img_ave_c] = fusion_strategy_w_dwt(l1_features_relu1_a,...
            l1_features_relu1_b, I_saliency1, I_saliency2, resizes_img, img,...
            unit_relu1_1);
    end
    
    %% relu2_1
    out_relu2_1_a = res_a(7).x;
    out_relu2_1_b = res_b(7).x;
    unit_relu2_1 = 2;

    l1_features_relu2_a = extract_l1_feature(out_relu2_1_a);
    l1_features_relu2_b = extract_l1_feature(out_relu2_1_b);
    
    if fcriteria == 'PCA'
        [F_saliency_relu2, l1_features_relu2_ave_a, l1_features_relu2_ave_b,...
            resizes2_img1_ave_c, resizes2_img2_ave_d] = ...
            fusion_strategy_w_pca(l1_features_relu2_a, l1_features_relu2_b,...
            I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
            img1, img2, unit_relu2_1);
    else
        [F_saliency_relu2, l1_features_relu2_ave_a, l1_features_relu2_ave_b,...
            resizes2_img_ave_c] = fusion_strategy_w_dwt(l1_features_relu2_a,...
            l1_features_relu2_b, I_saliency1, I_saliency2, resizes_img, img,...
            unit_relu2_1);
    end

    %% relu3_1
    out_relu3_1_a = res_a(12).x;
    out_relu3_1_b = res_b(12).x;
    unit_relu3_1 = 4;

    l1_features_relu3_a = extract_l1_feature(out_relu3_1_a);
    l1_features_relu3_b = extract_l1_feature(out_relu3_1_b);
    
    if fcriteria == 'PCA'
        [F_saliency_relu3, l1_features_relu3_ave_a, l1_features_relu3_ave_b,...
            resizes3_img1_ave_c, resizes3_img2_ave_d] = ...
            fusion_strategy_w_pca(l1_features_relu3_a, l1_features_relu3_b,...
            I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
            img1, img2, unit_relu3_1);
    else
        [F_saliency_relu3, l1_features_relu3_ave_a, l1_features_relu3_ave_b,...
            resizes3_img_ave_c] = fusion_strategy_w_dwt(l1_features_relu3_a,...
            l1_features_relu3_b, I_saliency1, I_saliency2, resizes_img, img,...
            unit_relu3_1);
    end

    %% relu4_1
    out_relu4_1_a = res_a(21).x;
    out_relu4_1_b = res_b(21).x;
    unit_relu4_1 = 8;

    l1_features_relu4_a = extract_l1_feature(out_relu4_1_a);
    l1_features_relu4_b = extract_l1_feature(out_relu4_1_b);
    
    if fcriteria == 'PCA'
        [F_saliency_relu4, l1_features_relu4_ave_a, l1_features_relu4_ave_b,...
            resizes4_img1_ave_c, resizes4_img2_ave_d] = ...
            fusion_strategy_w_pca(l1_features_relu4_a, l1_features_relu4_b,...
            I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
            img1, img2, unit_relu4_1);
    else
        [F_saliency_relu4, l1_features_relu4_ave_a, l1_features_relu4_ave_b,...
            resizes4_img_ave_c] = fusion_strategy_w_dwt(l1_features_relu4_a,...
            l1_features_relu4_b, I_saliency1, I_saliency2, resizes_img, img,...
            unit_relu4_1);
    end

    %% fusion strategy

    F_saliency = max(F_saliency_relu1, F_saliency_relu2);
    F_saliency = max(F_saliency, F_saliency_relu3);
    F_saliency = max(F_saliency, F_saliency_relu4);

    fused_img = F_lrr + F_saliency;
    time(i) = toc;
    figure;imshow(fused_img);

    imwrite(fused_img,fuse_path,'png');
end