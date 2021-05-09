function [ratio, nabf] = find_best_ratio_pca(image1, image2)

%load vgg19
net = load('imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

tmp = 1000;
for i=1:+0.1:2

    npd = 16;
    fltlmbd = 5;

    [I_lrr1, I_saliency1] = lowpass(image1, fltlmbd, npd, i);
    [I_lrr2, I_saliency2] = lowpass(image2, fltlmbd, npd, i);

    img1 = fusion_PCA(image1, I_saliency1);
    img2 = fusion_PCA(image2, I_saliency2);


    %% fuison lrr parts
    F_lrr = (I_lrr1+I_lrr2)/2;

    %% fuison saliency parts use VGG19
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

    l1_featrues_relu1_a = extract_l1_feature(out_relu1_1_a);
    l1_featrues_relu1_b = extract_l1_feature(out_relu1_1_b);
    l1_featrues_img1 = resize(img1);
    l1_featrues_img2 = resize(img2);

    [F_saliency_relu1, l1_featrues_relu1_ave_a, l1_featrues_relu1_ave_b] = ...
                fusion_strategy(l1_featrues_relu1_a, l1_featrues_relu1_b,...
                I_saliency1, I_saliency2, l1_featrues_img1, l1_featrues_img2,...
                img1, img2, unit_relu1_1);

    %% relu2_1
    out_relu2_1_a = res_a(7).x;
    out_relu2_1_b = res_b(7).x;
    unit_relu2_1 = 2;

    l1_featrues_relu2_a = extract_l1_feature(out_relu2_1_a);
    l1_featrues_relu2_b = extract_l1_feature(out_relu2_1_b);

    [F_saliency_relu2, l1_featrues_relu2_ave_a, l1_featrues_relu2_ave_b] = ...
                fusion_strategy(l1_featrues_relu2_a, l1_featrues_relu2_b,...
                I_saliency1, I_saliency2, l1_featrues_img1, l1_featrues_img2,...
                img1, img2, unit_relu2_1);

    %% relu3_1
    out_relu3_1_a = res_a(12).x;
    out_relu3_1_b = res_b(12).x;
    unit_relu3_1 = 4;

    l1_featrues_relu3_a = extract_l1_feature(out_relu3_1_a);
    l1_featrues_relu3_b = extract_l1_feature(out_relu3_1_b);

    [F_saliency_relu3, l1_featrues_relu3_ave_a, l1_featrues_relu3_ave_b] = ...
                fusion_strategy(l1_featrues_relu3_a, l1_featrues_relu3_b,...
                I_saliency1, I_saliency2, l1_featrues_img1, l1_featrues_img2,...
                img1, img2, unit_relu3_1);

    %% relu4_1
    out_relu4_1_a = res_a(21).x;
    out_relu4_1_b = res_b(21).x;
    unit_relu4_1 = 8;

    l1_featrues_relu4_a = extract_l1_feature(out_relu4_1_a);
    l1_featrues_relu4_b = extract_l1_feature(out_relu4_1_b);

    [F_saliency_relu4, l1_featrues_relu4_ave_a, l1_featrues_relu4_ave_b] = ...
                fusion_strategy(l1_featrues_relu4_a, l1_featrues_relu4_b,...
                I_saliency1, I_saliency2, l1_featrues_img1, l1_featrues_img2,...
                img1, img2, unit_relu4_1);

    %% fusion strategy

    F_saliency = max(F_saliency_relu1, F_saliency_relu2);
    F_saliency = max(F_saliency, F_saliency_relu3);
    F_saliency = max(F_saliency, F_saliency_relu4);

    fusion_im = F_lrr + F_saliency;
    imwrite(fusion_im,'fuse.png','png');
    fused = imread('fuse.png');
    fused = im2double(fused);
    NABF = analysis_nabf(fused,image1,image2)
   
    if NABF < tmp
        ratio = i;
        tmp = NABF; 
        nabf = NABF;
    end
    
    delete('fuse.png')
end






