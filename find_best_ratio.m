function [ratio_pca, ratio_dwt, nabf_pca, nabf_dwt, fused_with_pca, fused_with_dwt] = ...
    find_best_ratio(image1, image2)

%load vgg19
net = load('imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net);

tmp_pca = 1000;
tmp_dwt = 1000;

for i=1:+0.1:2

    npd = 16;
    fltlmbd = 5;
    
    [I_lrr1, I_saliency1] = lowpass(image1, fltlmbd, npd, i);
    [I_lrr2, I_saliency2] = lowpass(image2, fltlmbd, npd, i);

    img1 = fusion_PCA(image1, I_saliency1);
    img2 = fusion_PCA(I_saliency2, image2);
    img = fusion_DWT_db2(image2, I_saliency1, 5);

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

    l1_features_relu1_a = extract_l1_feature(out_relu1_1_a);
    l1_features_relu1_b = extract_l1_feature(out_relu1_1_b);
    resizes_img1 = resize(img1);
    resizes_img2 = resize(img2);
    resizes_img = resize(img);
    
    [F_saliency_relu_pca1, l1_features_relu1_ave_pcaa, l1_features_relu1_ave_pcab,...
        resizes1_img1_ave_pcac, resizes1_img2_ave_pcad] = ...
        fusion_strategy_w_pca(l1_features_relu1_a, l1_features_relu1_b,...
        I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
        img1, img2, unit_relu1_1);

    [F_saliency_relu_dwt1, l1_features_relu1_ave_dwta, l1_features_relu1_ave_dwtb,...
        resizes1_img_ave_dwtc] = fusion_strategy_w_dwt(l1_features_relu1_a,...
        l1_features_relu1_b, I_saliency1, I_saliency2, resizes_img, img,...
        unit_relu1_1);
    
    %% relu2_1
    out_relu2_1_a = res_a(7).x;
    out_relu2_1_b = res_b(7).x;
    unit_relu2_1 = 2;

    l1_features_relu2_a = extract_l1_feature(out_relu2_1_a);
    l1_features_relu2_b = extract_l1_feature(out_relu2_1_b);

    [F_saliency_relu_pca2, l1_features_relu2_ave_pcaa, l1_features_relu2_ave_pcab,...
        resizes2_img1_ave_pcac, resizes2_img2_ave_pcad] = ...
        fusion_strategy_w_pca(l1_features_relu2_a, l1_features_relu2_b,...
        I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
        img1, img2, unit_relu2_1);

    [F_saliency_relu_dwt2, l1_features_relu2_ave_dwta, l1_features_relu2_ave_dwtb,...
        resizes2_img_ave_dwtc] = fusion_strategy_w_dwt(l1_features_relu2_a,...
        l1_features_relu2_b, I_saliency1, I_saliency2, resizes_img, img,...
        unit_relu2_1);

    %% relu3_1
    out_relu3_1_a = res_a(12).x;
    out_relu3_1_b = res_b(12).x;
    unit_relu3_1 = 4;

    l1_features_relu3_a = extract_l1_feature(out_relu3_1_a);
    l1_features_relu3_b = extract_l1_feature(out_relu3_1_b);
    
    [F_saliency_relu_pca3, l1_features_relu3_ave_pcaa, l1_features_relu3_ave_pcab,...
        resizes3_img1_ave_pcac, resizes3_img2_ave_pcad] = ...
        fusion_strategy_w_pca(l1_features_relu3_a, l1_features_relu3_b,...
        I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
        img1, img2, unit_relu3_1);

    [F_saliency_relu_dwt3, l1_features_relu3_ave_dwta, l1_features_relu3_ave_dwtb,...
        resizes3_img_ave_dwtc] = fusion_strategy_w_dwt(l1_features_relu3_a,...
        l1_features_relu3_b, I_saliency1, I_saliency2, resizes_img, img,...
        unit_relu3_1);

    %% relu4_1
    out_relu4_1_a = res_a(21).x;
    out_relu4_1_b = res_b(21).x;
    unit_relu4_1 = 8;

    l1_features_relu4_a = extract_l1_feature(out_relu4_1_a);
    l1_features_relu4_b = extract_l1_feature(out_relu4_1_b);
    
    [F_saliency_relu_pca4, l1_features_relu4_ave_pcaa, l1_features_relu4_ave_pcab,...
        resizes4_img1_ave_pcac, resizes4_img2_ave_pcad] = ...
        fusion_strategy_w_pca(l1_features_relu4_a, l1_features_relu4_b,...
        I_saliency1, I_saliency2, resizes_img1, resizes_img2,...
        img1, img2, unit_relu4_1);

    [F_saliency_relu_dwt4, l1_features_relu4_ave_dwta, l1_features_relu4_ave_dwtb,...
        resizes4_img_ave_dwtc] = fusion_strategy_w_dwt(l1_features_relu4_a,...
        l1_features_relu4_b, I_saliency1, I_saliency2, resizes_img, img,...
        unit_relu4_1);

    %% fusion strategy with pca

    F_saliency_pca = max(F_saliency_relu_pca1, F_saliency_relu_pca2);
    F_saliency_pca = max(F_saliency_pca, F_saliency_relu_pca3);
    F_saliency_pca = max(F_saliency_pca, F_saliency_relu_pca4);

    fused_img_wpca = F_lrr + F_saliency_pca;
    imwrite(fused_img_wpca,'fused_wpca.png','png');
    fused_wpca = imread('fused_wpca.png');
    fused_wpca = im2double(fused_wpca);
    NABF_pca = analysis_nabf(fused_wpca,image1,image2);
   
    if NABF_pca < tmp_pca
        ratio_pca = i;
        fused_with_pca = fused_img_wpca;
        tmp_pca = NABF_pca; 
        nabf_pca = NABF_pca;
    end
    
    delete('fused_wpca.png')
    
    %% fusion strategy with dwt
    F_saliency_dwt = max(F_saliency_relu_dwt1, F_saliency_relu_dwt2);
    F_saliency_dwt = max(F_saliency_dwt, F_saliency_relu_dwt3);
    F_saliency_dwt = max(F_saliency_dwt, F_saliency_relu_dwt4);

    fused_img_wdwt = F_lrr + F_saliency_dwt;
    imwrite(fused_img_wdwt,'fused_wdwt.png','png');
    fused_wdwt = imread('fused_wdwt.png');
    fused_wdwt = im2double(fused_wdwt);
    NABF_dwt = analysis_nabf(fused_wdwt,image1,image2);
   
    if NABF_dwt < tmp_dwt
        ratio_dwt = i;
        fused_with_dwt = fused_img_wdwt;
        tmp_dwt = NABF_dwt; 
        nabf_dwt = NABF_dwt;
    end
    
    delete('fused_wdwt.png')
end