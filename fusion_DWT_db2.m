function[fusedImage] = fusion_DWT_db2(image1, image2, k)
% Image fusion
fusedImage = wfusimg(image1, image2,'db2', k, 'mean', 'mean');