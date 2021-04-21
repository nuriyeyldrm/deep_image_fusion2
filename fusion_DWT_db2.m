function[fusedImage] = fusion_DWT_db2(fitsImage1,fitsImage2,k)
% Image fusion
fusedImage = wfusimg(fitsImage1,fitsImage2,'db2',k,'mean','mean');