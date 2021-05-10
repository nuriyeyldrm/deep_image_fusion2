function[fusedImage] = fusion_PCA(image1,image2)
% by VPS Naidu, MSDF Lab, NAL, Bangalore
% Image Fusion using PCA algorithm
% creating the covariance matrix
C = cov([image1(:) image2(:)]);
% Obtain the Eigenvectors and Eigenvalues from the covariance matrix
[V, D] = eig(C);
if D(1,1) >= D(2,2)
    pca = V(:,1)./sum(V(:,1));
else
    pca = V(:,2)./sum(V(:,2));
end

fusedImage = pca(1)*image1 + pca(2)*image2;
