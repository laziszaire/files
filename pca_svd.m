clear
rng(1);
X = randn(10,100);
X = X - mean(X);
X = bsxfun(@rdivide,X,std(X));
K = X*X';
[V,D] = eig(K);
S = K*V;
d = diag(D);
%²î¸ö³£Êý±¶
w  = X'*V;
const = sqrt(sum(w.^2));
S_ = bsxfun(@rdivide,S,const);

%svd
[coeff,score,latent,tsquared,explained,mu] = pca(X);
