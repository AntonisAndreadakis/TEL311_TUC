function [U, S] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = myPCA(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE GOES HERE ======================
% Apply PCA by computing the eigenvectors and eigenvalues of the covariance matrix. 
%

R = cov(X);
[V,D] = eig(R);
eigenval = diag(D);
[eigenval, ind] = sort(eigenval, 1, 'descend');
eigenvec = V(:, ind);

U = eigenvec;
S = diag(eigenval);

% =========================================================================

end
