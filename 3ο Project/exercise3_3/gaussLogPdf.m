function logp = gaussLogPdf(X, mu, Sigma)
    % Log-density of a multivariate Gaussian, evaluated at every row of X.
    %
    % INPUT:
    %   X: (N X d) N example vectors of dimension d
    %   mu: (1 X d) mean vector
    %   Sigma: (d X d) covariance matrix
    %
    % OUTPUT:
    %   logp: (N X 1) log N(x_n | mu, Sigma)
    %
    % Working with logarithms avoids the underflow of the plain density,
    % and the Cholesky factor avoids computing inv() and det() directly.

    d = size(X, 2);
    Xc = bsxfun(@minus, X, mu);          % x - mu
    L = chol(Sigma, 'lower');            % Sigma = L*L'
    z = L \ Xc';                         % so that sum(z.^2) = (x-mu)'*inv(Sigma)*(x-mu)
    maha = sum(z.^2, 1)';                % squared Mahalanobis distances
    logDet = 2 * sum(log(diag(L)));      % log(det(Sigma))
    logp = -0.5 * (d*log(2*pi) + logDet + maha);
end
