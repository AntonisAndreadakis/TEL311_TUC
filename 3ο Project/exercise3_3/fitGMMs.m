function [Pm, M, S] = fitGMMs(X, K)
    % Fit a Gaussian Mixture Model with the EM algorithm.
    %
    % INPUT:
    %   X: (N X d) N example vectors of dimension d
    %   K: number of GMM components
    %
    % OUTPUT:
    %   Pm: (K X 1) mixing probabilities
    %   M: (K X d) mean vectors
    %   S: (K X d X d) covariance matrices

    [N, d] = size(X);
    maxIters = 200;      % upper bound on EM iterations
    tol = 1e-6;          % stop when the log-likelihood improves less than this
    reg = 1e-6*eye(d);   % small ridge that keeps every covariance invertible

    % ---- Initialize using kmeans: every cluster becomes one component ----
    labels = kmeans(X, K, 'Replicates', 5);
    Pm = zeros(K, 1);
    M = zeros(K, d);
    S = zeros(K, d, d);
    for k = 1:K
        Xk = X(labels == k, :);
        Pm(k) = size(Xk, 1) / N;
        M(k, :) = mean(Xk, 1);
        S(k, :, :) = cov(Xk) + reg;
    end

    prevLogLik = -Inf;
    for its = 1:maxIters
        % ---- E-step: responsibilities G(n,k) = P(component k | x_n) ----
        logJoint = zeros(N, K);   % log( Pm(k) * N(x_n | M(k), S(k)) )
        for k = 1:K
            Sk = reshape(S(k, :, :), d, d);
            logJoint(:, k) = log(Pm(k)) + gaussLogPdf(X, M(k, :), Sk);
        end
        % log-sum-exp over the components, for numerical stability
        maxLog = max(logJoint, [], 2);
        logPx = maxLog + log(sum(exp(bsxfun(@minus, logJoint, maxLog)), 2));
        G = exp(bsxfun(@minus, logJoint, logPx));

        % ---- Convergence check on the data log-likelihood ----
        logLik = sum(logPx);
        if abs(logLik - prevLogLik) < tol * abs(logLik)
            break;
        end
        prevLogLik = logLik;

        % ---- M-step: maximum-likelihood updates weighted by G ----
        Nk = sum(G, 1)';                       % effective samples per component
        Pm = Nk / N;
        M = bsxfun(@rdivide, G' * X, Nk);
        for k = 1:K
            Xc = bsxfun(@minus, X, M(k, :));
            Sk = (Xc' * bsxfun(@times, Xc, G(:, k))) / Nk(k);
            S(k, :, :) = (Sk + Sk')/2 + reg;   % keep it exactly symmetric
        end
    end
end
