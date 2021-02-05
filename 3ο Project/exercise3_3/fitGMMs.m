function [Pm, M, S] = fitGMMs(X, K)
    % INPUT:
    %   X: (N X d) N example vectors of dimension d
    %   K: number of GMM components
    %
    % OUTPUT:
    %   Pm: (K X 1) mixing probabilities
    %   M: (K X d) mean vectors
    %   S: (K X d X d) covariance matrices 
      
    [N, d] = size(X);
    
    % Initialize using kmeans. You may use the matlab kmeans.
    [~,c] = kmeans(X,K);
    % your code here        
    G = zeros(N, K); % likelihood of classes given features
    [pPi,pMiu,pSigma] = myFunc();
    
    for its = 1:100
        % E-step. Calculation of likelihood 'G'
        % your code here     
        p = real(gaussian());  % size=Px(N, K);
        p(isnan(p))=0;        % kill NaN
        p(isnan(p))=0;        % kill inf
        % Likelihood normalization (of G)
        G = p.*repmat(pPi, N, 1);      %numerator  = pi(k) * N(xi | pMiu(k), pSigma(k))
        G = G./repmat(sum(G, 2), 1, K); %denominator = pi(j) *N(xi | pMiu(j), pSigma(j))sum over j
        % M-step. Parameter updates
        % your code here        
        n = sum(G, 1);    % number of samples in each cluster
        n(isnan(n))=0;    % kill NaN
        n(isinf(n))=0;    % kill inf
        pMiu = diag(1./n)*G'*X2; % update through MLE
        pPi = n/N;
        for t = 1:K 
            X2 = X-repmat(pMiu(t, :), N, 1);
            pSigma(:,:,t) = (X2'*(diag(G(:, t))*X2))/n(t);
        end
        Pm = pPi; 
        M = pMiu;
        S = pSigma;
    end
    
    function [pPi,pMiu,pSigma] = myFunc()
        pPi = zeros(1, K);
        pMiu = c; 
        pSigma = zeros(d, d, K); 
        distmat = repmat(sum(X.*X, 2), 1, K) + repmat(sum(pMiu.*pMiu, 2)', N, 1) - 2*X*pMiu';
        [~, labels] = min(distmat, [], 2);%Return the minimum from each row
        for k=1:K
            y = X(labels == k, :);
            pPi(k) = size(y, 1)/N;
            pSigma(:, :, k) = cov(y);
         end
    end

    function p = gaussian()
    %Gaussian posterior probability 
        p = zeros(N, 1);
        for k = 1:1
            X2 = X-repmat(pMiu(k, :), N, 1); %x-m
            inv_S = inv(pSigma(:, :, k));
            np = sum((X2*inv_S).*X2, 2);
            var = (2*pi)^(-d/2)*sqrt(det(inv_S));
            p(:, 1) = var*exp(-0.5*np);
        end
    end
end
