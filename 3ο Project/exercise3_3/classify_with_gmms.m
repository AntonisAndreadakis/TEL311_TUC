function y = classify_with_gmms(x, Pm, M, S)
    % Assign an example to the most probable GMM component.
    %
    % INPUT:
    %   x: (1 X d) An example vector of dimension d
    %   Pm: (K X 1) mixing probabilities
    %   M: (K X d) mean vectors
    %   S: (K X d X d) covariance matrices
    %
    % OUTPUT:
    %   y:  The class index

    [K, d] = size(M);
    logPost = zeros(K, 1);   % log( Pm(k) * N(x | M(k), S(k)) ), up to a constant
    for k = 1:K
        Sk = reshape(S(k, :, :), d, d);
        logPost(k) = log(Pm(k)) + gaussLogPdf(x, M(k, :), Sk);
    end
    [~, y] = max(logPost);
end
