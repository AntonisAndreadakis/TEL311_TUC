function v = fisherLinearDiscriminant(X1, X2)
    % Fisher's Linear Discriminant for two classes.
    %   X1, X2: the samples of each class, one sample per row
    %   v: unit (row) vector of the direction that best separates the classes

    mu1 = mean(X1, 1); % mean value of X1
    mu2 = mean(X2, 1); % mean value of X2

    % subtract the class mean from every sample (keep one sample per row)
    X1_new = bsxfun(@minus, X1, mu1);
    X2_new = bsxfun(@minus, X2, mu2);

    S1 = (X1_new)' * (X1_new); % scatter matrix of X1
    S2 = (X2_new)' * (X2_new); % scatter matrix of X2

    Sw = S1 + S2; % Within class scatter matrix

    v = (Sw \ (mu1 - mu2)')'; % optimal direction: inv(Sw)*(mu1-mu2)

    v = v/norm(v); % return a vector of unit norm

end
