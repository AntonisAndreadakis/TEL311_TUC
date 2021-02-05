function v = fisherLinearDiscriminant(X1, X2)

    m1 = size(X1, 1);
    m2 = size(X2, 1);

    mu1 = mean(X1); % mean value of X1
    mu2 = mean(X2);% mean value of X2
    
    X1_new = [X1(:,1)-mu1(1);X1(:,2)-mu1(2)];
    X2_new = [X2(:,1)-mu2(1);X2(:,2)-mu2(2)];

    S1 = (X1_new)' *(X1_new);% scatter matrix of X1
    S2 = (X2_new)' *(X2_new);% scatter matrix of X2

    Sw = S1 + S2;% Within class scatter matrix
    
    %inv(Sw);

    v = (mu1-mu2)/Sw; %optimal direction for maximum class separation 

    v = v/norm(v); % return a vector of unit norm
    
end