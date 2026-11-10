function A = myLDA(Samples, Labels, NewDim)
% Input:
%   Samples: The Data Samples (one sample per row)
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
% Output:
%   A: (NumFeatures x NewDim) the LDA projection vectors, one per column

    [NumSamples, NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if (NumSamples ~= NumLabels)
        error('Number of Samples are not the same with the Number of Labels.');
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes

    P = zeros(NumClasses, 1);              % class prior probabilities
    mu = zeros(NumClasses, NumFeatures);   % class means
    Sw = zeros(NumFeatures, NumFeatures);  % within class scatter matrix

    %For each class i
    %Find the necessary statistics
    for i = 1 : NumClasses
        Xi = Samples(Labels == Classes(i), :);
        %Calculate the Class Prior Probability
        P(i) = size(Xi, 1)/NumSamples;
        %Calculate the Class Mean
        mu(i, :) = mean(Xi, 1);
        %Calculate the Within Class Scatter Matrix
        Sw = Sw + P(i)*cov(Xi, 1);
    end

    %Calculate the Global Mean
    m0 = P' * mu;

    %Calculate the Between Class Scatter Matrix
    Sb = zeros(NumFeatures, NumFeatures);
    for i = 1 : NumClasses
        Sb = Sb + P(i) * (mu(i, :) - m0)' * (mu(i, :) - m0);
    end

    %Perform Eigendecomposition of inv(Sw)*Sb
    %(solved as the generalized problem Sb*v = lambda*Sw*v)
    [V, D] = eig(Sb, Sw);
    eigenval = real(diag(D));
    [~, ind] = sort(eigenval, 1, 'descend');
    eigenvec = real(V(:, ind));

    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
    A = eigenvec(:, 1:NewDim);  % Return the LDA projection vectors

    % normalize every projection vector to unit length
    A = bsxfun(@rdivide, A, sqrt(sum(A.^2, 1)));
end
