function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
	
	[NumSamples, NumFeatures] = size(Samples);
    A = zeros(NumFeatures, NewDim);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels)
        then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes
    Sw = zeros(size(Samples, 2), size(Samples, 2));
    St = cov(Samples);
    m0 = 0;
    %For each class i
	%Find the necessary statistics
    for i = 1 : NumClasses
    %Calculate the Class Prior Probability
	P = sum(Labels==i-1)/NumLabels;
    %Calculate the Class Mean
	mu(i,:) = mean(Samples(Labels==i-1, :));
    %Calculate the Within Class Scatter Matrix
    Sw = Sw + (P*cov(Samples(Labels==i-1, :)));
    %Calculate the Global Mean
    m0 = mean(NumFeatures*mu(i,:))+m0;
    %Calculate the Between Class Scatter Matrix
    Sb = St - Sw;
    end
    %Eigen matrix EigMat=inv(Sw)*Sb
    Eigmat = inv(Sw)*Sb;
    [V,D] = eig(Eigmat);
    %Perform Eigendecomposition
    eigenval = diag(D);
    [~,ind] = sort(eigenval, 1, 'descend');
    eigenvec = V(:,ind);

    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A = eigenvec(:,1:2)';  % Return the LDA projection vectors
