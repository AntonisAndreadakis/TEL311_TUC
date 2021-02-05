function [dists] = Euclidean_Distance(testX, trainX)
%     Compute the Euclidean  distance between the 
%     current test sample and all training samples
%
% 	  testX: a single feature vector (test)
% 	  trainX: a matrix containing the set of training samples
%     dists: vector of the distances from the training samples

%  ADD your code here
[m1,n1] = size(testX);
[m2,n2] = size(trainX);
if (m1~=m2)
    error('Matrix dimensions do not match.');
end
if (n1<n2)
    copies = zeros(1,n2);
    dists(n1,:) = sqrt(sum((testX(:,n1+copies)-trainX).^2,1));
else
    copies = zeros(1,n1);
    dists(:,n2) = sqrt(sum((testX-trainX(:,n2+copies)).^2,1)');
end

end