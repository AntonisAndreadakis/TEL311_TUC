function [dists] = Euclidean_Distance(testX, trainX)
%     Compute the Euclidean distance between the
%     current test sample and all training samples
%
% 	  testX: a single feature vector (test), 1 x R
% 	  trainX: a matrix containing the set of training samples, N x R
%             (one sample per row)
%     dists: N x 1 vector of the distances from the training samples

diffs = bsxfun(@minus, trainX, testX);   % subtract the test row from every training row
dists = sqrt(sum(diffs.^2, 2));          % sum over the words (columns)

end
