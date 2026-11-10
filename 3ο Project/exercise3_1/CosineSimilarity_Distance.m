function dists = CosineSimilarity_Distance(testX, trainX)
%     Compute the cosine distance (1 - cosine similarity) between the
%     current test sample and all training samples.
%     A distance is returned (not the similarity) so that
%     "smallest = nearest" holds, as for Euclidean_Distance.
%
% 	  testX: a single feature vector (test), 1 x R
% 	  trainX: a matrix containing the set of training samples, N x R
%             (one sample per row)
%     dists: N x 1 vector of the distances from the training samples

num = trainX * testX';                                  % N x 1 dot products
denom = sqrt(sum(trainX.^2, 2)) * norm(testX);          % per-row norms
dists = 1 - num ./ max(denom, 1e-14);                   % guard against zero vectors

end
