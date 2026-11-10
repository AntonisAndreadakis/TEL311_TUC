function [test_z] = KNN_classify(k, train_f, train_y, test_f, dtype)
   % K-NN classification algorithm
   % k: Number of nearest neighbours
   % train_f: The matrix of training feature vectors (one sample per row)
   % train_y: The labels of the training data
   % test_f: The matrix of the test feature vectors (one sample per row)
   % dtype: Integer which defines the distance metric
   %    dtype=1: Call the function Euclidean_Distance
   %    dtype=2: Call the function CosineSimilarity_Distance
   % test_z: The predicted label of every test sample

   n = size(test_f, 1);
   test_z = zeros(n, 1);
   for i = 1 : n
       if dtype == 1
           d = Euclidean_Distance(test_f(i, :), train_f);
       else
           d = CosineSimilarity_Distance(test_f(i, :), train_f);
       end
       [~, b] = sort(d);                      % nearest first
       test_z(i) = mode(train_y(b(1:k)));     % majority vote
   end
end
