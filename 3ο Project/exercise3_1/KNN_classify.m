function [test_z] = KNN_classify(k, train_f, train_y, test_f, dtype)
   % K-NN classification algorithm
   % k: Number of nearest neighbours
   % train_f: The matrix of training feature vectors
   % train_y: The labels of the training data
   % test_f: The matrix of the test feature vectors
   % dtype: Integer which defines the distance metric
   %    dtype=1: Call the function Euclidean_Distance
   %    dtype=2: Call the function CosineSimilarity_Distance
 
   % Add your code here
   [n,~] = size(train_y);
   test_z = 0;
   for i = 1 : n
       if dtype == 1
           d = Euclidean_Distance(train_f,test_f);  
       elseif (dtype == 2 || dtype ~= 1)
           d = CosineSimilarity_Distance(train_f,test_f);
       end
       [~,b] = sort(d,'ascend');
       test_z = [test_z; mode(train_y(b(1:k)))];
   end
  test_z = test_z(2:end); % leave first out, see on top (test_z=0)
end