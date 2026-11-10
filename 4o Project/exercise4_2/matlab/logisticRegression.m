clear
close all
clc
load('exam_scores_data1.txt')

X_all = exam_scores_data1(:, 1:2); % input
Y_all = exam_scores_data1(:, 3); % target

N = size(X_all, 1); % Number of examples

% Normalize input data (X_all)
% your code here
m = mean(X_all);
s = std(X_all);
X_all = bsxfun(@rdivide, bsxfun(@minus, X_all, m), s); % per feature
% Define the model
num_features = 2;
input_depth = num_features;
output_depth = 1;
batch_size = 8;
learning_rate = 0.001;

% Define the weights
W = normrnd(0, sqrt(2.0/(input_depth + output_depth)), input_depth, output_depth);
b = zeros(output_depth, 1);

% Training the model
num_epochs = 500;
num_batches = N - batch_size + 1;

for epoch = 1:num_epochs
    epoch_loss = 0;
    for i = 1:num_batches % Sliding window of length = batch_size and shift = 1
        X = X_all(i:i+batch_size-1, :);
        Y = Y_all(i:i+batch_size-1, :);

        Y_predicted = forward(X, W, b);
        batch_loss = cross_entropy(Y, Y_predicted);
        epoch_loss = epoch_loss + batch_loss;

        [d_CE_d_W, d_CE_d_b] = backward(X, Y, Y_predicted);

        % Update weights
        % your code here
        W = W - learning_rate*d_CE_d_W;
        b = b - learning_rate*d_CE_d_b;
    end
    epoch_loss = epoch_loss/num_batches;
    disp(strcat('epoch_loss = ', num2str(epoch_loss)))
end

% Using the trained model to predict the probabilities of some examples and the compute the accuracy
% Predict the normalized example [45, 85]
example = ([45, 85] - m)./s;
fprintf('\n');
disp('Predicting the probabilities of example [45, 85]:')
% your code here
probability = forward(example, W, b);
fprintf('Probability = %.2f%%\n', 100*probability);

% Predict the accuracy of the training examples
Y_predicted = double(forward(X_all, W, b) > 0.5);
accuracy = 100*mean(Y_predicted == Y_all); % your code here
fprintf('\n');
fprintf('Accuracy = %.2f%%\n', accuracy);
