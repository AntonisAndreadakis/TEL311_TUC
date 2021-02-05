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
X_all = (X_all - m)/s;
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
num_batches = N - batch_size;

for epoch = 1:num_epochs
    epoch_loss = 0; 
    for i = 1:num_batches % Sliding window of length = batch_size and shift = 1
        X = X_all(i:i+batch_size, :); 
        Y = Y_all(i:i+batch_size, :);

        Y_predicted = forward(X, W, b);
        batch_loss = cross_entropy(Y, Y_predicted(:,1));   
        epoch_loss = epoch_loss + batch_loss;

        [d_CE_d_W, d_CE_d_b] = backward(X, Y, Y_predicted(:,1)); 

        % Update weights
        % your code here
        tmp = learning_rate.*d_CE_d_W;
        tmp = tmp(:); %make it column in order to success the above equation
        W = W - tmp;
        b = b - learning_rate*d_CE_d_b;
        prediction = forward(X, W, b); %make prediction
        for i = 1 : size(prediction)
            if prediction(i,2) > 0.5
                prediction(i,1) = 1;
            else
                prediction(i,1) = 0;
            end
        end
    end
    epoch_loss = epoch_loss/num_batches;
    disp(strcat('epoch_loss = ', num2str(epoch_loss)))
end

mean_prediction = mean(prediction(:,1)); %sta8ero sto 0,7778 gia 500 fores
% kai diaforopoieitai otan trexoyme 55 fores to loop
% apo 0,67 - 0,89

% Using the trained model to predict the probabilities of some examples and the compute the accuracy
% Predict the normalized example [45, 85]
example = ([45, 85] - m)./s;
fprintf("\n");
disp('Predicting the probabilities of example [45, 85]:')
% your code here
kapiametavliti = 100*forward(example', W, b);
%fprintf("\n");
k = mean(mean(kapiametavliti));
fprintf("Propability = %.2f%%\n", k);

Y_predicted = forward(X_all, W, b);
for i = 1 : size(Y_predicted)
    if Y_predicted(i,2) > 0.5
        Y_predicted(i,1) = 1;
    else
        Y_predicted(i,1) = 0;
    end
end
%Y_predicted = sum(Y_predicted);
correct = 0;
% Predict the accuracy of the training examples
for t = 1 : size(Y_all,1)
    if Y_all(t) == Y_predicted(t,1)
        correct = correct+1;
    end
end
accuracy = 100*(correct / length(Y_all)); % your code here
fprintf("\n");
fprintf("Accuracy = %.2f%%\n", accuracy);