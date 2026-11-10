clc;
clear;
close all;

% Part 1: linearly separable samples, hard margin SVM (no slack variables)

load('./twofeature1.txt');
n = size(twofeature1, 1) - 1; % leave out the last example
y = twofeature1(1:n, 1);
X = twofeature1(1:n, 2:3);

Xpos = X(y==1, :); % positive examples
Xneg = X(y==-1, :); % negative examples
%figure(1);
%  Visualize the example dataset
hold on
plot(Xpos(:, 1), Xpos(:, 2), 'b.');
plot(Xneg(:, 1), Xneg(:, 2), 'r.');
hold off
axis square%, xlim([0 10])

% Form the matrices for the quadratic optimization. See the matlab manual for "quadprog"
% Maximising L(lambda) = sum(lambda) - 1/2*lambda'*H*lambda is the same as
% minimising 1/2*lambda'*H*lambda + f'*lambda with H(i,j) = y_i*y_j*x_i*x_j'
% and f = -1, under the constraints lambda >= 0 and sum(lambda.*y) = 0.
H = (y*y').*(X*X');
f = -ones(n,1);
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(n,1); %create zero lower bounds
ub = [];         % hard margin: no upper bound on lambda
lambda = quadprog(H,f,A,b,Aeq,beq,lb,ub); % Find the Lagrange multipliers
indices = find(lambda > 0.0001); % Find the support vectors
Xsup = (X(indices,:));            % The support vectors only
ysup = y(indices)';
lambdasup = lambda(indices);

fprintf('Support vector indices: %s\n', sprintf('%d ', indices));
fprintf('Support vector weights: %s\n', sprintf('%.4f ', lambdasup));
% Find the weights: w = sum_i lambda_i*y_i*x_i
w = X'*(lambda.*y);
% Find the bias term w0: every support vector lies on its margin,
% y_i*(x_i*w + w0) = 1  =>  w0 = y_i - x_i*w  (averaged over the support vectors)
w0 = mean((ysup'-Xsup*w));
fprintf('Estimated weight vector: %s\n', sprintf('%.4f ', w));
fprintf('Bias: %f\n', w0);

% Plot support vectors
Xsup_pos = Xsup(ysup==1, :);
Xsup_neg = Xsup(ysup==-1, :);
hold on
plot(Xsup_pos(:, 1), Xsup_pos(:, 2), 'bo');
plot(Xsup_neg(:, 1), Xsup_neg(:, 2), 'ro');
hold off

% Find the width of the margin: the lines x*w + w0 = +1 and x*w + w0 = -1
% are 2/||w|| apart
width = 2/norm(w);
fprintf('Width of the margin: %f\n', width);
fprintf('\n');

% Plot decision boundary x*w + w0 = 0  =>  x2 = -(w(1)*x1 + w0)/w(2)
x1 = linspace(min(X(:,1)),max(X(:,1)),10);
x2 = -(w(1)*x1 + w0)/w(2);
hold on
plot(x1, x2, '--k')
plot(x1, -(w(1)*x1 + w0 - 1)/w(2), 'b')  %Plot the Margin of class +1
plot(x1, -(w(1)*x1 + w0 + 1)/w(2), 'r')  %Plot the Margin of class -1
hold off
axis([min(X(:,1))-0.5, max(X(:,1))+0.5, min(X(:,2))-0.5, max(X(:,2))+0.5]);
xlabel('x1'); ylabel('x2');
title('Hard margin SVM (50 examples)');
