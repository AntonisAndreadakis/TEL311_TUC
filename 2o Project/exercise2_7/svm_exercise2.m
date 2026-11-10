clc;
clear;
close all;

% Part 2: slack variables (soft margin SVM)

C = 10000; % Choose C = 0.01, 0.1, 1, 10, 100, 1000, 10000

load('./twofeature1.txt');
n = size(twofeature1, 1); % all the examples
y = twofeature1(1:n, 1);
X = twofeature1(1:n, 2:3);

Xpos = X(y==1, :); % positive examples
Xneg = X(y==-1, :); % negative examples

% Visualize the example dataset
hold on
plot(Xpos(:, 1), Xpos(:, 2), 'b.',Xneg(:, 1), Xneg(:, 2), 'r.');
axis square

% Form the matrices for the quadratic optimization. See the matlab manual for "quadprog"
% Same dual problem as in part 1, but with 0 <= lambda <= C.
H = (y*y').*(X*X');
f = -ones(n,1);
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(n,1); %create zero lower bounds
ub = C*ones(n,1);
lambda = quadprog(H,f,A,b,Aeq,beq,lb,ub); % Find the Lagrange multipliers
tol = 1e-4*min(1,C);             % numerical tolerance for lambda = 0 or lambda = C
indices = find(lambda > tol);    % Find the support vectors
Xsup = X(indices,:);             % The support vectors only 
ysup = y(indices)';
lambdasup = lambda(indices);
% Support vectors with 0 < lambda < C lie exactly on the margin
k_ind = find(lambda > tol & lambda < C-tol);
fprintf('C = %g\n', C);
fprintf('Support vector indices: %s\n', sprintf('%d ', indices));
fprintf('Support vector weights: %s\n', sprintf('%.4f ', lambdasup));

% Find the weights: w = sum_i lambda_i*y_i*x_i
w = sum(repmat(lambda .* y, 1, 2) .* X, 1)';
% Find the bias term w0: for the support vectors on the margin
% y_k*(x_k*w + w0) = 1  =>  w0 = y_k - x_k*w  (averaged over them)
if isempty(k_ind)
  % All the support vectors are at the bound lambda = C (very small C):
  % w0 is not unique, take the middle of the interval allowed by the KKT conditions
  s = X*w;
  w0 = (max(-1-s(y==-1 & lambda>C-tol)) + min(1-s(y==1 & lambda>C-tol)))/2;
else
  w0 = mean(y(k_ind) - X(k_ind,:)*w);
end
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
fprintf('Misclassified training examples: %d\n', sum(sign(X*w + w0) ~= y));
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
title(sprintf('Soft margin SVM (51 examples), C = %g', C));
