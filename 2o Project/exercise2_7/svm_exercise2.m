clc;
clear;
close all;

% C = 10000; % Choose C = 0.01, 0.1, 1, 10, 100, 1000, 10000

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
H = (y*y').*(X*X');
f = -ones(n,1);
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(n,1); %create zero lower bounds
C = 10000;
ub = C*ones(n,1);
lambda = quadprog(H,f,A,b,Aeq,beq,lb,ub); % Find the Lagrange multipliers
indices = find(lambda > 0.0001); % Find the support vectors
Xsup = X(indices,:);            % The support vectors only 
ysup = y(indices)';
lambdasup = sum((lambda.*y).*(X*X'))';
t = find(abs(lambdasup>0&abs(lambdasup)<C-0.0001));
k_ind = find(lambda>0 & lambda<C-0.0001);
new_k_ind = X(k_ind,:);
fprintf('Support vector indices: %s\n', sprintf('%d ', indices));
fprintf('Support vector weights: %s\n', sprintf('%.4f ', lambda(indices)));

% Find the weights
w = sum(repmat(lambda .* y, 1, 2) .* X, 1)';
% Find the bias term w0
w0 = ((y(k_ind(1)) * w .* X(k_ind(1),:)') - 1) ./ y(k_ind(1));
fprintf('Estimated weight vector: %s\n', sprintf('%.4f ', w));
fprintf('Bias: %f\n', w0);
fprintf('\n');

% Plot support vectors
Xsup_pos = Xsup(ysup==1, :);
Xsup_neg = Xsup(ysup==-1, :);
hold on
plot(Xsup_pos(:, 1), Xsup_pos(:, 2), 'bo');
plot(Xsup_neg(:, 1), Xsup_neg(:, 2), 'ro');
hold off
% Find the width of the margin
width = 1/sqrt(sum(lambda(indices))+(1/C)*(lambda'*lambda));
% Plot decision boundary
x1 = linspace(0.5,4.5,10);
x2 = -0.55*x1+4.5;
hold on
plot(x1, x2, '--k')
plot(Xsup_pos(:),width,'b')  %Plot the Margin of class +1
plot(Xsup_neg(:),width,'r')  %Plot the Margin of class -1
hold off
