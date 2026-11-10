clear all
close all

% Part 3: samples that are not linearly separable

C = 1; % Choose C = 0.01, 0.1, 1, 10, 100, 1000, 10000

load('./twofeature2.txt');
n = size(twofeature2, 1); 
y = twofeature2(1:n, 1);
X = twofeature2(1:n, 2:3);

% Form the augmented dataset. 
% (x1, x2) -> (x1, x2, ||x||^2), with ||x||^2 = x1^2 + x2^2
Xa = [X, sum(X.^2, 2)];

Xpos = Xa(y==1, :); % positive examples
Xneg = Xa(y==-1, :); % negative examples

%  Visualize the example dataset
hold on
plot(Xpos(:, 1), Xpos(:, 2), 'b.');
plot(Xneg(:, 1), Xneg(:, 2), 'r.');
hold off
axis square;

% Form the matrices for the quadratic optimization. See the matlab manual for "quadprog"
% Same dual problem as in part 2 on the augmented samples:
% minimise 1/2*lambda'*H*lambda + f'*lambda, H(i,j) = y_i*y_j*xa_i*xa_j',
% subject to sum(lambda.*y) = 0 and 0 <= lambda <= C.
H = (y*y').*(Xa*Xa');

f = -ones(n,1);

A = [];

b = [];

Aeq = y';

beq = 0;

lb = zeros(n,1);
ub = C*ones(n,1);

lambda = quadprog(H,f,A,b,Aeq,beq,lb,ub); % Find the Lagrange multipliers

tol = 1e-4*min(1,C);          % numerical tolerance for lambda = 0 or lambda = C
indices = find(lambda > tol); % Find the indices of the support vectors
Xsup = Xa(indices,:); % The support vectors only
ysup = y(indices);
lambdasup = lambda(indices);

% Find the weights: w = sum_i lambda_i*y_i*xa_i (row vector)
w = (lambdasup.*ysup)'*Xsup;

% Find the bias term w0: for the support vectors on the margin (0 < lambda < C)
% y_k*(w*xa_k' + w0) = 1  =>  w0 = y_k - w*xa_k'  (averaged over them)
k_ind = find(lambda > tol & lambda < C-tol);
if isempty(k_ind)
  % All the support vectors are at the bound lambda = C (very small C):
  % w0 is not unique, take the middle of the interval allowed by the KKT conditions
  s = Xa*w';
  w0 = (max(-1-s(y==-1 & lambda>C-tol)) + min(1-s(y==1 & lambda>C-tol)))/2;
else
  w0 = mean(y(k_ind) - Xa(k_ind,:)*w');
end

fprintf('C = %g\n', C);
fprintf('Number of support vectors: %d\n', length(indices));
fprintf('Estimated weight vector: %s\n', sprintf('%.4f ', w));
fprintf('Bias: %f\n', w0);
fprintf('Misclassified training examples: %d\n', sum(sign(Xa*w' + w0) ~= y));

% Plot support vectors
Xsup_pos = Xsup(ysup==1, :);
Xsup_neg = Xsup(ysup==-1, :);

hold on
plot(Xsup_pos(:, 1), Xsup_pos(:, 2), 'bo');
plot(Xsup_neg(:, 1), Xsup_neg(:, 2), 'ro');
hold off


% Plot decision boundary
hold on
for x1 = -1:0.01:1
    for x2 = -1:0.01:1
        if (abs(w*[x1, x2, x1^2+x2^2]' + w0) < 0.01) % fill the 3rd coordinate
           plot(x1, x2, 'k.')
        end
    end
end
hold off
xlabel('x1'); ylabel('x2');
title(sprintf('SVM with the extra feature ||x||^2, C = %g', C));
