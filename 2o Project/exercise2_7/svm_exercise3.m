clear all
close all

C = 1; % Choose C = 0.01, 0.1, 1, 10, 100, 1000, 10000

load('./twofeature2.txt');
n = size(twofeature2, 1); 
y = twofeature2(1:n, 1);
X = twofeature2(1:n, 2:3);

% Form the augmented dataset. 
Xa = 

Xpos = Xa(y==1, :); % positive examples
Xneg = Xa(y==-1, :); % negative examples

%  Visualize the example dataset
hold on
plot(Xpos(:, 1), Xpos(:, 2), 'b.');
plot(Xneg(:, 1), Xneg(:, 2), 'r.');
hold off
axis square;

% Form the matrices for the quadratic optimization. See the matlab manual for "quadprog"
H = 

f = 

A =  

b = 

Aeq = 

beq = 

lambda = quadprog(....); % Find the Lagrange multipliers

indices = find(lambda > 0.0001); % Find the indices of the support vectors
Xsup = ; % The support vectors only
ysup = ;
lambdasup = ;

% Find the bias term w0
w0 = 

% Find the weights
w = 

% Plot support vectors
Xsup_pos = 
Xsup_neg = 

hold on
plot(Xsup_pos(:, 1), Xsup_pos(:, 2), 'bo');
plot(Xsup_neg(:, 1), Xsup_neg(:, 2), 'ro');
hold off


% Plot decision boundary
hold on
for x1 = -1:0.01:1
    for x2 = -1:0.01:1
        if (abs(w*[x1, x2, ....]' + w0) < 0.01) % fill the 3rd coordinate
           plot(x1, x2, 'k.')
        end
    end
end
hold off


