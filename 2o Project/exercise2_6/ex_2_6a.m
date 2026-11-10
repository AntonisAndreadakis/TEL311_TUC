clc;
clear;
close all;
% Exercise 2.6 A: plot the samples of the two classes, the support vectors
% and the optimal (maximum margin) separating line.
w1 = [3, 1];  u1 = [1, 0];
w2 = [3, -1]; u2 = [0, 1];
w3 = [5, 1];  u3 = [0, -1];
w4 = [5, -1]; u4 = [-1, 0];
%
Xpos = [w1; w2; w3; w4];   % class omega_1 (+1), one sample per row
Xneg = [u1; u2; u3; u4];   % class omega_2 (-1)

% The closest samples of the two classes are [3,1], [3,-1] (omega_1) and
% [1,0] (omega_2): these are the support vectors. The KKT conditions
%   w = sum_i lambda_i*y_i*x_i,  sum_i lambda_i*y_i = 0,  y_i*(w*x_i'+w0) = 1
% on these three samples give lambda = [1/4, 1/4, 1/2], w = [1, 0], w0 = -2,
% i.e. the separating line x1 = 2 with margins x1 = 1 and x1 = 3.
Xsup = [w1; w2; u1];
ysup = [1; 1; -1];
lambda = [1/4; 1/4; 1/2];
w = (lambda.*ysup)'*Xsup;
w0 = mean(ysup - Xsup*w');
fprintf('w = [%g %g], w0 = %g, margin width = %g\n', w(1), w(2), w0, 2/norm(w));

figure(1); hold on;
plot(Xpos(:,1), Xpos(:,2), 'b+', 'MarkerSize', 10, 'LineWidth', 2);
plot(Xneg(:,1), Xneg(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot(Xsup(:,1), Xsup(:,2), 'ks', 'MarkerSize', 16);
% w(2) = 0, so the lines w*x'+w0 = c are vertical: x1 = (c - w0)/w(1)
t = [-2.5, 2.5];
plot((0-w0)/w(1)*[1 1], t, 'k-', 'LineWidth', 2);
plot((1-w0)/w(1)*[1 1], t, 'b--');
plot((-1-w0)/w(1)*[1 1], t, 'r--');
hold off;
axis([-2 6 -2.5 2.5]); grid on;
xlabel('x1'); ylabel('x2');
legend('\omega_1 (+1)', '\omega_2 (-1)', 'Support vectors', ...
       'Separating line x1 = 2', 'Margin (+1)', 'Margin (-1)', ...
       'Location', 'northoutside');
title('Exercise 2.6 A');
