clc;
clear;
close all;
% Exercise 2.6 B: plot the samples, apply the transformation Phi and plot
% the transformed samples with the optimal separating line.
Xpos = [2, 2; 2, -2; -2, -2; -2, 2];   % class omega_1 (+1), one sample per row
Xneg = [1, 1; 1, -1; -1, -1; -1, 1];   % class omega_2 (-1)

% B.1: original samples. omega_2 lies inside the square of omega_1, so no
% line separates the two classes in the original space.
figure(1); hold on;
plot(Xpos(:,1), Xpos(:,2), 'b+', 'MarkerSize', 10, 'LineWidth', 2);
plot(Xneg(:,1), Xneg(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
axis([-3 3 -3 3]); axis square; grid on;
xlabel('x1'); ylabel('x2');
legend('\omega_1 (+1)', '\omega_2 (-1)', 'Location', 'northoutside');
title('Exercise 2.6 B: original samples');

% B.2: Phi(x) = x - 4*||x||_2 - 4, with ||x||_2^2 = x1^2 + x2^2.
% The scalar 4*||x||_2 + 4 is subtracted from both coordinates of x.
Ppos = Xpos - (4*sqrt(sum(Xpos.^2, 2)) + 4)*[1, 1];
Pneg = Xneg - (4*sqrt(sum(Xneg.^2, 2)) + 4)*[1, 1];
disp('Phi(x) for omega_1:'); disp(Ppos);
disp('Phi(x) for omega_2:'); disp(Pneg);

% B.3: in the transformed space the two closest samples of the two classes
% are Phi([2,2]) and Phi([-1,-1]); they are the support vectors. With two
% support vectors xp (+1), xn (-1) the KKT conditions give
%   lambda_p = lambda_n = lambda,  w = lambda*(xp - xn),
%   w*xp' + w0 = 1,  w*xn' + w0 = -1  =>  lambda = 2/||xp - xn||^2,
%   w0 = -w*(xp + xn)'/2
% (the line is the perpendicular bisector of the segment xp-xn).
xp = Ppos(1,:);
xn = Pneg(3,:);
lambda = 2/sum((xp - xn).^2);
w = lambda*(xp - xn);
w0 = -w*(xp + xn)'/2;
fprintf('Support vectors: [%g %g] (+1), [%g %g] (-1)\n', xp(1), xp(2), xn(1), xn(2));
fprintf('lambda = %g, w = [%g %g], w0 = %g, margin width = %g\n', ...
        lambda, w(1), w(2), w0, 2/norm(w));
% All samples must satisfy y*(w*x'+w0) >= 1
fprintf('min y*(w*x''+w0) over all samples: %g\n', ...
        min([Ppos*w' + w0; -(Pneg*w' + w0)]));

figure(2); hold on;
plot(Ppos(:,1), Ppos(:,2), 'b+', 'MarkerSize', 10, 'LineWidth', 2);
plot(Pneg(:,1), Pneg(:,2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot([xp(1); xn(1)], [xp(2); xn(2)], 'ks', 'MarkerSize', 16);
% Lines w*x'+w0 = c  =>  x2 = (c - w0 - w(1)*x1)/w(2)
t = linspace(-20, -6, 10);
plot(t, (0 - w0 - w(1)*t)/w(2), 'k-', 'LineWidth', 2);
plot(t, (1 - w0 - w(1)*t)/w(2), 'b--');
plot(t, (-1 - w0 - w(1)*t)/w(2), 'r--');
hold off;
axis([-20 -6 -20 -6]); axis square; grid on;
xlabel('\Phi_1(x)'); ylabel('\Phi_2(x)');
legend('\omega_1 (+1)', '\omega_2 (-1)', 'Support vectors', ...
       'Separating line', 'Margin (+1)', 'Margin (-1)', ...
       'Location', 'northoutside');
title('Exercise 2.6 B: transformed samples');
