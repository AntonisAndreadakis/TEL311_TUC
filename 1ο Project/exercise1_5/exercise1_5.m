clc;
clear;
close all;

m1 = [3 3];
m2 = [6 6];

S1 = [1.2 -0.4; -0.4 1.2];
S2 = [1.2 0.4; 0.4 1.2];

DT=0.01;

x1 = [-3:DT:12]; %Horizontal axis
x2 = [-3:DT:12];
[X1,X2] = meshgrid(x1,x2);

% contour levels of the class conditional densities (their peak is ~0.14)
levels = [.0001 .001 .01 .05 .1];

Pw1 = [0.1 0.25 0.5 0.75 0.9];
Pw2 = 1 - Pw1;
colors = 'kmgcy';
h = zeros(1, length(Pw1));   % handles of the boundary curves, for the legend

%% ---------------- S1 ~= S2 ----------------
% Decision boundary: g1(x) = g2(x), with
%   gi(x) = -0.5*(x-mi)'*inv(Si)*(x-mi) - 0.5*ln|Si| + ln P(wi)
% Here |S1| = |S2| = 1.28 and
%   inv(S1) = [0.9375 0.3125; 0.3125 0.9375]
%   inv(S2) = [0.9375 -0.3125; -0.3125 0.9375]
% The squared terms cancel and what remains is
%   (x-m2)'*inv(S2)*(x-m2) - (x-m1)'*inv(S1)*(x-m1) = 22.5 - 1.25*x*y
% so the boundary is the hyperbola
%   1.25*x*y = 22.5 - 2*ln(P(w2)/P(w1))

Y1 = mvnpdf([X1(:) X2(:)], m1, S1);
Y1R = reshape(Y1,length(x2),length(x1));

Y2 = mvnpdf([X1(:) X2(:)], m2, S2);
Y2R = reshape(Y2,length(x2),length(x1));

figure(1);
hold on
contour(x1,x2,Y1R,levels,'LineColor','r')
contour(x1,x2,Y2R,levels,'LineColor','b')
grid on

x = x1;
x(abs(x) < 1e-9) = NaN;   % the hyperbola is not defined at x = 0
for i = 1:length(Pw1)
    y = (22.5 - 2*log(Pw2(i)/Pw1(i)))./(1.25*x);
    h(i) = plot(x, y, colors(i), 'LineWidth', 1);
end
axis([-3 12 -3 12]);
xlabel('x_1'); ylabel('x_2');
title('\Sigma_1 \neq \Sigma_2 (red: class 1, blue: class 2)');
legend(h, 'P(w_1)=0.1', 'P(w_1)=0.25', 'P(w_1)=0.5', 'P(w_1)=0.75', 'P(w_1)=0.9')
hold off

%% ---------------- S1 == S2 (both equal to S2) ----------------
% With a common covariance matrix S the quadratic terms cancel and the
% boundary is the straight line  w'*x + w0 = 0, with
%   w  = inv(S)*(m1-m2) = [-1.875; -1.875]
%   w0 = -0.5*(m1*inv(S)*m1' - m2*inv(S)*m2') + ln(P(w1)/P(w2))
%      = 16.875 + ln(P(w1)/P(w2))
% that is
%   3.75*(x + y) = 33.75 - 2*ln(P(w2)/P(w1))

Y1 = mvnpdf([X1(:) X2(:)], m1, S2);   % class 1 now also has covariance S2
Y1R = reshape(Y1,length(x2),length(x1));

figure(2);
hold on
contour(x1,x2,Y1R,levels,'LineColor','r')
contour(x1,x2,Y2R,levels,'LineColor','b')
grid on
x = x1;
for i = 1:length(Pw1)
    y = (33.75 - 2*log(Pw2(i)/Pw1(i)) - 3.75*x)/3.75;
    h(i) = plot(x, y, colors(i), 'LineWidth', 1);
end
axis([-3 12 -3 12]);
xlabel('x_1'); ylabel('x_2');
title('\Sigma_1 = \Sigma_2 (red: class 1, blue: class 2)');
legend(h, 'P(w_1)=0.1', 'P(w_1)=0.25', 'P(w_1)=0.5', 'P(w_1)=0.75', 'P(w_1)=0.9')
hold off
