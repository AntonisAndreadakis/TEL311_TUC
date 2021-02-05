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

Y1 = mvnpdf([X1(:) X2(:)], m1, S1);
Y1R = reshape(Y1,length(x2),length(x1));

Y2 = mvnpdf([X1(:) X2(:)], m2, S2);
Y2R = reshape(Y2,length(x2),length(x1));

figure(1);
hold on
contour(x1,x2,Y1R,[.0001 .001 .01 .05: .1 :.95 .99 .999 .9999],'LineColor','r')
contour(x1,x2,Y2R,[.0001 .001 .01 .05: .1 :.95 .99 .999 .9999],'LineColor','b')
grid on

%S1 ~= S2
Pw1 = [0.1 0.25 0.5 0.75 0.9];
Pw2 = 1 - Pw1;

for i = 1:length(Pw1)
    syms x y
    y = (22.4 - 2*log(Pw2(i)/Pw1(i)))/(1.25*x);
    fplot(y);
    axis([-3 12 -3 12]);
    legend('class 1', 'class 2', 'P(w)=0.1', 'P(w)=0.25', 'P(w)=0.5','P(w)=0.75', 'P(w)=0.9')
end

%Repeating for %S1 == S2
figure(2);
hold on
contour(x1,x2,Y1R,[.0001 .001 .01 .05: .1 :.95 .99 .999 .9999],'LineColor','r')
contour(x1,x2,Y2R,[.0001 .001 .01 .05: .1 :.95 .99 .999 .9999],'LineColor','b')
grid on
for i = 1:length(Pw1)
    syms x y
    y = (67.7 - 2*log(Pw2(i)/Pw1(i)) - 7.6*x)/(7.6);
    fplot(y);
    axis([-3 12 -3 12]);
    legend('class 1', 'class 2', 'P(w)=0.1', 'P(w)=0.25', 'P(w)=0.5','P(w)=0.75', 'P(w)=0.9')
end