clc;
clear;
close all;

H = [3.1803;3.0208;6.3968;5.0169;4.0323;6.2086;5.3474;3.5154;7.0094;5.4343;7.2184;6.7070;3.9738;3.0621;4.8906;4.5041;3.7346;5.7467;7.2327;4.1803;3.7299;4.6305;5.9945;3.7187;3.1980];
n = 25;
std = 1.25;
m0 = 0;
a = (std^2);

y=0;
x=linspace(-10,10,100);
%size(H)
% MLE of mean
for i=1:size(H,2)
  y=[y mean(H(1,1:i))];
end
% EROTIMA A:
figure(1);hold on;
for i = 1 : size(y,2)
    plot(x,normpdf(x, m0, sqrt(a*(10*a)/(a+i*10*a))),'linewidth',3);
end

% EROTIMA B:
figure(2);hold on;
for i=1:size(y,2)
  plot(x,normpdf(x, (i*(10*a)*y(1,i))/(a+i*10*a), sqrt(a*(10*a)/(a+i*10*a))),'linewidth',3);
end

figure(3);hold on;
for i=1:size(y,2)
  plot(x,normpdf(x, (i*a*y(1,i))/(a+i*a), sqrt(a*(a)/(a+i*a))),'linewidth',3);
end

figure(4);hold on;
for i=1:size(y,2)
  plot(x,normpdf(x, (i*(0.1*a)*y(1,i))/(a+i*0.1*a), sqrt(a*(0.1*a)/(a+i*0.1*a))),'linewidth',3);
end

figure(5);hold on;
for i=1:size(y,2)
  plot(x,normpdf(x, (i*(0.01*a)*y(1,i))/(a+i*0.01*a), sqrt(a*(0.01*a)/(a+i*0.01*a))),'linewidth',3);
end

