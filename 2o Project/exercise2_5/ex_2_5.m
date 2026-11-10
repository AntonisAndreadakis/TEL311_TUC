clc;
clear;
close all;

% Read the n = 25 measurements. The file has the form "H(25)=[ v1 v2 ... ]",
% so it cannot be read with load(); the numbers after '[' are parsed instead.
txt = fileread('data_2_5.txt');
H = sscanf(txt(strfind(txt,'[')+1:end), '%f');
n = length(H);
sigma = 1.25;        % known standard deviation of the measurements
m0 = 0;              % prior mean of mu
a = (sigma^2);       % known variance sigma^2

% Sample mean of the first k measurements, k = 1..n
y = zeros(1,n);
for i=1:n
  y(i) = mean(H(1:i));
end

% Bayesian estimation of mu with Gaussian prior N(m0, s0) (s0 = sigma_0^2):
%   p(mu|H(n)) = N(mu_n, s_n),
%   mu_n = (n*s0*mean_n + a*m0)/(n*s0 + a),   s_n = s0*a/(n*s0 + a)
%   p(x|H(n))  = N(mu_n, a + s_n)             (predictive density)
x=linspace(-5,12,1000);

% EROTIMA A: p(mu|H(n)) for n = 1..25, sigma_0^2 = 10*sigma^2
s0 = 10*a;
figure(1);hold on;
for i = 1 : n
    mu_n = (i*s0*y(i) + a*m0)/(i*s0 + a);
    s_n  = s0*a/(i*s0 + a);
    plot(x,normpdf(x, mu_n, sqrt(s_n)),'linewidth',1);
end
hold off;
xlabel('\mu'); ylabel('p(\mu|H(n))');
title('p(\mu|H(n)), n = 1..25, \sigma_0^2 = 10\sigma^2');
fprintf('A: sigma0^2 = 10*sigma^2, n = %d: mu_n = %f, sigma_n = %f\n', n, mu_n, sqrt(s_n));

% EROTIMA B: p(x|H(n)) for n = 1..25 and for different prior variances
factors = [10, 1, 0.1, 0.01];      % sigma_0^2 = factor * sigma^2
for k = 1 : length(factors)
    s0 = factors(k)*a;
    figure(k+1);hold on;
    for i = 1 : n
        mu_n = (i*s0*y(i) + a*m0)/(i*s0 + a);
        s_n  = s0*a/(i*s0 + a);
        plot(x,normpdf(x, mu_n, sqrt(a + s_n)),'linewidth',1);
    end
    hold off;
    xlabel('x'); ylabel('p(x|H(n))');
    title(sprintf('p(x|H(n)), n = 1..25, \\sigma_0^2 = %g\\sigma^2', factors(k)));
    fprintf('B: sigma0^2 = %g*sigma^2, n = %d: mean = %f, std = %f\n', ...
            factors(k), n, mu_n, sqrt(a + s_n));
end
fprintf('Sample mean of all %d measurements: %f\n', n, y(n));
