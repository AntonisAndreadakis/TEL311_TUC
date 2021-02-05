clc;
clear;
close all;

load fisheriris.mat

X = meas;
N = size(X, 1); % number of examples
species
speciesNum = grp2idx(species);
iris_labels = 1*cellfun(@(x)isequal(x,'versicolor'),species)+2*cellfun(@(x)isequal(x,'virginica'),species);
K = 3; % Number of classes
[Pm, M, S] = fitGMMs(X, K);

Y_hat = zeros(N, 1);
for i = 1:N
    Y_hat(i, 1) = classify_with_gmms(X(i, :), Pm, M, S)-1;
end

figure(1)
gscatter(iris_labels,Y_hat,speciesNum,['r','g','b'])
xlabel('X')
ylabel('Yhat')
title('fitted GMM')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')
% Accuracy %
l1 = length(find(Y_hat==iris_labels));
GMMaccuracy = 100*(l1/size(Y_hat,1));
c = zeros(N,1);
c = kmeans(X,K);
l2 = length(find(c==iris_labels));
kmeansAccuracy = 100*(l2/size(Y_hat,1));
fprintf('\n\tGMM accuracy = %.2f%%.\n', GMMaccuracy);
fprintf('\n\tkmeans accuracy = %.2f%%.\n', kmeansAccuracy);

figure(2)
gscatter(iris_labels,c,speciesNum,['r','g','b'])
xlabel('irisLabels')
ylabel('kmeans')
title('fitted kmeans')
set(gca,'FontSize',12)
set(gca,'FontWeight','bold')