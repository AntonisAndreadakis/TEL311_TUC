clc;
clear;
close all;

% In GNU Octave the Statistics package must be loaded first
if exist('OCTAVE_VERSION', 'builtin')
    pkg load statistics
end

load fisheriris.mat

X = meas;
N = size(X, 1);                    % number of examples
iris_labels = grp2idx(species);    % 1: setosa, 2: versicolor, 3: virginica
K = 3;                             % Number of classes

% ---- GMM fitted with EM ----
[Pm, M, S] = fitGMMs(X, K);

Y_hat = zeros(N, 1);
for i = 1:N
    Y_hat(i, 1) = classify_with_gmms(X(i, :), Pm, M, S);
end

% ---- kmeans, for comparison ----
c = kmeans(X, K, 'Replicates', 5);

% ---- Accuracy ----
% Cluster numbers are arbitrary, so they are matched to the classes first
[GMMaccuracy, Y_gmm] = clusterAccuracy(Y_hat, iris_labels, K);
[kmeansAccuracy, Y_kmeans] = clusterAccuracy(c, iris_labels, K);
fprintf('\n\tGMM accuracy = %.2f%%.\n', GMMaccuracy);
fprintf('\n\tkmeans accuracy = %.2f%%.\n', kmeansAccuracy);

% ---- Plots: petal length vs petal width, coloured by assigned class ----
names = {'setosa', 'versicolor', 'virginica'};
results = {Y_gmm, Y_kmeans};
titles = {sprintf('fitted GMM (accuracy %.2f%%)', GMMaccuracy), ...
          sprintf('fitted kmeans (accuracy %.2f%%)', kmeansAccuracy)};
for fig = 1:2
    figure(fig)
    Y = results{fig};
    wrong = Y ~= iris_labels;
    gscatter(X(:, 3), X(:, 4), names(Y)', 'rgb', '.', 15);
    hold on
    plot(X(wrong, 3), X(wrong, 4), 'ko', 'MarkerSize', 10, ...
         'DisplayName', 'wrong class');
    hold off
    xlabel('petal length (cm)')
    ylabel('petal width (cm)')
    title(titles{fig})
    legend('Location', 'northwest')
    set(gca, 'FontSize', 12)
    set(gca, 'FontWeight', 'bold')
end
