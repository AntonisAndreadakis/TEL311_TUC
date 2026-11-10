clc; clear all; close all;

data = load('mnist.mat');

images = zeros(size(data.trainX, 1), 28, 28);
labels = zeros(size(data.trainY, 1), 1);

for i = 1:size(data.trainX, 1)
    img = data.trainX(i, :);
    images(i, :, :) = reshape(img, 28, 28)';
    labels(i) = data.trainY(i);
end

digit_C1_indices = find(labels == 1); % digit 1
digit_C2_indices = find(labels == 2); % digit 2

digit_C1_images = images(digit_C1_indices, :, :);
digit_C2_images = images(digit_C2_indices, :, :);


aspectRatios = zeros(size(data.trainX, 1), 1);
% Compute the aspect ratios of all images and store the value of the i-th image in aRatios(i)
for i = 1 : size(data.trainX, 1)
    aspectRatios(i) = computeAspectRatio(images(i, :, :));
end
fprintf('All digits: min aspect ratio = %.4f, max aspect ratio = %.4f\n', ...
    min(aspectRatios), max(aspectRatios));

% For the histograms use the range of the images of the classes C1 and C2
isC1 = (labels == 1);
isC2 = (labels == 2);
minAspectRatio = min(aspectRatios(isC1 | isC2));
maxAspectRatio = max(aspectRatios(isC1 | isC2));
fprintf('C1 and C2: min aspect ratio = %.4f, max aspect ratio = %.4f\n', ...
    minAspectRatio, maxAspectRatio);

numBins = 3;   % 1: Low, 2: Medium, 3: High

binEnds = linspace(minAspectRatio, maxAspectRatio, numBins+1);

C1_bins = zeros(numBins, 1);
C2_bins = zeros(numBins, 1);

% Use the findBin function to get the counts for the histogram
for i = 1 : size(data.trainX, 1)
    if isC1(i)
        bin = findBin(aspectRatios(i), binEnds);
        C1_bins(bin) = C1_bins(bin) + 1;
    elseif isC2(i)
        bin = findBin(aspectRatios(i), binEnds);
        C2_bins(bin) = C2_bins(bin) + 1;
    end
end
all_bins = C1_bins + C2_bins;

binNames = {'Low', 'Medium', 'High'};
figure(1)
bar(all_bins)
set(gca, 'XTick', 1:numBins, 'XTickLabel', binNames)
title('Aspect ratio histogram: C1 and C2')
figure(2)
bar(C1_bins)
set(gca, 'XTick', 1:numBins, 'XTickLabel', binNames)
title('Aspect ratio histogram: C1 (digit 1)')
figure(3)
bar(C2_bins)
set(gca, 'XTick', 1:numBins, 'XTickLabel', binNames)
title('Aspect ratio histogram: C2 (digit 2)')

N1 = sum(C1_bins);   % number of images of class C1
N2 = sum(C2_bins);   % number of images of class C2

% Prior Probabilities
PC1 = N1/(N1 + N2);
PC2 = N2/(N1 + N2);

% Likelihoods P(L|C), P(M|C), P(H|C)
PgivenC1 = C1_bins/N1;
PgivenC2 = C2_bins/N2;

% Evidence P(L), P(M), P(H)
Pevidence = all_bins/(N1 + N2);

% Posterior Probabilities for X = L (Bayes rule)
PC1givenL = (PgivenC1(1)*PC1)/Pevidence(1);
PC2givenL = (PgivenC2(1)*PC2)/Pevidence(1);

fprintf('\nPriors:      P(C1) = %.4f, P(C2) = %.4f\n', PC1, PC2);
fprintf('Likelihoods: P(L|C1) = %.4f, P(M|C1) = %.4f, P(H|C1) = %.4f\n', PgivenC1);
fprintf('             P(L|C2) = %.4f, P(M|C2) = %.4f, P(H|C2) = %.4f\n', PgivenC2);
fprintf('Evidence:    P(L) = %.4f, P(M) = %.4f, P(H) = %.4f\n', Pevidence);
fprintf('Posteriors:  P(C1|X=L) = %.4f, P(C2|X=L) = %.4f\n', PC1givenL, PC2givenL);
