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
minAspectRatio = min(aspectRatios);
maxAspectRatio = max(aspectRatios);

numBins = 3;

binEnds = linspace(minAspectRatio, maxAspectRatio, numBins+1);

C1_bins = zeros(numBins, 1);
C2_bins = zeros(numBins, 1);
all_bins = zeros(numBins, 1);

% Use the findBin function to get the counts for the histogram
for i = 1 : size(data.trainX, 1)
    if labels(i) ==1 || labels(i) == 2
    bin = findBin(aspectRatios(i), binEnds);
    if bin == 1
        all_bins(1) = all_bins(1) + 1;
    elseif bin == 2
            all_bins(2) = all_bins(2) + 1;
        elseif bin == 3
                all_bins(3) = all_bins(3) + 1;
     end
            bins(i) = bin;
    end
    if labels(i) == 1
        if bin == 1
            C1_bins(1) = C1_bins(1) + 1;
        elseif bin == 2
            C1_bins(2) = C1_bins(2) + 1;
        elseif bin == 3
            C1_bins(3) = C1_bins(3) + 1;
        end
        bins_C1(i) = bin;
    end
    if labels(i) == 2
        if bin == 1
            C2_bins(1) = C2_bins(1) + 1;
        elseif bin == 2
            C2_bins(2) = C2_bins(2) + 1;
        elseif bin == 3
            C2_bins(3) = C2_bins(3) + 1;
        end
        bins_C2(i) = bin;
    end
end
    
figure(1)
hist(bins, 3)
figure(2)
hist(bins_C1, 3)
figure(3)
hist(bins_C2, 3)

% Prior Probabilities
PC1 = length(bins_C1)/length(bins);
PC2 = length(bins_C2)/length(bins);

% Likelihoods
PgivenC1 = C1_bins/length(bins_C1);
PgivenC2 = C2_bins/length(bins_C2);

% Evidence 
Pevidence = all_bins/length(bins);
% Posterior Probabilities
PC1givenL = (PgivenC1(1)*PC1)/Pevidence(1);
PC2givenL = (PgivenC2(1)*PC2)/Pevidence(1);