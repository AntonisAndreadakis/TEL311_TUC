function [acc, mapped] = clusterAccuracy(clusters, labels, K)
    % Accuracy of a clustering against the true labels.
    %
    % Cluster numbers are arbitrary (cluster 1 need not be class 1), so the
    % clusters are first matched to the classes: every one-to-one assignment
    % of cluster numbers to class numbers is tried and the best one is kept.
    %
    % INPUT:
    %   clusters: (N X 1) cluster index of every example, in 1..K
    %   labels: (N X 1) true class of every example, in 1..K
    %   K: number of clusters / classes
    %
    % OUTPUT:
    %   acc: accuracy (%) under the best matching
    %   mapped: (N X 1) the clusters renamed to their matched classes

    allPerms = perms(1:K);   % K! candidates; fine for small K
    acc = -1;
    mapped = clusters;
    for p = 1:size(allPerms, 1)
        candidate = allPerms(p, clusters)';
        a = 100 * mean(candidate == labels);
        if a > acc
            acc = a;
            mapped = candidate;
        end
    end
end
