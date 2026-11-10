function [Z] = projectDataLDA(X, v)
% Project the samples X (one per row) on the projection vectors v.
%   v: one projection vector per column (NumFeatures x NewDim).
%      A single vector may also be given as a row.

% ====================== YOUR CODE HERE ======================

if size(v, 1) ~= size(X, 2)
    v = v';       % a single vector given as a row
end
Z = X*v;
% =============================================================

end
