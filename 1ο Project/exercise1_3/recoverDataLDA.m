function [X_rec] = recoverDataLDA(Z, v)
% Map the projections Z back to the original space, on the line (or
% subspace) spanned by the unit projection vectors v.
%   v: one projection vector per column (NumFeatures x NewDim).
%      A single vector may also be given as a row.

% ====================== YOUR CODE HERE ======================

if size(v, 2) ~= size(Z, 2)
    v = v';       % a single vector given as a row
end
X_rec = Z*v';
% =============================================================

end
