function [Z] = projectDataLDA(X, v)

% You need to return the following variables correctly.
Z = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================

[m, ~] = size(v);
if m==1 
    Z = X*v';
else    
    for i=1:m
        Z(:,i)= X*v(i,:)';
    end
end
% =============================================================

end
