function [d_CE_d_W, d_CE_d_b] = backward(X, Y, Y_predicted)
    % your code here
    % dJ/dz = Y_predicted - Y, dJ/dW = X' * dJ/dz / B, dJ/db = sum(dJ/dz) / B
    m = size(X,1);
    y = Y_predicted - Y;
    d_CE_d_W = (1/m) * X'*y;   % same shape as W (2 x 1)
    d_CE_d_b = (1/m)*sum(y);
end
