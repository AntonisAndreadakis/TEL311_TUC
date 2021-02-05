function [d_CE_d_W, d_CE_d_b] = backward(X, Y, Y_predicted)
    % your code here
    m = size(X,1);
    y = Y_predicted - Y;
    d_CE_d_W = (1/m) * y'*X;%dot(y,X);
    d_CE_d_b = (1/m)*sum(y);
end
