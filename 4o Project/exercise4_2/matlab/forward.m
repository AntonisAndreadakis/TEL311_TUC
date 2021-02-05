function y = forward(x, W, b)
    % your code here
    % formula: y = w * x + b 
    a = x*W'+b;
    y = sigmoid(a);
end
