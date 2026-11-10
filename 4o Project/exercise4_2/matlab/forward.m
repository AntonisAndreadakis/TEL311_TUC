function y = forward(x, W, b)
    % your code here
    % formula: y = f(x * W + b), x: B x 2, W: 2 x 1, b: 1 x 1
    a = x*W + b;
    y = sigmoid(a);
end
