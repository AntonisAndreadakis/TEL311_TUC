clc;
clear;
close all;
w1 = [3, 1];  u1 = [1, 0];
w2 = [3, -1]; u2 = [0, 1];
w3 = [5, 1];  u3 = [0, -1];
w4 = [5, -1]; u4 = [-1, 0];
%
x1 = {w1' , w2' , w3' , w4'};
x2 = {u1' , u2' , u3' , u4'};

xTrain = meshgrid(x1, x2);
xlabel('x1'); ylabel('x2');




