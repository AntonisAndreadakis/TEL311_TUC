clc;
clear;
close all;

load('digits');
A = reshape(train3(43, :), 28, 28)';
figure(12);
imagesc(A);
colorbar

b0=mean(train0);
b1=mean(train1);
b2=mean(train2);
b3=mean(train3);
b4=mean(train4);
b5=mean(train5);
b6=mean(train6);
b7=mean(train7);
b8=mean(train8);
b9=mean(train9);

figure(1);
imagesc(reshape(b0,28,28)');
title('MLE Bernoulli Distribution of digit 0');
figure(2)
imagesc(reshape(b1,28,28)');
title('MLE Bernoulli Distribution of digit 1');
figure(3)
imagesc(reshape(b2,28,28)');
title('MLE Bernoulli Distribution of digit 2');
figure(4)
imagesc(reshape(b3,28,28)');
title('MLE Bernoulli Distribution of digit 3');
figure(5)
imagesc(reshape(b4,28,28)');
title('MLE Bernoulli Distribution of digit 4');
figure(6)
imagesc(reshape(b5,28,28)');
title('MLE Bernoulli Distribution of digit 5');
figure(7)
imagesc(reshape(b6,28,28)');
title('MLE Bernoulli Distribution of digit 6');
figure(8)
imagesc(reshape(b7,28,28)');
title('MLE Bernoulli Distribution of digit 7');
figure(9)
imagesc(reshape(b8,28,28)');
title('MLE Bernoulli Distribution of digit 8');
figure(10)
imagesc(reshape(b9,28,28)');
title('MLE Bernoulli Distribution of digit 9');

test_set=[test0, test1, test2, test3, test4, test5, test6, test7, test8, test9];
correct=0;
test_set_size=size(test0,1)+size(test1,1)+size(test2,1)+size(test3,1)+...
              size(test4,1)+size(test5,1)+size(test6,1)+size(test7,1)+...
              size(test8,1)+size(test9,1);
confusion_matrix=zeros(10,10);

% One row per digit with the Bernoulli parameters p_yi. Many pixels have
% p=0 or p=1 exactly, which gives ln(0)=-Inf (and 0*Inf=NaN), so the
% parameters are clamped to [ep, 1-ep] before taking logarithms.
B=[b0;b1;b2;b3;b4;b5;b6;b7;b8;b9];
ep=1e-10;
B=min(max(B,ep),1-ep);
logB=log(B);
log1mB=log(1-B);
 
for i=1:size(test0,2):size(test_set,2)
  label=floor(1+i/784);
  for j=1:size(test_set,1)
    x=double(test_set(j,i:i+size(test0,2)-1));
    % Naive Bayes (Bernoulli): p(x|y)=prod_i p_yi^x_i*(1-p_yi)^(1-x_i), so
    % log p(x|y)=sum_i x_i*ln(p_yi)+(1-x_i)*ln(1-p_yi). Equal priors (same
    % number of training samples per digit), so they are left out.
    prediction=(logB*x'+log1mB*(1-x)')';
    [~,pred]=max(prediction);
    confusion_matrix(label,pred)=confusion_matrix(label,pred)+1;
    if(pred==label)
      correct=correct+1;
    end
  end
  disp(['Accuracy after testing digits ',num2str(label-1),' : ',num2str(correct/(label*size(test0,1)))]);
end
disp(['Accuracy: ', num2str(correct/test_set_size)]);
confusion_matrix=confusion_matrix./size(test0,1);

% Displaying the confusion matrix
figure(11)
imagesc(0:9,0:9,confusion_matrix);
colorbar
xlabel('Classified as digit'); ylabel('True digit');
title('Confusion Matrix');
disp('Confusion matrix (row: true digit 0..9, column: classified as 0..9):');
disp(confusion_matrix);

