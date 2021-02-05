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
 
for i=1:size(test0,2):size(test_set,2)
  label=floor(1+i/784);
  for j=1:size(test_set,1)
    prediction=[];
    prediction=[...
    mean(b0.*test_set(j,i:i+size(test0,2)-1)+(1-b0).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b1.*test_set(j,i:i+size(test0,2)-1)+(1-b1).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b2.*test_set(j,i:i+size(test0,2)-1)+(1-b2).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b3.*test_set(j,i:i+size(test0,2)-1)+(1-b3).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b4.*test_set(j,i:i+size(test0,2)-1)+(1-b4).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b5.*test_set(j,i:i+size(test0,2)-1)+(1-b5).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b6.*test_set(j,i:i+size(test0,2)-1)+(1-b6).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b7.*test_set(j,i:i+size(test0,2)-1)+(1-b7).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b8.*test_set(j,i:i+size(test0,2)-1)+(1-b8).*(1-test_set(j,i:i+size(test0,2)-1))),...
    mean(b9.*test_set(j,i:i+size(test0,2)-1)+(1-b9).*(1-test_set(j,i:i+size(test0,2)-1)))];
    [~,pred]=max(prediction);
    confusion_matrix(label,pred)=confusion_matrix(label,pred)+1;
    if(pred==label)
      correct=correct+1;
    end
  end
  disp(['Accuracy after testing digits ',num2str(label-1),' : ',num2str(correct/(label*size(test0,25)))]);
end
disp(['Accuracy: ', num2str(correct/test_set_size)]);
confusion_matrix=confusion_matrix./size(test0,1);

% Displaying the confusion matrix
figure(11)
imagesc(confusion_matrix);
title('Confusion Matrix');

