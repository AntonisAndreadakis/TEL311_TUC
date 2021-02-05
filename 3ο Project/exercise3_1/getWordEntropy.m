function [e]=getWordEntropy(f)
    %Calculate the vector of word entropies e 
    %from the Term Frequency matrix f
    %Add your code here
    %First we compute the propabilities
    for i = 1 : size(f,1) %num of rows of matrix f
        for j = 1 : size(f,2) %num of columns of f
            p(i,j) = f(i,j)./length(f(:,j));
            if p(i,j) == 0 %defeat p = 0
                p(i,j) = 0.00001;
            else
                    p(i,j) = p(i,j)*log2(p(i,j));
            end
            
        end
    end
    %Now we can calculate the entropies according:
    for j = 1 : size(f,2)
        e(1,j) = 1 + sum(p(:,j))/log2(size(f,1));
    end  
end