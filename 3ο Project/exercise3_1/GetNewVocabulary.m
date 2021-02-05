function [newVocabulary, newTermEntropy, new_f] = GetNewVocabulary(r, Vocabulary, e, f)
    %Return the top r words in the Vocabulary (newVocabulary) 
    %    based on their entropy value
    %Also return 
    %   newTermEntropy: their corresponding Entropy value
    %   new_f: the new Term Document Matrix based on the
    %                newVocabulary

    % ADD Your Code Here
    
    [ind,z] = sort(e(:));
    newTermEntropy = ind(1:r,1);
    [~,a] = ind2sub(size(e),z(1:r));
    for i = 1 : r
        newVocabulary(i,:) = Vocabulary(a(i),:);
        new_f(:,i) = f(:,a(i));
    end
    new_f = full(new_f);
end