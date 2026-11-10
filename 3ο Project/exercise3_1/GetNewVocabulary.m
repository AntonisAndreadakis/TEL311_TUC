function [newVocabulary, newTermEntropy, new_f] = GetNewVocabulary(r, Vocabulary, e, f)
    %Return the r words of the Vocabulary (newVocabulary) with the
    %    lowest entropy weight e, i.e. the frequent, broadly distributed
    %    words. (Words with e close to 1 occur in only a handful of
    %    documents and leave most feature vectors empty.)
    %Also return
    %   newTermEntropy: their corresponding entropy weight
    %   new_f: the TF-IDF weighted Term Document Matrix restricted to
    %          the newVocabulary (f must hold the raw term counts)

    [sortedE, order] = sort(e(:));   % ascending
    keep = order(1:r);

    newVocabulary = Vocabulary(keep, :);
    newTermEntropy = sortedE(1:r);
    new_f = full(tfidf(f(:, keep), 1));
end
