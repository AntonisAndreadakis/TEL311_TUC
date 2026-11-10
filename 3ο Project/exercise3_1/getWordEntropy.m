function [e]=getWordEntropy(f)
    %Calculate the vector of word entropy weights e (1 x nWords)
    %from the Term Frequency matrix f (nDocs x nWords)
    %
    %   e_j = 1 + sum_i p_ij*log2(p_ij) / log2(nDocs),   p_ij = f_ij / gf_j
    %
    %where gf_j is the total count of word j over all documents.
    %   e = 1: word concentrated in a single document
    %   e = 0: word spread uniformly over all documents

    nDocs = size(f, 1);
    gf = sum(f, 1);
    gf(gf == 0) = 1;               % words that never occur: avoid 0/0
    p = bsxfun(@rdivide, f, gf);   % probability of word j in document i

    plogp = zeros(size(p));        % 0*log2(0) is taken as 0
    nz = p > 0;
    plogp(nz) = p(nz) .* log2(p(nz));

    e = 1 + sum(plogp, 1) / log2(nDocs);
end
