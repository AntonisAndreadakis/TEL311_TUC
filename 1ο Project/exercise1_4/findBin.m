function index = findBin(value, binEnds)
    numBins = length(binEnds) - 1;
    % return the index of the bin where value belongs {1, ..., numBins}.
    index = numBins;
    for i = 1 : index
        if value > binEnds(i)
            index = i;
        end
    end
end