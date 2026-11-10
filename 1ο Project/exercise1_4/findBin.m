function index = findBin(value, binEnds)
    numBins = length(binEnds) - 1;
    % return the index of the bin where value belongs {1, ..., numBins}.
    % Bin i covers [binEnds(i), binEnds(i+1)); the last bin also includes
    % its upper end, so that the maximum value falls in bin numBins.
    index = 1;
    for i = 1 : numBins
        if value >= binEnds(i)
            index = i;
        end
    end
end
