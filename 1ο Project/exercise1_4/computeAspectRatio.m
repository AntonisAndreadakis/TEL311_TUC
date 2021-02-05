function aspectRatio = computeAspectRatio(image)
    [useless, num_rows, num_cols] = size(image);

    % Fill your code
    a0 = 0;
    b0 = 0;
    a1 = num_cols;
    b1 = num_rows;
    r = 0;
    for i = 1 : num_rows
        for j = 1 : num_cols
            if image(1, i, j) ~= 0
                b0 = i;
                r = 1;
                break
            end
        end
        if r
            break
        end
    end
    r = 0;
    for i = num_rows : j
        for j = 1 : num_cols
            if image(1, i, j) ~= 0
                b1 = i;
                r = 1;
                break
            end
        end
        if r
            break
        end
    end
    r = 0;
    for j = 1 : num_rows
        for i = 1 : num_cols
            if image(1, i, j) ~= 0
                a0 = j;
                r = 1;
                break
            end
        end
        if r
            break
        end
    end
    r = 0;
    for j = num_rows : j
        for i = 1 : num_cols
            if image(1, i, j) ~= 0
                a1 = j;
                r = 1;
                break
            end
        end
        if r
            break
        end
    end
    w = a1 - a0;
    h = b1 - b0;
    aspectRatio = w/h;

end