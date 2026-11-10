function aspectRatio = computeAspectRatio(image)
    % Aspect ratio (width/height) of the smallest rectangle that encloses
    % the digit, i.e. all the non-zero pixels of the image.
    %   image: 1 x num_rows x num_cols
    [~, num_rows, num_cols] = size(image);
    im = reshape(image, num_rows, num_cols);

    rows = find(any(im ~= 0, 2));   % rows that contain part of the digit
    cols = find(any(im ~= 0, 1));   % columns that contain part of the digit

    if isempty(rows)                % blank image: no rectangle
        aspectRatio = NaN;
        return
    end

    % +1 because both the first and the last row/column belong to the digit
    h = rows(end) - rows(1) + 1;
    w = cols(end) - cols(1) + 1;
    aspectRatio = w/h;

end
