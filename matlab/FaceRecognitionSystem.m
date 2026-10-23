%% =========================================================================
%  MATLAB FACE RECOGNITION & IDENTIFICATION SYSTEM
%  Compatible with MATLAB Online (R2023a+)
% ==========================================================================
%
%  This script implements the complete 4-module Face Recognition pipeline
%  using ONLY classical image processing & computer vision techniques.
%  No deep learning toolboxes are used.
%
%  Modules Implemented:
%    1. Preprocessing & Normalization
%    2. Feature Extraction & Template Generation
%    3. Database & Storage (MAT-file based)
%    4. Matching & Identification
%
%  HOW TO RUN IN MATLAB ONLINE:
%    1. Upload this file to MATLAB Online (https://matlab.mathworks.com)
%    2. Run the script (F5 or click Run)
%    3. The interactive menu will appear in the Command Window
%    4. Use the demo option (Option 3) to see the pipeline in action
%
%  IMPORTANT: MATLAB Online does not support live webcam.
%             Upload face images manually and use image-based enrollment.
%
% ==========================================================================

function FaceRecognitionSystem()
% FACERECOGNITIONSYSTEM  Main entry point for the Face Recognition System.
%
%   Launches an interactive menu to enroll users, identify faces, and
%   visualize the preprocessing pipeline — all in MATLAB Online.

    clear;
    fprintf('\n========================================\n');
    fprintf('  FACE RECOGNITION SYSTEM — MATLAB\n');
    fprintf('========================================\n\n');

    % Initialize the database (MAT-file)
    db = DB_initialize('face_db.mat');

    % Main interactive loop
    while true
        fprintf('\n--- MAIN MENU ---\n');
        fprintf('1. Enroll user from image file\n');
        fprintf('2. Identify face from image file\n');
        fprintf('3. Run preprocessing demo (built-in image)\n');
        fprintf('4. List enrolled users\n');
        fprintf('5. Delete a user\n');
        fprintf('6. Exit\n');
        choice = input('Select option [1-6]: ', 's');

        switch strtrim(choice)
            case '1'
                name = input('Enter user name: ', 's');
                imgPath = input('Enter image file path (or press Enter for demo): ', 's');
                if isempty(imgPath)
                    img = generate_synthetic_face();
                else
                    img = imread(strtrim(imgPath));
                end
                db = enroll_user(db, strtrim(name), img);

            case '2'
                imgPath = input('Enter image file path (or Enter for demo): ', 's');
                if isempty(imgPath)
                    img = generate_synthetic_face();
                else
                    img = imread(strtrim(imgPath));
                end
                threshold = 3.85;
                identify_face(db, img, threshold);

            case '3'
                run_preprocessing_demo();

            case '4'
                DB_list_users(db);

            case '5'
                DB_list_users(db);
                uid = input('Enter user ID to delete: ');
                if isnumeric(uid) && ~isempty(uid)
                    db = DB_delete_user(db, uid);
                end

            case '6'
                fprintf('\nGoodbye!\n');
                return;

            otherwise
                fprintf('Invalid choice. Please enter 1-6.\n');
        end
    end
end


%% =========================================================================
%%  MODULE 1: PREPROCESSING & NORMALIZATION
%% =========================================================================

function gray = convert_to_grayscale(img)
% CONVERT_TO_GRAYSCALE  Convert RGB image to grayscale using luminance formula.
%
%   Y = 0.299*R + 0.587*G + 0.114*B
%
%   This weighted sum matches the human visual system's sensitivity:
%   green contributes the most to perceived brightness, then red, then blue.
%
%   Args:
%       img  : Input image (uint8, H×W×3 RGB or H×W grayscale)
%   Returns:
%       gray : Grayscale image (uint8, H×W)

    if ndims(img) == 3
        % Manual luminance-weighted conversion
        R = double(img(:,:,1));
        G = double(img(:,:,2));
        B = double(img(:,:,3));
        gray = uint8(0.299*R + 0.587*G + 0.114*B);
    else
        gray = img;  % Already grayscale
    end
end


function yuv = convert_to_yuv(img)
% CONVERT_TO_YUV  Convert RGB image to YUV color space manually.
%
%   YUV separates luminance (Y) from chrominance (U, V):
%       Y =  0.299*R + 0.587*G + 0.114*B
%       U = -0.147*R - 0.289*G + 0.436*B + 128
%       V =  0.615*R - 0.515*G - 0.100*B + 128
%
%   The Y channel alone is robust to illumination changes.
%
%   Args:
%       img : RGB image (uint8, H×W×3)
%   Returns:
%       yuv : YUV image (uint8, H×W×3), channels: [Y, U, V]

    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    Y =  0.299*R + 0.587*G + 0.114*B;
    U = -0.147*R - 0.289*G + 0.436*B + 128;
    V =  0.615*R - 0.515*G - 0.100*B + 128;

    yuv = uint8(cat(3, Y, U, V));
end


function hsi = convert_to_hsi(img)
% CONVERT_TO_HSI  Convert RGB to HSI (Hue, Saturation, Intensity) color space.
%
%   HSI separates chromatic (H, S) from intensity (I), making the I channel
%   ideal for illumination-robust face processing.
%
%   Formulas:
%       I = (R+G+B)/3
%       S = 1 - 3*min(R,G,B)/(R+G+B)
%       H = arccos[(R-G + R-B) / (2*sqrt((R-G)^2 + (R-B)(G-B)))]
%
%   Args:
%       img : RGB image (uint8, H×W×3)
%   Returns:
%       hsi : HSI image (double, H×W×3), values in [0,1]

    R = double(img(:,:,1)) / 255;
    G = double(img(:,:,2)) / 255;
    B = double(img(:,:,3)) / 255;

    % Intensity
    I = (R + G + B) / 3;

    % Saturation
    minRGB = min(min(R, G), B);
    S = 1 - 3 .* minRGB ./ (R + G + B + 1e-10);

    % Hue (arccos formula)
    num = 0.5 * ((R - G) + (R - B));
    den = sqrt((R - G).^2 + (R - B).*(G - B)) + 1e-10;
    theta = acos(max(-1, min(1, num ./ den)));

    H = theta;
    H(B > G) = 2*pi - H(B > G);
    H = H / (2*pi);  % Normalize to [0,1]

    hsi = cat(3, H, S, I);
end


function eq = histogram_equalization(gray)
% HISTOGRAM_EQUALIZATION  Apply global histogram equalization to grayscale image.
%
%   Algorithm:
%       1. Compute pixel frequency histogram H[k] for k=0..255
%       2. Compute CDF: C[k] = sum(H[0..k])
%       3. Normalize CDF to [0,255]: s[k] = round(255 * C[k] / N)
%       4. Map each pixel r to s[r]
%
%   This redistributes intensities to span the full [0,255] range,
%   compensating for uneven illumination.
%
%   Args:
%       gray : Grayscale image (uint8, H×W)
%   Returns:
%       eq   : Equalized grayscale image (uint8, H×W)

    [h, w] = size(gray);
    N = h * w;

    % Step 1: Compute histogram
    hist_vals = histcounts(double(gray(:)), 0:256) / N;

    % Step 2 & 3: CDF and mapping
    cdf = cumsum(hist_vals);
    mapping = uint8(round(255 * cdf));

    % Step 4: Apply mapping (LUT)
    eq = uint8(mapping(double(gray) + 1));
end


function clahe_out = apply_clahe(gray, clip_limit, tile_size)
% APPLY_CLAHE  Contrast Limited Adaptive Histogram Equalization (CLAHE).
%
%   Divides the image into non-overlapping tiles of size tile_size × tile_size,
%   applies histogram equalization to each tile with clipping at clip_limit,
%   then bilinearly interpolates between tiles to avoid block artifacts.
%
%   This is superior to global HE for faces: local contrast is enhanced
%   while uniform regions are not over-amplified.
%
%   Args:
%       gray       : Grayscale image (uint8, H×W)
%       clip_limit : Histogram clip limit [0,1] (default 0.01)
%       tile_size  : Tile dimensions [rows, cols] (default [8,8])
%   Returns:
%       clahe_out  : CLAHE-enhanced grayscale image (uint8, H×W)

    if nargin < 2, clip_limit = 0.01; end
    if nargin < 3, tile_size = [8, 8]; end

    [H, W] = size(gray);
    clahe_out = double(gray);

    num_tiles_y = floor(H / tile_size(1));
    num_tiles_x = floor(W / tile_size(2));

    % Process each tile
    tile_maps = zeros(num_tiles_y, num_tiles_x, 256);

    for ty = 1:num_tiles_y
        for tx = 1:num_tiles_x
            y1 = (ty-1)*tile_size(1) + 1;
            y2 = ty * tile_size(1);
            x1 = (tx-1)*tile_size(2) + 1;
            x2 = tx * tile_size(2);

            tile = gray(y1:y2, x1:x2);
            tile_N = numel(tile);

            % Compute and clip histogram
            h_tile = histcounts(double(tile(:)), 0:256) / tile_N;
            excess = max(0, h_tile - clip_limit);
            h_tile = h_tile + sum(excess) / 256;  % Redistribute excess
            h_tile = min(h_tile, clip_limit);

            % Compute CDF and mapping
            cdf = cumsum(h_tile / sum(h_tile));
            tile_maps(ty, tx, :) = round(255 * cdf);
        end
    end

    % Apply tile mappings with bilinear interpolation (simplified version)
    for ty = 1:num_tiles_y
        for tx = 1:num_tiles_x
            y1 = (ty-1)*tile_size(1) + 1;
            y2 = min(ty * tile_size(1), H);
            x1 = (tx-1)*tile_size(2) + 1;
            x2 = min(tx * tile_size(2), W);

            mapping = squeeze(tile_maps(ty, tx, :));
            for py = y1:y2
                for px = x1:x2
                    pv = double(gray(py, px)) + 1;
                    clahe_out(py, px) = mapping(pv);
                end
            end
        end
    end

    clahe_out = uint8(clahe_out);
end


function filtered = homomorphic_filter(gray, d0, gamma_low, gamma_high)
% HOMOMORPHIC_FILTER  2D Frequency-Domain Homomorphic Filter (Log-FFT).
%
%   Decouples illumination from reflectance:
%       I(x,y) = L(x,y) * R(x,y)
%       ln(1 + I) = ln(L) + ln(R)
%   High-emphasis Gaussian filter attenuates low-frequency illumination
%   and amplifies invariant structural facial reflectance.
    if nargin < 2, d0 = 20.0; end
    if nargin < 3, gamma_low = 0.75; end
    if nargin < 4, gamma_high = 1.25; end

    img_f = double(gray);
    img_log = log1p(img_f);

    fft_val = fftshift(fft2(img_log));
    [rows, cols] = size(gray);
    u = (0:rows-1) - rows / 2;
    v = (0:cols-1) - cols / 2;
    [V, U] = meshgrid(v, u);
    D = sqrt(U.^2 + V.^2);

    H = (gamma_high - gamma_low) * (1.0 - exp(-(D.^2) ./ (d0^2 + 1e-10))) + gamma_low;
    filtered_fft = fft_val .* H;

    exp_img = expm1(real(ifft2(ifftshift(filtered_fft))));
    p1 = prctile(exp_img(:), 1);
    p99 = prctile(exp_img(:), 99);
    stretched = (exp_img - p1) / (p99 - p1 + 1e-10);
    stretched = max(0.0, min(1.0, stretched)) * 255.0;
    filtered = uint8(stretched);
end


function blurred = gaussian_blur(gray, sigma, kernel_size)
% GAUSSIAN_BLUR  Apply Gaussian low-pass spatial filter.
%
%   Creates a Gaussian kernel:
%       G(x,y) = exp(-(x^2 + y^2) / (2*sigma^2))
%   and convolves it with the image to attenuate high-frequency noise.
%
%   Args:
%       gray        : Grayscale image (uint8, H×W)
%       sigma       : Standard deviation of the Gaussian (default 1.5)
%       kernel_size : Size of the square kernel (default 5, must be odd)
%   Returns:
%       blurred : Smoothed image (uint8, H×W)

    if nargin < 2, sigma = 1.5; end
    if nargin < 3, kernel_size = 5; end
    if mod(kernel_size, 2) == 0, kernel_size = kernel_size + 1; end

    % Build Gaussian kernel manually
    half = floor(kernel_size / 2);
    [X, Y] = meshgrid(-half:half, -half:half);
    kernel = exp(-(X.^2 + Y.^2) / (2 * sigma^2));
    kernel = kernel / sum(kernel(:));  % Normalize to unit sum

    % Convolve (use 'replicate' padding to handle borders)
    blurred = uint8(conv2(double(gray), kernel, 'same'));
end


function filtered = median_filter_2d(gray, kernel_size)
% MEDIAN_FILTER_2D  Apply median spatial filter (non-linear).
%
%   For each pixel, replaces its value with the median of its
%   kernel_size × kernel_size neighborhood.
%
%   The median operator is robust to salt-and-pepper noise because
%   outlier pixel values cannot shift the median (unlike mean filters).
%
%   Args:
%       gray        : Grayscale image (uint8, H×W)
%       kernel_size : Window size (default 3, must be odd)
%   Returns:
%       filtered    : Median-filtered image (uint8, H×W)

    if nargin < 2, kernel_size = 3; end
    if mod(kernel_size, 2) == 0, kernel_size = kernel_size + 1; end

    half = floor(kernel_size / 2);
    [H, W] = size(gray);
    filtered = gray;
    padded = padarray(double(gray), [half, half], 'replicate');

    for y = 1:H
        for x = 1:W
            neighborhood = padded(y:y+kernel_size-1, x:x+kernel_size-1);
            filtered(y, x) = median(neighborhood(:));
        end
    end
end


function [mag, gx, gy] = sobel_edges(gray)
% SOBEL_EDGES  Compute gradient magnitude and components using Sobel operators.
%
%   Sobel kernels:
%       Gx = [-1  0 +1; -2  0 +2; -1  0 +1]
%       Gy = [-1 -2 -1;  0  0  0; +1 +2 +1]
%   Magnitude: M = sqrt(Gx^2 + Gy^2)
%
%   Args:
%       gray : Grayscale image (uint8, H×W)
%   Returns:
%       mag  : Gradient magnitude (uint8, H×W)
%       gx   : Horizontal gradient (double, H×W)
%       gy   : Vertical gradient (double, H×W)

    Kx = [-1 0 1; -2 0 2; -1 0 1];
    Ky = [-1 -2 -1; 0 0 0; 1 2 1];

    img_d = double(gray);
    gx = conv2(img_d, Kx, 'same');
    gy = conv2(img_d, Ky, 'same');

    magnitude = sqrt(gx.^2 + gy.^2);
    mag = uint8(255 * magnitude / (max(magnitude(:)) + 1e-10));
end


function edges = canny_edge_detection(gray, low_thresh, high_thresh)
% CANNY_EDGE_DETECTION  Multi-stage optimal edge detector.
%
%   Pipeline:
%     1. Gaussian smoothing (sigma=1.0, 5×5 kernel)
%     2. Sobel gradient computation
%     3. Non-maximum suppression (thin edges to 1 pixel)
%     4. Double thresholding (strong: >high_thresh, weak: >low_thresh)
%     5. Hysteresis edge tracking (connect weak to strong edges)
%
%   Args:
%       gray        : Grayscale image (uint8, H×W)
%       low_thresh  : Lower hysteresis threshold (default 0.1 × maxMag)
%       high_thresh : Upper hysteresis threshold (default 0.3 × maxMag)
%   Returns:
%       edges       : Binary edge map (logical, H×W). Edge pixels = true.

    if nargin < 2, low_thresh_ratio  = 0.1; else, low_thresh_ratio  = low_thresh;  end
    if nargin < 3, high_thresh_ratio = 0.3; else, high_thresh_ratio = high_thresh; end

    % Step 1: Gaussian smoothing
    smoothed = gaussian_blur(gray, 1.0, 5);

    % Step 2: Sobel gradients
    [~, gx, gy] = sobel_edges(smoothed);
    magnitude = sqrt(gx.^2 + gy.^2);
    direction = atan2(gy, gx);  % [-pi, pi]

    % Quantize gradient direction to 4 angles (0°, 45°, 90°, 135°)
    dir_deg = mod(round(rad2deg(direction) / 45) * 45, 180);

    % Step 3: Non-maximum suppression
    [H, W] = size(magnitude);
    nms = zeros(H, W);
    for y = 2:H-1
        for x = 2:W-1
            d = dir_deg(y, x);
            m = magnitude(y, x);
            if d == 0    % Horizontal edge: compare left/right
                n1 = magnitude(y, x-1); n2 = magnitude(y, x+1);
            elseif d == 45   % Diagonal edge
                n1 = magnitude(y-1, x+1); n2 = magnitude(y+1, x-1);
            elseif d == 90   % Vertical edge: compare up/down
                n1 = magnitude(y-1, x); n2 = magnitude(y+1, x);
            else             % 135° diagonal
                n1 = magnitude(y-1, x-1); n2 = magnitude(y+1, x+1);
            end
            if m >= n1 && m >= n2
                nms(y, x) = m;
            end
        end
    end

    % Step 4: Double thresholding
    max_val  = max(nms(:));
    low_t    = low_thresh_ratio  * max_val;
    high_t   = high_thresh_ratio * max_val;
    strong   = nms >= high_t;
    weak     = nms >= low_t & nms < high_t;

    % Step 5: Hysteresis (simplified BFS-style)
    edges = strong;
    [wy, wx] = find(weak);
    changed = true;
    while changed
        changed = false;
        for k = 1:length(wy)
            y = wy(k); x = wx(k);
            if ~edges(y, x)
                % Check 8-connected neighborhood for a strong edge
                neighbors = edges(max(1,y-1):min(H,y+1), max(1,x-1):min(W,x+1));
                if any(neighbors(:))
                    edges(y, x) = true;
                    changed = true;
                end
            end
        end
    end
end


function face_roi = detect_and_crop_face(img, target_size)
% DETECT_AND_CROP_FACE  Detect primary face and return normalized ROI.
%
%   Uses MATLAB's built-in vision.CascadeObjectDetector (Viola-Jones)
%   for Haar Cascade face detection. Falls back to center-crop if
%   the Computer Vision Toolbox is unavailable.
%
%   After detection, applies:
%       1. Grayscale conversion
%       2. Histogram equalization (CLAHE)
%       3. Gaussian blur (noise reduction)
%       4. Resize to target_size
%
%   Args:
%       img         : Input RGB or grayscale image (uint8, H×W or H×W×3)
%       target_size : [height, width] of output ROI (default [100,100])
%   Returns:
%       face_roi    : Normalized face ROI (uint8, target_size)
%                     or [] if no face detected

    if nargin < 2, target_size = [100, 100]; end

    gray = convert_to_grayscale(img);

    % Try Viola-Jones face detector (requires Computer Vision Toolbox)
    try
        detector = vision.CascadeObjectDetector();
        bbox = step(detector, img);
        if isempty(bbox)
            face_roi = [];
            fprintf('[PREPROCESS] No face detected.\n');
            return;
        end
        % Take the largest detected face by area
        [~, idx] = max(bbox(:,3) .* bbox(:,4));
        x = bbox(idx,1); y = bbox(idx,2);
        w = bbox(idx,3); h = bbox(idx,4);

        % Canonical inner face crop (focus on eyes, nose, mouth; ignore hair/collar variations)
        cy = max(1, floor(y + 0.10 * h));
        ch = min(size(gray, 1) - cy + 1, floor(0.82 * h));
        cx = max(1, floor(x + 0.08 * w));
        cw = min(size(gray, 2) - cx + 1, floor(0.84 * w));
        roi = gray(cy:cy+ch-1, cx:cx+cw-1);

    catch
        % Fallback: center crop (no Computer Vision Toolbox)
        fprintf('[PREPROCESS] CascadeObjectDetector unavailable.\n');
        fprintf('[PREPROCESS] Using center-crop as face ROI.\n');
        [H, W] = size(gray);
        margin_y = floor(H * 0.15);
        margin_x = floor(W * 0.15);
        roi = gray(margin_y+1:H-margin_y, margin_x+1:W-margin_x);
    end

    % Normalization pipeline
    face_roi = imresize(roi, target_size);      % Resize to fixed size
    face_roi = homomorphic_filter(face_roi);    % Frequency-domain illumination decoupling
    face_roi = histogram_equalization(face_roi);% Illumination normalization
    face_roi = gaussian_blur(face_roi, 0.8, 3); % Noise reduction
end


%% =========================================================================
%%  MODULE 2: FEATURE EXTRACTION & TEMPLATE GENERATION
%% =========================================================================

function hist_lbp = compute_lbp_histogram(gray, num_bins)
% COMPUTE_LBP_HISTOGRAM  Compute LBP texture descriptor.
%
%   Local Binary Pattern (LBP) encodes local texture by comparing each
%   pixel to its 8 neighbors. If neighbor ≥ center → bit=1, else bit=0.
%   The 8-bit result (0-255) is the LBP code for that pixel.
%
%   The normalized histogram of all LBP codes captures the distribution
%   of micro-textures across the face image.
%
%   Algorithm (8-neighbor, radius=1):
%       pattern = 0
%       For k = 0..7 (neighbors in clockwise order):
%           if I(neighbor_k) >= I(center): pattern |= (1 << k)
%       LBP[pixel] = pattern
%
%   Args:
%       gray     : Grayscale face ROI (uint8, H×W)
%       num_bins : Number of histogram bins (default 256)
%   Returns:
%       hist_lbp : Normalized LBP histogram (double, 1×num_bins)

    if nargin < 2, num_bins = 256; end

    g = double(gray);
    [H, W] = size(g);
    lbp = zeros(H, W, 'uint8');

    % 8-neighbor comparison (vectorized for speed)
    % Neighbor bit positions (clockwise from top-left):
    % bit0:top-left  bit1:top  bit2:top-right  bit3:right
    % bit4:bot-right bit5:bot  bit6:bot-left   bit7:left
    c = g(2:H-1, 2:W-1);  % Center pixels

    lbp(2:H-1,2:W-1) = ...
        uint8(g(1:H-2,1:W-2) >= c) * 1   + ...
        uint8(g(1:H-2,2:W-1) >= c) * 2   + ...
        uint8(g(1:H-2,3:W  ) >= c) * 4   + ...
        uint8(g(2:H-1,3:W  ) >= c) * 8   + ...
        uint8(g(3:H  ,3:W  ) >= c) * 16  + ...
        uint8(g(3:H  ,2:W-1) >= c) * 32  + ...
        uint8(g(3:H  ,1:W-2) >= c) * 64  + ...
        uint8(g(2:H-1,1:W-2) >= c) * 128;

    % Compute normalized histogram
    hist_lbp = histcounts(double(lbp(:)), 0:num_bins) / numel(lbp);
end


function hist_grad = compute_gradient_histogram(gray, num_bins)
% COMPUTE_GRADIENT_HISTOGRAM  Sobel gradient orientation histogram.
%
%   Computes the magnitude-weighted histogram of gradient orientations.
%   Captures the distribution of edge directions — encodes global facial
%   shape (contours of jaw, nose, eyebrows).
%
%   Algorithm:
%       1. Compute Sobel Gx, Gy
%       2. M = sqrt(Gx^2 + Gy^2)
%       3. θ = atan2d(Gy, Gx) + 360 (to map to [0°, 360°))
%       4. hist[bin] += M  (magnitude-weighted)
%
%   Args:
%       gray     : Grayscale face ROI (uint8, H×W)
%       num_bins : Orientation bins (default 36 → 10° per bin)
%   Returns:
%       hist_grad: Normalized gradient histogram (double, 1×num_bins)

    if nargin < 2, num_bins = 36; end

    [~, gx, gy] = sobel_edges(gray);
    magnitude   = sqrt(gx.^2 + gy.^2);
    orientation = mod(atan2d(gy, gx) + 360, 360);  % [0°, 360°)

    % Magnitude-weighted histogram
    edges = linspace(0, 360, num_bins + 1);
    hist_grad = zeros(1, num_bins);
    for b = 1:num_bins
        mask = orientation >= edges(b) & orientation < edges(b+1);
        hist_grad(b) = sum(magnitude(mask));
    end
    total = sum(hist_grad) + 1e-10;
    hist_grad = hist_grad / total;
end


function hog_feat = compute_hog(gray, cell_size, num_bins)
% COMPUTE_HOG  Histogram of Oriented Gradients (HOG) descriptor.
%
%   Divides the image into cell_size × cell_size pixel cells.
%   For each cell, computes an unsigned gradient orientation histogram
%   (0°–180°, num_bins bins). Normalizes overlapping 2×2 cell blocks
%   using L2 normalization for illumination robustness.
%
%   The concatenated block histograms form the HOG descriptor — a compact
%   and discriminative representation of local facial structure.
%
%   Args:
%       gray      : Grayscale face ROI (uint8, H×W)
%       cell_size : Pixel dimension of each cell (default 8)
%       num_bins  : Orientation bins per cell (default 9)
%   Returns:
%       hog_feat  : HOG feature vector (double, 1×N)

    if nargin < 2, cell_size = 8; end
    if nargin < 3, num_bins  = 9; end

    [H, W] = size(gray);
    ncx = floor(W / cell_size);
    ncy = floor(H / cell_size);

    % Compute gradients
    [~, gx, gy] = sobel_edges(gray);
    magnitude   = sqrt(gx.^2 + gy.^2);
    % Unsigned orientation [0°, 180°)
    orientation = mod(atan2d(abs(gy), abs(gx) + 1e-10), 180);

    % Per-cell gradient histogram
    cell_hists = zeros(ncy, ncx, num_bins);
    bin_width  = 180 / num_bins;

    for cy = 1:ncy
        for cx = 1:ncx
            y0 = (cy-1)*cell_size + 1;
            y1 = cy * cell_size;
            x0 = (cx-1)*cell_size + 1;
            x1 = cx * cell_size;

            cell_mag = magnitude(y0:y1, x0:x1);
            cell_ori = orientation(y0:y1, x0:x1);

            for b = 1:num_bins
                lo = (b-1)*bin_width;
                hi = b * bin_width;
                mask = cell_ori >= lo & cell_ori < hi;
                cell_hists(cy, cx, b) = sum(cell_mag(mask));
            end
        end
    end

    % Block normalization (2×2 cells per block, stride=1)
    block_size = 2;
    hog_feat = [];
    for by = 1:ncy-block_size+1
        for bx = 1:ncx-block_size+1
            block = cell_hists(by:by+block_size-1, bx:bx+block_size-1, :);
            block_vec = block(:)';
            norm = sqrt(sum(block_vec.^2)) + 1e-10;
            hog_feat = [hog_feat, block_vec / norm]; %#ok<AGROW>
        end
    end

    if isempty(hog_feat)
        hog_feat = cell_hists(:)';
    end
end


function template = extract_feature_template(face_roi)
% EXTRACT_FEATURE_TEMPLATE  Extract combined multi-descriptor feature vector.
%
%   Concatenates three complementary descriptors:
%       1. LBP histogram  (256-dim)  — micro-texture
%       2. Gradient hist  (36-dim)   — shape/orientation
%       3. HOG descriptor (variable) — local structure
%
%   The combined template is more robust than any single descriptor.
%
%   Args:
%       face_roi : Preprocessed grayscale face ROI (uint8, H×W)
%   Returns:
%       template : Combined feature vector (double, 1×N)

    lbp_feat  = compute_lbp_histogram(face_roi, 256);
    grad_feat = compute_gradient_histogram(face_roi, 36);
    hog_feat  = compute_hog(face_roi, 8, 9);

    template  = [lbp_feat, grad_feat, hog_feat];
end


%% =========================================================================
%%  MODULE 3: DATABASE & STORAGE (MAT-file based)
%% =========================================================================

function db = DB_initialize(db_path)
% DB_INITIALIZE  Initialize or load the face database (MAT-file).
%
%   The database is a MATLAB struct array stored as a .mat file.
%   Schema:
%       db.path     — file path
%       db.users    — struct array with fields:
%                         .id           (uint32)
%                         .name         (char)
%                         .enrolled_at  (char, ISO-8601)
%                         .template     (double vector)
%       db.next_id  — auto-increment counter
%
%   Args:
%       db_path : Path to the .mat database file (default 'face_db.mat')
%   Returns:
%       db      : Database struct

    if nargin < 1, db_path = 'face_db.mat'; end

    if isfile(db_path)
        loaded = load(db_path, 'db');
        db = loaded.db;
        fprintf('[DB] Loaded existing database: %s (%d users)\n', ...
                db_path, length(db.users));
    else
        db.path    = db_path;
        db.users   = struct('id', {}, 'name', {}, ...
                            'enrolled_at', {}, 'template', {});
        db.next_id = 1;
        save(db_path, 'db');
        fprintf('[DB] Created new database: %s\n', db_path);
    end
end


function db = DB_save(db)
% DB_SAVE  Persist the database struct to its MAT-file.
%
%   Args:
%       db : Database struct (with db.path set)
%   Returns:
%       db : Unchanged database struct (for chaining)

    save(db.path, 'db');
end


function [db, user_id] = DB_enroll_user(db, name, template)
% DB_ENROLL_USER  Enroll a new user in the database.
%
%   Args:
%       db       : Database struct
%       name     : User's name (char)
%       template : Feature vector (double, 1×N)
%   Returns:
%       db       : Updated database struct
%       user_id  : Auto-assigned integer ID

    user_id = db.next_id;
    new_user.id          = user_id;
    new_user.name        = name;
    new_user.enrolled_at = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
    new_user.template    = template;

    db.users(end+1) = new_user;
    db.next_id = db.next_id + 1;
    db = DB_save(db);

    fprintf('[DB] Enrolled "%s" (ID=%d, template_len=%d)\n', ...
            name, user_id, length(template));
end


function DB_list_users(db)
% DB_LIST_USERS  Print a table of all enrolled users.
%
%   Args:
%       db : Database struct

    if isempty(db.users)
        fprintf('[DB] No users enrolled.\n');
        return;
    end
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('%-6s  %-20s  %-22s  %s\n', 'ID', 'Name', 'Enrolled At', 'Tmpl Len');
    fprintf('%s\n', repmat('=', 1, 60));
    for i = 1:length(db.users)
        u = db.users(i);
        fprintf('%-6d  %-20s  %-22s  %d\n', ...
                u.id, u.name, u.enrolled_at, length(u.template));
    end
    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('Total: %d user(s)\n\n', length(db.users));
end


function db = DB_delete_user(db, user_id)
% DB_DELETE_USER  Remove a user from the database by ID.
%
%   Args:
%       db      : Database struct
%       user_id : Integer ID of the user to delete
%   Returns:
%       db      : Updated database struct

    idx = find([db.users.id] == user_id);
    if isempty(idx)
        fprintf('[DB] User ID=%d not found.\n', user_id);
    else
        db.users(idx) = [];
        db = DB_save(db);
        fprintf('[DB] Deleted user ID=%d.\n', user_id);
    end
end


%% =========================================================================
%%  MODULE 4: MATCHING & IDENTIFICATION
%% =========================================================================

function dist = euclidean_distance(vec_a, vec_b)
% EUCLIDEAN_DISTANCE  Compute L2 (Euclidean) distance between two vectors.
%
%   Definition: d(a,b) = sqrt(sum((a_i - b_i)^2))
%
%   This is the primary metric for template matching. Smaller values
%   indicate more similar faces.
%
%   Args:
%       vec_a : Query feature vector (double, 1×N)
%       vec_b : Stored feature vector (double, 1×N)
%   Returns:
%       dist  : Scalar Euclidean distance

    dist = sqrt(sum((double(vec_a) - double(vec_b)).^2));
end


function sim = cosine_similarity(vec_a, vec_b)
% COSINE_SIMILARITY  Compute cosine similarity between two vectors.
%
%   Definition: sim(a,b) = (a · b) / (||a|| × ||b||)
%
%   Range: [0,1] for non-negative features. Higher = more similar.
%   Invariant to vector magnitude (illumination scale changes).
%
%   Args:
%       vec_a : Query feature vector (double, 1×N)
%       vec_b : Stored feature vector (double, 1×N)
%   Returns:
%       sim   : Cosine similarity score in [0,1]

    a = double(vec_a);
    b = double(vec_b);
    sim = dot(a, b) / (norm(a) * norm(b) + 1e-10);
end


function result = identify_face(db, query_img, threshold)
% IDENTIFY_FACE  Identify the person in a query image against the database.
%
%   Full pipeline:
%       1. Preprocess query image (grayscale → CLAHE → blur → resize)
%       2. Extract feature template (LBP + gradient hist + HOG)
%       3. Compute Euclidean distance to every stored template
%       4. Sort results by distance
%       5. If min_distance < threshold: return matched identity
%          Else: return "Unknown"
%
%   Args:
%       db        : Database struct
%       query_img : Query image (uint8, H×W×3 or H×W)
%       threshold : Euclidean distance threshold (default 8.0)
%   Returns:
%       result    : Struct with fields:
%                       .matched  (logical)
%                       .name     (char)
%                       .user_id  (int or [])
%                       .distance (double)
%                       .similarity (double)

    if nargin < 3, threshold = 8.0; end

    % Default result
    result.matched    = false;
    result.name       = 'Unknown';
    result.user_id    = [];
    result.distance   = inf;
    result.similarity = 0;

    % Step 1: Preprocess
    face_roi = detect_and_crop_face(query_img, [100, 100]);
    if isempty(face_roi)
        fprintf('[IDENTIFY] No face detected in query image.\n');
        return;
    end

    % Step 2: Extract features
    query_template = extract_feature_template(face_roi);

    % Step 3: Check database
    if isempty(db.users)
        fprintf('[IDENTIFY] Database is empty. Enroll users first.\n');
        return;
    end

    % Step 4: Compute distances to all stored templates
    n_users  = length(db.users);
    dists    = zeros(1, n_users);
    sims     = zeros(1, n_users);

    for i = 1:n_users
        stored = db.users(i).template;
        % Handle template dimension mismatch (re-enrollment may change length)
        len = min(length(query_template), length(stored));
        if len == 0, continue; end
        dists(i) = euclidean_distance(query_template(1:len), stored(1:len));
        sims(i)  = cosine_similarity (query_template(1:len), stored(1:len));
    end

    % Step 5: Find best match
    [min_dist, best_idx] = min(dists);

    fprintf('\n--- Identification Results ---\n');
    fprintf('%-20s %12s %12s\n', 'Name', 'Distance', 'Similarity');
    fprintf('%s\n', repmat('-', 1, 46));

    % Display all scores (sorted)
    [sorted_dists, sort_idx] = sort(dists);
    for k = 1:min(5, n_users)  % Show top-5
        u = db.users(sort_idx(k));
        marker = '';
        if sort_idx(k) == best_idx && min_dist < threshold
            marker = ' ← MATCHED';
        end
        fprintf('%-20s %12.4f %12.4f%s\n', ...
                u.name, sorted_dists(k), sims(sort_idx(k)), marker);
    end
    fprintf('%s\n', repmat('-', 1, 46));

    % Apply threshold decision
    if min_dist < threshold
        result.matched    = true;
        result.name       = db.users(best_idx).name;
        result.user_id    = db.users(best_idx).id;
        result.distance   = min_dist;
        result.similarity = sims(best_idx);
        fprintf('IDENTITY: %s  (distance=%.4f < threshold=%.1f)\n', ...
                result.name, min_dist, threshold);
    else
        fprintf('IDENTITY: Unknown  (distance=%.4f >= threshold=%.1f)\n', ...
                min_dist, threshold);
    end
    fprintf('\n');
end


%% =========================================================================
%%  ENROLLMENT WRAPPER
%% =========================================================================

function db = enroll_user(db, name, img)
% ENROLL_USER  Preprocess an image and enroll the face template in the DB.
%
%   Args:
%       db   : Database struct
%       name : User's name (char)
%       img  : Face image (uint8, H×W×3 or H×W)
%   Returns:
%       db   : Updated database struct

    face_roi = detect_and_crop_face(img, [100, 100]);
    if isempty(face_roi)
        fprintf('[ENROLL] No face detected. Enrollment aborted.\n');
        return;
    end

    template = extract_feature_template(face_roi);
    [db, uid] = DB_enroll_user(db, name, template);
    fprintf('[ENROLL] "%s" enrolled successfully (ID=%d).\n', name, uid);
end


%% =========================================================================
%%  MODULE 5: VISUALIZATION — PREPROCESSING PIPELINE DEMO
%% =========================================================================

function run_preprocessing_demo()
% RUN_PREPROCESSING_DEMO  Visualize all preprocessing stages on a synthetic face.
%
%   Generates a synthetic face-like image, applies each preprocessing
%   stage, and plots the results in a tiled figure for visual inspection.
%   Runs entirely without external images or webcam.

    fprintf('\n[DEMO] Running preprocessing pipeline demo...\n');

    % ---- Generate synthetic face-like image ----
    H = 200; W = 200;
    face_rgb = uint8(220 * ones(H, W, 3));  % White background

    % Face oval
    [Xg, Yg] = meshgrid(1:W, 1:H);
    cx = W/2; cy = H/2;
    face_mask = ((Xg-cx)/70).^2 + ((Yg-cy)/85).^2 <= 1;
    for c = 1:3
        chan = double(face_rgb(:,:,c));
        skin = [180, 160, 140];
        chan(face_mask) = skin(c);
        face_rgb(:,:,c) = uint8(chan);
    end

    % Left eye
    eye_mask1 = ((Xg-(cx-22))/14).^2 + ((Yg-(cy-20))/9).^2 <= 1;
    % Right eye
    eye_mask2 = ((Xg-(cx+22))/14).^2 + ((Yg-(cy-20))/9).^2 <= 1;
    % Apply dark color
    for c = 1:3
        chan = double(face_rgb(:,:,c));
        eye_color = [40, 50, 60];
        chan(eye_mask1 | eye_mask2) = eye_color(c);
        face_rgb(:,:,c) = uint8(chan);
    end

    % ---- Apply preprocessing stages ----
    gray_img    = convert_to_grayscale(face_rgb);
    eq_img      = histogram_equalization(gray_img);
    yuv_img     = convert_to_yuv(face_rgb);
    clahe_img   = apply_clahe(gray_img, 0.01, [8,8]);
    gauss_img   = gaussian_blur(gray_img, 1.5, 5);
    median_img  = median_filter_2d(gray_img, 3);
    [sobel_img] = sobel_edges(gray_img);
    canny_img   = uint8(canny_edge_detection(gray_img) * 255);

    % ---- Plot results ----
    figure('Name', 'Face Recognition — Preprocessing Pipeline', ...
           'NumberTitle', 'off', 'Color', [0.1 0.1 0.1], ...
           'Position', [100 100 900 600]);

    stages = {face_rgb,   'Original RGB',       [];
              gray_img,   'Grayscale (Y)',       gray;
              eq_img,     'Hist. Equalized',     gray;
              yuv_img(:,:,1), 'YUV — Y Channel', gray;
              clahe_img,  'CLAHE',               gray;
              gauss_img,  'Gaussian Blur σ=1.5', gray;
              median_img, 'Median Filter 3×3',   gray;
              sobel_img,  'Sobel Edges',         gray;
              canny_img,  'Canny Edges',         gray};

    n_stages = size(stages, 1);
    cols = 3; rows = ceil(n_stages / cols);

    for i = 1:n_stages
        subplot(rows, cols, i);
        img_data = stages{i, 1};
        if ndims(img_data) == 3
            imshow(img_data);
        else
            imshow(img_data, []);
        end
        title(stages{i, 2}, 'Color', [0.9 0.9 0.9], ...
              'FontSize', 9, 'FontWeight', 'bold');
    end

    sgtitle('Preprocessing Pipeline — Face Recognition System', ...
            'Color', 'w', 'FontSize', 13, 'FontWeight', 'bold');

    % ---- Feature Extraction Summary ----
    face_roi  = imresize(gray_img, [100, 100]);
    lbp_hist  = compute_lbp_histogram(face_roi, 256);
    grad_hist = compute_gradient_histogram(face_roi, 36);

    figure('Name', 'Feature Extraction', 'NumberTitle', 'off', ...
           'Color', [0.1 0.1 0.1], 'Position', [200 200 800 300]);

    subplot(1,2,1);
    bar(lbp_hist, 'FaceColor', [0.2 0.7 0.4], 'EdgeColor', 'none');
    title('LBP Histogram (256 bins)', 'Color', 'w', 'FontWeight', 'bold');
    xlabel('LBP Code', 'Color', [0.7 0.7 0.7]);
    ylabel('Normalized Frequency', 'Color', [0.7 0.7 0.7]);
    set(gca, 'Color', [0.15 0.15 0.15], 'XColor', 'w', 'YColor', 'w');

    subplot(1,2,2);
    bar(grad_hist, 'FaceColor', [0.3 0.5 0.9], 'EdgeColor', 'none');
    title('Gradient Orientation Histogram (36 bins)', 'Color', 'w', 'FontWeight', 'bold');
    xlabel('Orientation Bin (0°–360°)', 'Color', [0.7 0.7 0.7]);
    ylabel('Normalized Magnitude', 'Color', [0.7 0.7 0.7]);
    set(gca, 'Color', [0.15 0.15 0.15], 'XColor', 'w', 'YColor', 'w');

    fprintf('[DEMO] Preprocessing demo complete. Figures displayed.\n');
    fprintf('[DEMO] Template length: %d dimensions\n', ...
            length(lbp_hist) + length(grad_hist));
end


%% =========================================================================
%%  HELPER: Generate Synthetic Face Image
%% =========================================================================

function img = generate_synthetic_face()
% GENERATE_SYNTHETIC_FACE  Create a programmatic face image for testing.
%
%   Returns a 200×200 RGB uint8 image resembling a simplified face.
%   Used when no real image is available (MATLAB Online demo mode).

    H = 200; W = 200;
    img = uint8(220 * ones(H, W, 3));

    [Xg, Yg] = meshgrid(1:W, 1:H);
    cx = W/2; cy = H/2;

    % Face oval
    mask = ((Xg-cx)/70).^2 + ((Yg-cy)/85).^2 <= 1;
    for c = 1:3
        ch = double(img(:,:,c));
        s = [180,160,140];
        ch(mask) = s(c);
        img(:,:,c) = uint8(ch);
    end

    % Eyes
    em1 = ((Xg-(cx-22))/14).^2 + ((Yg-(cy-20))/9).^2 <= 1;
    em2 = ((Xg-(cx+22))/14).^2 + ((Yg-(cy-20))/9).^2 <= 1;
    for c = 1:3
        ch = double(img(:,:,c));
        ec = [40,50,60];
        ch(em1 | em2) = ec(c);
        img(:,:,c) = uint8(ch);
    end

    % Nose (triangle approximation)
    nm = ((Xg-cx).^2 / 64 + (Yg-(cy+15)).^2 / 144) <= 1;
    nm = nm & (Yg > cy);
    for c = 1:3
        ch = double(img(:,:,c));
        nc = [150,130,110];
        ch(nm) = nc(c);
        img(:,:,c) = uint8(ch);
    end

    % Mouth
    mm = ((Xg-cx)/22).^2 + ((Yg-(cy+40))/10).^2 <= 1;
    mm = mm & (Yg > cy+40);
    for c = 1:3
        ch = double(img(:,:,c));
        mc = [100,80,60];
        ch(mm) = mc(c);
        img(:,:,c) = uint8(ch);
    end
end
