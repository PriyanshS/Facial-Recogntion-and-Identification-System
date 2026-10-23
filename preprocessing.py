"""
=============================================================================
MODULE 1: Preprocessing & Normalization
=============================================================================
Implements classical digital image processing techniques:
  - Color space conversion (RGB → Grayscale / YUV / HSI)
  - Histogram equalization for illumination normalization
  - Spatial filtering (Gaussian blur, Median filter) for noise reduction
  - Gradient-based edge detection (Sobel + Canny operators)
  - ROI localization & cropping via thresholding
=============================================================================
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1.1  Color Space Conversions
# ---------------------------------------------------------------------------

def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image (OpenCV default) to grayscale using the
    standard luminance-weighted formula:
        Y = 0.299*R + 0.587*G + 0.114*B
    This separates luminance from chrominance information.

    Args:
        image: Input BGR image (uint8, H×W×3).

    Returns:
        Grayscale image (uint8, H×W).
    """
    # OpenCV stores images as BGR, not RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_yuv(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to YUV color space.
    Y  = luminance (brightness)
    U,V = chrominance (color difference) channels

    The Y channel alone is equivalent to grayscale and is robust to
    lighting changes — useful for normalizing illumination.

    Args:
        image: Input BGR image (uint8, H×W×3).

    Returns:
        YUV image (uint8, H×W×3).  Channel order: [Y, U, V].
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


def bgr_to_hsi(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to HSI (Hue, Saturation, Intensity) color space
    using manual formula — not available directly in OpenCV.

    HSI separates chromatic content (H, S) from intensity (I), making
    the Intensity channel ideal for illumination-robust processing.

    Args:
        image: Input BGR image (uint8, H×W×3).

    Returns:
        HSI image as float32 array (H×W×3). Values in [0,1] for H,S,I.
    """
    # Normalize to [0, 1]
    img = image.astype(np.float32) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Intensity channel
    I = (R + G + B) / 3.0

    # Saturation channel (avoid division by zero)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.where(I > 0, 1.0 - (min_rgb / (I + 1e-10)), 0.0)

    # Hue channel via arc-cosine formula
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)  # Normalize to [0, 1]

    hsi = np.stack([H, S, I], axis=-1).astype(np.float32)
    return hsi


# ---------------------------------------------------------------------------
# 1.2  Illumination Normalization — Histogram Equalization
# ---------------------------------------------------------------------------

def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image to redistribute
    pixel intensity values across the full [0, 255] range.

    This compensates for uneven illumination (e.g., shadows, over-exposure)
    and significantly improves face recognition under varying lighting.

    Algorithm:
        1. Compute the normalized intensity histogram H[i].
        2. Compute cumulative distribution function (CDF).
        3. Map each pixel value through: s = round((L-1) * CDF(r))

    Args:
        gray: Grayscale image (uint8, H×W).

    Returns:
        Equalized grayscale image (uint8, H×W).
    """
    return cv2.equalizeHist(gray)


def clahe_equalization(gray: np.ndarray,
                       clip_limit: float = 2.0,
                       tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Unlike global histogram equalization, CLAHE operates on small local
    tiles, preventing over-amplification of noise in uniform regions.
    'clip_limit' caps the histogram to avoid excessive contrast in
    near-constant regions — essential for facial detail preservation.

    Args:
        gray:       Grayscale image (uint8, H×W).
        clip_limit: Threshold for contrast limiting (default 2.0).
        tile_size:  Size of the grid tiles (default 8×8).

    Returns:
        CLAHE-enhanced grayscale image (uint8, H×W).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def homomorphic_filter(gray: np.ndarray,
                       d0: float = 20.0,
                       gamma_low: float = 0.75,
                       gamma_high: float = 1.25,
                       c: float = 1.0) -> np.ndarray:
    """
    Apply Homomorphic Filtering in the 2D Frequency Domain (FFT).

    Decouples non-uniform illumination from facial reflectance:
        I(x,y) = L(x,y) * R(x,y)
        ln(I + 1) = ln(L) + ln(R)

    In the Fourier domain, low frequencies correspond to slowly varying
    illumination L(x,y), while high frequencies correspond to facial
    reflectance R(x,y) (bone structure, eyes, lips).

    Filter Transfer Function (High-Emphasis Gaussian High-Pass):
        H(u,v) = (gamma_high - gamma_low) * [1 - exp(-c * D^2 / D0^2)] + gamma_low

    This suppresses lighting shifts (gamma_low < 1.0) and boosts invariant
    facial structural reflectance (gamma_high > 1.0).

    Args:
        gray:       Input grayscale image (uint8, H×W).
        d0:         Cutoff frequency parameter (default 20.0).
        gamma_low:  Low-frequency gain (default 0.75, attenuates illumination).
        gamma_high: High-frequency gain (default 1.25, enhances reflectance).
        c:          Slope / steepness factor (default 1.0).

    Returns:
        Illumination-normalized grayscale image (uint8, H×W).
    """
    img_f = gray.astype(np.float64)
    img_log = np.log1p(img_f)

    # 2D Fast Fourier Transform with DC-center shift
    fft = np.fft.fft2(img_log)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = gray.shape
    u = np.arange(rows) - rows / 2.0
    v = np.arange(cols) - cols / 2.0
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    # High-emphasis filter transfer function
    H = (gamma_high - gamma_low) * (1.0 - np.exp(-c * (D**2) / (d0**2 + 1e-10))) + gamma_low
    filtered = fft_shift * H

    # Inverse Fourier Transform back to spatial domain
    ifft_shift = np.fft.ifftshift(filtered)
    ifft = np.fft.ifft2(ifft_shift)
    exp_img = np.expm1(np.real(ifft))

    # Robust percentile-based dynamic range contrast stretching [0, 255]
    p_low, p_high = np.percentile(exp_img, (1, 99))
    stretched = np.clip((exp_img - p_low) / (p_high - p_low + 1e-10), 0.0, 1.0) * 255.0
    return stretched.astype(np.uint8)


# ---------------------------------------------------------------------------
# 1.3  Spatial Filtering (Noise Reduction)
# ---------------------------------------------------------------------------

def gaussian_blur(gray: np.ndarray,
                  kernel_size: int = 5,
                  sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian low-pass filter for noise smoothing.

    The Gaussian kernel acts as a low-pass filter in the frequency domain,
    attenuating high-frequency noise while preserving low-frequency structure.
    Kernel values are sampled from: G(x,y) = exp(-(x²+y²)/(2σ²))

    Args:
        gray:        Grayscale input image (uint8, H×W).
        kernel_size: Size of the square Gaussian kernel (must be odd).
        sigma:       Standard deviation of the Gaussian distribution.

    Returns:
        Smoothed grayscale image (uint8, H×W).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Enforce odd kernel size
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)


def median_filter(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Median filter — a non-linear spatial filter.

    Replaces each pixel with the median value in its neighborhood.
    Particularly effective against salt-and-pepper noise because the
    median is robust to outlier pixel values, unlike Gaussian blur.

    Args:
        gray:        Grayscale input image (uint8, H×W).
        kernel_size: Size of the square filter window (must be odd).

    Returns:
        Median-filtered image (uint8, H×W).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(gray, kernel_size)


# ---------------------------------------------------------------------------
# 1.4  Edge Detection (Gradient-based)
# ---------------------------------------------------------------------------

def sobel_edges(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Compute gradient magnitude using Sobel operators.

    Sobel uses separable 3×3 derivative kernels to approximate the image
    gradient in the X and Y directions:
        Gx = [-1 0 +1; -2 0 +2; -1 0 +1]
        Gy = [-1 -2 -1;  0  0  0; +1 +2 +1]
    Magnitude: |G| = sqrt(Gx² + Gy²)

    High gradient magnitude reveals edges = facial feature boundaries.

    Args:
        gray:  Grayscale image (uint8, H×W).
        ksize: Kernel size for Sobel (1, 3, 5, or 7).

    Returns:
        Edge magnitude image (uint8, H×W).
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Normalize to [0, 255]
    magnitude = np.uint8(255 * magnitude / (magnitude.max() + 1e-10))
    return magnitude


def canny_edges(gray: np.ndarray,
                low_thresh: int = 50,
                high_thresh: int = 150) -> np.ndarray:
    """
    Apply the Canny edge detector — a multi-stage optimal edge detector.

    Pipeline:
        1. Gaussian smoothing (noise reduction)
        2. Gradient computation (Sobel)
        3. Non-maximum suppression (thin edges to 1-pixel width)
        4. Double thresholding (strong / weak edges)
        5. Edge tracking via hysteresis (connectivity)

    The result is a clean binary edge map highlighting facial contours.

    Args:
        gray:         Grayscale image (uint8, H×W).
        low_thresh:   Lower threshold for hysteresis linking.
        high_thresh:  Upper threshold for strong edge detection.

    Returns:
        Binary edge map (uint8, H×W). Edge pixels = 255.
    """
    return cv2.Canny(gray, low_thresh, high_thresh)


# ---------------------------------------------------------------------------
# 1.5  Face Detection & ROI Cropping (Haar Cascade)
# ---------------------------------------------------------------------------

# Load OpenCV's pre-trained Haar Cascade detectors
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def _find_best_eye_pair(eyes, fw: int, fh: int):
    """
    Filter candidate eye detections to identify the true left/right eye centers
    based on human facial anthropometry.
    """
    valid_eyes = []
    for (ex, ey, ew, eh) in eyes:
        cx = ex + ew / 2.0
        cy = ey + eh / 2.0
        if 0.15 * fh <= cy <= 0.65 * fh:
            valid_eyes.append((cx, cy, ew, eh))
    if len(valid_eyes) < 2:
        return None

    best_pair = None
    best_score = float('inf')
    for i in range(len(valid_eyes)):
        for j in range(i + 1, len(valid_eyes)):
            e1, e2 = valid_eyes[i], valid_eyes[j]
            left, right = (e1, e2) if e1[0] < e2[0] else (e2, e1)
            # Left eye in left half (x < 0.58 fw), right eye in right half (x > 0.42 fw)
            if not (left[0] < 0.58 * fw and right[0] > 0.42 * fw):
                continue
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            dist = np.sqrt(dx**2 + dy**2)
            # Inter-pupillary distance should be roughly 24% to 60% of face width
            if not (0.24 * fw <= dist <= 0.60 * fw):
                continue
            # Score penalizes deviations from nominal 38% width separation and tilt
            score = abs(dist - 0.38 * fw) + abs(dy) * 1.2
            if score < best_score:
                best_score = score
                best_pair = (left, right)
    return best_pair


def detect_and_crop_face(image: np.ndarray,
                         target_size: tuple = (128, 128),
                         scale_factor: float = 1.1,
                         min_neighbors: int = 4) -> np.ndarray | None:
    """
    Detect the primary face in an image and return the normalized,
    canonically cropped and de-rotated face ROI.

    Handles Tilted Faces & Head Roll:
        1. Multi-angle Detection Fallback: If upright Haar cascade detects 0 faces
           (e.g., head tilted > 12°), tests rotated views (±15°, ±25°).
        2. Eye-Line Affine De-Rotation: Detects left and right eye coordinates in the
           upper facial region. Calculates the tilt angle θ = atan2(dy, dx) and rotates
           the face to bring eyes to a perfectly horizontal canonical level.

    Handles Bearded & Shaven Faces:
        Isolates the inner facial oval (focusing on the immutable periocular region:
        eyes, eyebrows, glabella, and upper nose bridge), trimming outer hair, ears,
        and jaw boundaries that change between clean-shaven and bearded states.

    Args:
        image:         BGR or grayscale input image.
        target_size:   Output (width, height) of the cropped face (default 128×128).
        scale_factor:  Scale factor for the detection pyramid.
        min_neighbors: Minimum neighbor rectangles for a valid detection.

    Returns:
        Preprocessed face ROI as uint8 array (target_size[1] × target_size[0]),
        or None if no face is detected.
    """
    # Ensure grayscale for detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Upright frontal face detection
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(50, 50)
    )

    # Step 2: Multi-angle detection fallback for tilted faces (> 12° head roll)
    if len(faces) == 0:
        h_img, w_img = gray.shape
        center = (w_img / 2.0, h_img / 2.0)
        for ang in [-22.0, -12.0, 12.0, 22.0]:
            M_ang = cv2.getRotationMatrix2D(center, ang, 1.0)
            rot_gray = cv2.warpAffine(gray, M_ang, (w_img, h_img), borderMode=cv2.BORDER_REPLICATE)
            rot_faces = _FACE_CASCADE.detectMultiScale(rot_gray, 1.1, 3, minSize=(45, 45))
            if len(rot_faces) > 0:
                gray = rot_gray
                faces = rot_faces
                break

    if len(faces) == 0:
        return None  # No face detected

    # Select the most prominent face (largest area)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    fx, fy, fw, fh = faces[0]

    # Step 3: Eye-line affine de-rotation (precise geometric alignment)
    eyes_roi = gray[fy : fy + int(fh * 0.60), fx : fx + fw]
    eyes = _EYE_CASCADE.detectMultiScale(eyes_roi, scaleFactor=1.05, minNeighbors=3, minSize=(16, 16))

    aligned = None
    pair = _find_best_eye_pair(eyes, fw, fh)
    if pair is not None:
        left, right = pair
        p1 = (fx + left[0], fy + left[1])
        p2 = (fx + right[0], fy + right[1])
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = np.sqrt(dx**2 + dy**2)
        eye_angle = np.degrees(np.arctan2(dy, dx))
        desired_dist = target_size[0] * 0.36
        scale = desired_dist / dist
        eyes_center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

        # De-rotate and scale so eyes are horizontally level at fixed canonical position (y=38%)
        M = cv2.getRotationMatrix2D(eyes_center, eye_angle, scale)
        M[0, 2] += (target_size[0] * 0.50 - eyes_center[0])
        M[1, 2] += (target_size[1] * 0.38 - eyes_center[1])
        aligned = cv2.warpAffine(gray, M, target_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Fallback to standardized inner-face boundary if eye alignment is unavailable
    if aligned is None:
        cy = int(max(0, fy + fh * 0.10))
        ch = int(min(gray.shape[0] - cy, fh * 0.82))
        cx = int(max(0, fx + fw * 0.08))
        cw = int(min(gray.shape[1] - cx, fw * 0.84))
        roi = gray[cy : cy + ch, cx : cx + cw]
        aligned = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

    # Step 4: Frequency-domain homomorphic filtering + CLAHE + Gaussian smoothing
    aligned = homomorphic_filter(aligned, d0=20.0, gamma_low=0.75, gamma_high=1.25)
    aligned = clahe_equalization(aligned, clip_limit=2.0, tile_size=(8, 8))
    aligned = gaussian_blur(aligned, kernel_size=3, sigma=0.6)

    return aligned


def full_preprocess_pipeline(image: np.ndarray,
                             target_size: tuple = (100, 100)) -> np.ndarray | None:
    """
    Run the complete preprocessing pipeline on a raw input image.

    Pipeline stages:
        1. Grayscale conversion (luminance extraction)
        2. Face detection & ROI cropping
        3. CLAHE illumination normalization
        4. Gaussian spatial filtering
        5. Fixed-size resizing

    Args:
        image:       Input BGR image from webcam or file.
        target_size: Target (width, height) for the output face image.

    Returns:
        Normalized grayscale face image (uint8, H×W), or None if no face found.
    """
    return detect_and_crop_face(image, target_size=target_size)
