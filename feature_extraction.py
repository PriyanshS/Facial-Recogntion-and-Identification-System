"""
=============================================================================
MODULE 2: Feature Extraction & Template Generation
=============================================================================
Implements classical computer vision feature extraction techniques:
  - LBP  (Local Binary Patterns)  — texture descriptor
  - Sobel gradient histogram       — edge/shape descriptor
  - HOG  (Histogram of Oriented Gradients) — structural descriptor
  - Pixel intensity vector         — baseline raw descriptor
  - Combined feature concatenation — robust multi-descriptor template
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 2.1  Local Binary Pattern (LBP) — Texture Descriptor
# ---------------------------------------------------------------------------

def compute_lbp(gray: np.ndarray,
                radius: int = 1,
                num_points: int = 8,
                num_bins: int = 256) -> np.ndarray:
    """
    Compute the Local Binary Pattern (LBP) texture descriptor.

    LBP encodes the local texture of a pixel by comparing it to its
    circular neighborhood of 'num_points' neighbors at radius 'radius'.
    For each neighbor, set bit = 1 if neighbor ≥ center, else 0.
    The resulting binary pattern (0–255 for 8-bit) represents local texture.

    LBP histogram captures the distribution of micro-textures across the
    face (skin texture, wrinkles, edge patterns) — highly discriminative
    and partially illumination-invariant.

    Algorithm:
        For each pixel (x,y):
            pattern = 0
            For each neighbor k at angle θk = 2π*k/P:
                xk = x + R*cos(θk),  yk = y - R*sin(θk)
                if I(xk,yk) ≥ I(x,y): pattern |= (1 << k)
            LBP[x,y] = pattern

    Args:
        gray:      Preprocessed grayscale face image (uint8, H×W).
        radius:    Radius of the circular neighborhood.
        num_points: Number of sampling points on the circle.
        num_bins:  Number of histogram bins (2^num_points for full LBP).

    Returns:
        Normalized LBP histogram as float32 array of length 'num_bins'.
    """
    rows, cols = gray.shape
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)

    # Precompute neighbor offsets for the circular neighborhood
    angles = [2 * np.pi * k / num_points for k in range(num_points)]
    offsets = [(radius * np.cos(a), -radius * np.sin(a)) for a in angles]

    for y in range(radius, rows - radius):
        for x in range(radius, cols - radius):
            center = int(gray[y, x])
            pattern = 0
            for k, (dy, dx) in enumerate(offsets):
                # Bilinear interpolation for non-integer neighbor coordinates
                ny, nx = y + dy, x + dx
                # Round to nearest integer for simplicity (fast version)
                iny, inx = int(round(ny)), int(round(nx))
                # Clamp to image boundaries
                iny = np.clip(iny, 0, rows - 1)
                inx = np.clip(inx, 0, cols - 1)
                neighbor = int(gray[iny, inx])
                if neighbor >= center:
                    pattern |= (1 << k)
            lbp_image[y, x] = pattern

    # Compute normalized histogram of LBP codes
    hist, _ = np.histogram(lbp_image.ravel(), bins=num_bins,
                           range=(0, num_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-10)  # Normalize to probability distribution

    return hist


def compute_lbp_fast(gray: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """
    Fast LBP using vectorized NumPy operations (8-neighbor, radius=1).

    Uses bit-shift operations on array slices for the 8 neighbors:
        N0 = top-left, N1 = top, N2 = top-right, N3 = right,
        N4 = bottom-right, N5 = bottom, N6 = bottom-left, N7 = left

    This avoids the nested Python loop in compute_lbp() for real-time use.

    Args:
        gray:     Grayscale face image (uint8, H×W).
        num_bins: Number of histogram bins (default 256).

    Returns:
        Normalized LBP histogram (float32, length num_bins).
    """
    g = gray.astype(np.int16)

    # Extract 8 neighbors of each interior pixel
    lbp = np.zeros_like(g, dtype=np.uint8)
    lbp[1:-1, 1:-1]  = (g[0:-2, 0:-2] >= g[1:-1, 1:-1]).astype(np.uint8) * 1
    lbp[1:-1, 1:-1] += (g[0:-2, 1:-1] >= g[1:-1, 1:-1]).astype(np.uint8) * 2
    lbp[1:-1, 1:-1] += (g[0:-2, 2:  ] >= g[1:-1, 1:-1]).astype(np.uint8) * 4
    lbp[1:-1, 1:-1] += (g[1:-1, 2:  ] >= g[1:-1, 1:-1]).astype(np.uint8) * 8
    lbp[1:-1, 1:-1] += (g[2:  , 2:  ] >= g[1:-1, 1:-1]).astype(np.uint8) * 16
    lbp[1:-1, 1:-1] += (g[2:  , 1:-1] >= g[1:-1, 1:-1]).astype(np.uint8) * 32
    lbp[1:-1, 1:-1] += (g[2:  , 0:-2] >= g[1:-1, 1:-1]).astype(np.uint8) * 64
    lbp[1:-1, 1:-1] += (g[1:-1, 0:-2] >= g[1:-1, 1:-1]).astype(np.uint8) * 128

    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-10)
    return hist


# ---------------------------------------------------------------------------
# 2.2  Gradient Histogram (Sobel-based) — Shape/Edge Descriptor
# ---------------------------------------------------------------------------

def compute_gradient_histogram(gray: np.ndarray,
                               num_bins: int = 36) -> np.ndarray:
    """
    Compute a histogram of gradient orientations using Sobel operators.

    This captures the distribution of edge directions in the face image,
    encoding structural information such as the orientation of key facial
    contours (jawline, eyebrows, nose bridge).

    Algorithm:
        1. Compute Sobel gradients Gx, Gy
        2. Compute magnitude M = sqrt(Gx² + Gy²)
        3. Compute orientation θ = atan2(Gy, Gx) in degrees [0°, 360°)
        4. Build a weighted histogram: bin[θ] += M[θ]  (magnitude-weighted)

    Args:
        gray:     Grayscale face image (uint8, H×W).
        num_bins: Number of orientation bins (default 36 → 10° per bin).

    Returns:
        Normalized gradient orientation histogram (float32, length num_bins).
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Convert orientation to degrees in [0, 360)
    orientation = (np.degrees(np.arctan2(grad_y, grad_x)) + 360) % 360

    # Build magnitude-weighted orientation histogram
    hist, _ = np.histogram(orientation.ravel(), bins=num_bins,
                           range=(0, 360), weights=magnitude.ravel())
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-10)
    return hist


# ---------------------------------------------------------------------------
# 2.3  HOG (Histogram of Oriented Gradients) — Structural Descriptor
# ---------------------------------------------------------------------------

def compute_hog_features(gray: np.ndarray,
                         cell_size: int = 8,
                         num_bins: int = 9,
                         block_size: int = 2) -> np.ndarray:
    """
    Compute HOG (Histogram of Oriented Gradients) feature descriptor.

    HOG captures the local shape by dividing the image into small spatial
    cells, computing gradient histograms per cell, then normalizing over
    overlapping blocks for illumination robustness.

    Algorithm:
        1. Divide image into cells of size cell_size × cell_size
        2. For each cell, compute gradient orientation histogram (9 bins, 0-180°)
        3. Group cells into blocks (block_size × block_size cells per block)
        4. L2-normalize each block's combined histogram
        5. Concatenate all block histograms

    This gives a compact yet rich descriptor of facial structure.

    Args:
        gray:       Grayscale face image (uint8, H×W).
        cell_size:  Pixel size of each cell (default 8).
        num_bins:   Orientation bins per cell (default 9 → 20° per bin).
        block_size: Cells per block dimension (default 2 → 2×2 block).

    Returns:
        HOG feature vector (float32, length = blocks * block_size^2 * num_bins).
    """
    h, w = gray.shape
    num_cells_y = h // cell_size
    num_cells_x = w // cell_size

    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Unsigned orientation [0°, 180°) for HOG
    orientation = np.degrees(np.arctan2(np.abs(grad_y), np.abs(grad_x) + 1e-10))

    # Build per-cell histograms
    cell_hists = np.zeros((num_cells_y, num_cells_x, num_bins), dtype=np.float32)
    bin_width = 180.0 / num_bins

    for cy in range(num_cells_y):
        for cx in range(num_cells_x):
            # Extract cell pixels
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            cell_ori = orientation[y0:y1, x0:x1].ravel()

            # Soft assignment to bins (bilinear interpolation)
            for mag, ori in zip(cell_mag, cell_ori):
                bin_idx = ori / bin_width
                lo = int(bin_idx) % num_bins
                hi = (lo + 1) % num_bins
                frac = bin_idx - int(bin_idx)
                cell_hists[cy, cx, lo] += mag * (1 - frac)
                cell_hists[cy, cx, hi] += mag * frac

    # Block normalization (L2-norm)
    feature_vector = []
    num_blocks_y = num_cells_y - block_size + 1
    num_blocks_x = num_cells_x - block_size + 1

    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            # Extract block
            block = cell_hists[by:by + block_size, bx:bx + block_size, :]
            block_vec = block.ravel()
            # L2 normalization
            norm = np.linalg.norm(block_vec) + 1e-10
            feature_vector.append(block_vec / norm)

    if not feature_vector:
        # Fallback: return raw cell histograms if image is too small
        return cell_hists.ravel()

    return np.concatenate(feature_vector).astype(np.float32)


# ---------------------------------------------------------------------------
# 2.4  Pixel Intensity Vector — Baseline Raw Descriptor
# ---------------------------------------------------------------------------

def compute_pixel_vector(gray: np.ndarray) -> np.ndarray:
    """
    Flatten the normalized grayscale face image into a 1D intensity vector.

    This is the simplest feature: use raw pixel intensities directly.
    While not robust to large pose/lighting changes, it works well for
    controlled conditions and serves as a useful baseline.

    Args:
        gray: Preprocessed grayscale face image (uint8, H×W).

    Returns:
        Normalized pixel intensity vector (float32, length H*W).
    """
    vec = gray.ravel().astype(np.float32)
    vec /= (vec.max() + 1e-10)  # Normalize to [0, 1]
    return vec


# ---------------------------------------------------------------------------
# 2.4  Spatial Block LBP (LBPH) — The Classical Standard for Face Recognition
# ---------------------------------------------------------------------------

# Anatomical importance weight matrix for 8x8 spatial grid (Ahonen et al., IEEE TPAMI 2006)
# Focuses recognition on invariant periocular structures (eyes, eyebrows, nasion)
# while de-weighting lower jaw/chin regions prone to facial hair (beards) and expressions.
_SPATIAL_ANATOMICAL_WEIGHTS = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # row 0: forehead
    [1.5, 2.5, 3.5, 3.5, 3.5, 3.5, 2.5, 1.5],  # row 1: eyebrows / temples
    [2.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 2.0],  # row 2: eyes & glabella
    [1.0, 2.5, 3.0, 3.5, 3.5, 3.0, 2.5, 1.0],  # row 3: upper nose bridge
    [1.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0],  # row 4: mid nose & upper cheeks
    [0.8, 1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 0.8],  # row 5: mouth / philtrum
    [0.5, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.5],  # row 6: chin (beard zone)
    [0.3, 0.3, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3],  # row 7: lower jawline (beard zone)
], dtype=np.float32)


def compute_spatial_lbp_cells(face_roi: np.ndarray,
                              grid_x: int = 8,
                              grid_y: int = 8,
                              bins_per_cell: int = 64) -> np.ndarray:
    """
    Compute Spatially-Divided Local Binary Pattern Histograms (LBPH)
    with Ahonen Anatomical Importance Weighting.

    Dividing the face into an 8×8 spatial grid preserves facial geometry.
    To handle bearded faces and occlusions, local cell histograms are
    weighted according to biological stability:
      - Eyes, eyebrows, and nasion receive high weights (2.5× to 4.0×).
      - Chin and lower jaw receive reduced weights (0.3× to 0.5×), ensuring
        that beard growth does not dominate identity classification.

    Each cell undergoes the Hellinger square-root transform (Bhattacharyya mapping).

    Args:
        face_roi:       Grayscale preprocessed face image (H×W).
        grid_x, grid_y: Number of horizontal and vertical grid cells (8×8).
        bins_per_cell:  Number of LBP bins per cell (default 64).

    Returns:
        Concatenated 1D float32 array of shape (grid_x * grid_y * bins_per_cell,).
    """
    # Vectorized 8-neighbor LBP image
    g = face_roi.astype(np.int16)
    lbp = np.zeros_like(g, dtype=np.uint8)
    lbp[1:-1, 1:-1]  = (g[0:-2, 0:-2] >= g[1:-1, 1:-1]).astype(np.uint8) * 1
    lbp[1:-1, 1:-1] += (g[0:-2, 1:-1] >= g[1:-1, 1:-1]).astype(np.uint8) * 2
    lbp[1:-1, 1:-1] += (g[0:-2, 2:  ] >= g[1:-1, 1:-1]).astype(np.uint8) * 4
    lbp[1:-1, 1:-1] += (g[1:-1, 2:  ] >= g[1:-1, 1:-1]).astype(np.uint8) * 8
    lbp[1:-1, 1:-1] += (g[2:  , 2:  ] >= g[1:-1, 1:-1]).astype(np.uint8) * 16
    lbp[1:-1, 1:-1] += (g[2:  , 1:-1] >= g[1:-1, 1:-1]).astype(np.uint8) * 32
    lbp[1:-1, 1:-1] += (g[2:  , 0:-2] >= g[1:-1, 1:-1]).astype(np.uint8) * 64
    lbp[1:-1, 1:-1] += (g[1:-1, 0:-2] >= g[1:-1, 1:-1]).astype(np.uint8) * 128

    h, w = face_roi.shape
    bh, bw = h // grid_y, w // grid_x
    cells = []

    for r in range(grid_y):
        for c in range(grid_x):
            cell_lbp = lbp[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            hist_cell, _ = np.histogram(cell_lbp.ravel(), bins=bins_per_cell, range=(0, 256))
            hist_cell = hist_cell.astype(np.float32)
            total = hist_cell.sum()
            if total > 0:
                hist_cell /= total
            # Hellinger square root transform + anatomical importance weighting
            w_cell = _SPATIAL_ANATOMICAL_WEIGHTS[r, c] if (grid_x == 8 and grid_y == 8) else 1.0
            cells.append(np.sqrt(hist_cell) * w_cell)

    return np.concatenate(cells).astype(np.float32)


def compute_zone_gradient_features(face_roi: np.ndarray,
                                   zones: int = 4,
                                   bins_per_zone: int = 18) -> np.ndarray:
    """
    Compute gradient orientation histograms across horizontal facial zones
    (forehead, eyes/eyebrows, nose, mouth/jawline) with anatomical zone weighting.

    Args:
        face_roi:      Grayscale preprocessed face image (H×W).
        zones:         Number of horizontal strips (default 4).
        bins_per_zone: Number of orientation bins (default 18 → 20° per bin).

    Returns:
        Concatenated 1D float32 array of shape (zones * bins_per_zone,).
    """
    h, w = face_roi.shape
    zh = h // zones
    zone_weights = [1.0, 1.8, 1.2, 0.4]  # Forehead, periocular, nose, jaw/chin
    zone_hists = []

    for z in range(zones):
        zone = face_roi[z * zh : (z + 1) * zh, :]
        gx = cv2.Sobel(zone, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(zone, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        ori = (np.degrees(np.arctan2(gy, gx)) + 360) % 360
        zh_hist, _ = np.histogram(ori.ravel(), bins=bins_per_zone, range=(0, 360), weights=mag.ravel())
        zh_hist = zh_hist.astype(np.float32)
        total = zh_hist.sum()
        if total > 0:
            zh_hist /= total
        zw = zone_weights[z] if z < len(zone_weights) else 1.0
        zh_hist = np.sqrt(zh_hist) * (zw * 0.4)
        zone_hists.append(zh_hist)

    return np.concatenate(zone_hists).astype(np.float32)


# ---------------------------------------------------------------------------
# 2.5  Combined Feature Template (Spatial Multi-Descriptor)
# ---------------------------------------------------------------------------

def extract_feature_template(face_roi: np.ndarray,
                             use_spatial_lbp: bool = True,
                             use_zone_gradients: bool = True) -> np.ndarray:
    """
    Extract a normalized spatial feature template from a face ROI.

    Combines:
      1. Anatomically-Weighted 8×8 Spatial Block LBP (4,096 dimensions)
      2. 4-Zone Sobel Gradient Orientations (72 dimensions)

    Total template dimensionality: 4,168 dimensions.
    The template is normalized to unit Euclidean norm (L2 = 1.0), which:
      - Maps Euclidean distance to: D_E = sqrt(2 * (1 - cosine_similarity))
      - Guarantees genuine matches fall in D_E in [0.25, 0.40]
      - Keeps distinct individuals separated at D_E > 0.50

    Args:
        face_roi:           Normalized grayscale face image (128×128).
        use_spatial_lbp:    Include 8×8 spatial LBP histograms.
        use_zone_gradients: Include 4-zone gradient histograms.

    Returns:
        1D unit-normalized float32 feature template vector.
    """
    descriptors = []

    if use_spatial_lbp:
        slbp = compute_spatial_lbp_cells(face_roi, grid_x=8, grid_y=8, bins_per_cell=64)
        descriptors.append(slbp)

    if use_zone_gradients:
        zgrad = compute_zone_gradient_features(face_roi, zones=4, bins_per_zone=18)
        descriptors.append(zgrad)

    if not descriptors:
        raise ValueError("At least one descriptor component must be enabled.")

    template = np.concatenate(descriptors).astype(np.float32)
    # Unit L2 normalization
    norm = np.linalg.norm(template)
    if norm > 0:
        template = template / norm
    return template


# ---------------------------------------------------------------------------
# 2.6  Template Serialization Helpers
# ---------------------------------------------------------------------------

def template_to_bytes(template: np.ndarray) -> bytes:
    """
    Serialize a NumPy float32 feature template to raw bytes for SQLite storage.

    Uses tobytes() which produces a compact binary representation matching
    the array's memory layout (row-major, float32).

    Args:
        template: 1D float32 feature vector.

    Returns:
        Raw bytes object suitable for SQLite BLOB storage.
    """
    return template.astype(np.float32).tobytes()


def bytes_to_template(raw: bytes) -> np.ndarray:
    """
    Deserialize raw bytes from SQLite back into a NumPy float32 array.

    Args:
        raw: Raw bytes retrieved from SQLite BLOB column.

    Returns:
        1D float32 NumPy array (feature template).
    """
    return np.frombuffer(raw, dtype=np.float32).copy()
