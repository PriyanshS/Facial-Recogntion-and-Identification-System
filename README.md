# Face Recognition & Identification System

A modular, end-to-end **classical computer vision** face recognition project built with **Python** (OpenCV, NumPy, SQLite) and **MATLAB** (MATLAB Online compatible). Implements the complete image processing pipeline from an undergraduate Image & Video Processing course.

---

## Project Structure

```
Facial_Recognition/
├── app.py                  # FastAPI Backend API Server (serves frontend & APIs)
├── static/                 # Web Studio Frontend
│   ├── index.html          # HTML5 User Interface
│   ├── style.css           # Vanilla CSS Dark Glassmorphism Design
│   └── app.js              # Client-side Pipeline & Canvas Visualizer
├── sample_data/            # High-resolution test portraits (Alice, Bob, Charlie)
├── preprocessing.py        # Module 1: Preprocessing & Normalization
├── feature_extraction.py   # Module 2: Feature Extraction & Template Generation
├── database.py             # Module 3: SQLite Database & Storage
├── matching.py             # Module 4: Matching & Identification
├── main.py                 # CLI application
├── requirements.txt        # Python dependencies
├── face_recognition.db     # SQLite database
└── matlab/
    └── FaceRecognitionSystem.m   # Complete MATLAB implementation
```

---

## Module Overview

### Module 1 — `preprocessing.py`

| Technique | Function | Purpose |
|-----------|----------|---------|
| RGB → Grayscale | `rgb_to_grayscale()` | Luminance extraction (Y = 0.299R + 0.587G + 0.114B) |
| RGB → YUV | `bgr_to_yuv()` | Separate luminance from chrominance |
| RGB → HSI | `bgr_to_hsi()` | Hue/Saturation/Intensity for illumination separation |
| Global Hist. Eq. | `equalize_histogram()` | Redistribute pixel intensities over [0,255] |
| CLAHE | `clahe_equalization()` | Locally adaptive contrast normalization |
| Gaussian Blur | `gaussian_blur()` | Low-pass spatial filter for noise reduction |
| Median Filter | `median_filter()` | Non-linear filter for salt-and-pepper noise |
| Sobel Edges | `sobel_edges()` | Gradient magnitude via Gx/Gy kernels |
| Canny Edges | `canny_edges()` | Multi-stage optimal edge detection |
| Face Detection | `detect_and_crop_face()` | Haar Cascade ROI localization & cropping |

### Module 2 — `feature_extraction.py`

| Descriptor | Function | Dimensionality |
|------------|----------|----------------|
| LBP Histogram | `compute_lbp_fast()` | 256-dim |
| Gradient Orientation Hist. | `compute_gradient_histogram()` | 36-dim |
| HOG | `compute_hog_features()` | Variable (~360–900 dim) |
| Raw Pixel Vector | `compute_pixel_vector()` | H×W = 10,000-dim |
| **Combined Template** | `extract_feature_template()` | **~650–1200-dim** |

### Module 3 — `database.py`

- SQLite database with `users` and `enrollment_log` tables
- BLOB storage for serialized NumPy float32 feature vectors
- Methods: `enroll_user()`, `get_all_users()`, `update_user_template()`, `delete_user()`, `list_users()`

### Module 4 — `matching.py`

| Distance Metric | Function | Usage |
|-----------------|----------|-------|
| Euclidean (L2) | `euclidean_distance()` | **Primary matching metric** |
| Cosine Similarity | `cosine_similarity()` | Scale-invariant similarity |
| Chi-Squared | `chi_squared_distance()` | Histogram comparison |
| Manhattan (L1) | `manhattan_distance()` | Outlier-robust distance |
| NCC | `normalized_correlation()` | Illumination-invariant correlation |

---

## Web Studio (Frontend)

The project includes an interactive web dashboard:

```bash
# Launch the web application server
uvicorn app:app --host 127.0.0.1 --port 8050
```

Then open your browser to **`http://localhost:8050`**.

### Frontend Features:
- **Identification Studio**: Drag-and-drop custom faces, run 1-click test scenarios, or capture snapshots via live browser webcam.
- **Dynamic Threshold Slider**: Move the slider between 3.0 and 15.0 to see real-time classification changes between **MATCHED** and **UNKNOWN**.
- **DIP Pipeline Inspector (9 Stages)**: Visual side-by-side comparison of Grayscale, YUV Y-channel, Histogram Equalization, CLAHE, Gaussian blur, Median filter, Sobel edge magnitude, Canny contours, and Cropped ROI.
- **Feature Extraction Visualizer**: Micro-texture distributions via LBP 256-bin histogram, Sobel gradient orientation chart, and HOG sparkline bars.
- **Enrollment & Database**: Register new users directly with webcam/photo uploads, view template dimensions, and manage the SQLite database.

---

## Python Setup & CLI Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> SQLite3 is part of Python's standard library — no separate install needed.

### 2. Launch interactive menu

```bash
python main.py
```

### 3. Command-line usage

```bash
# Enroll a user via webcam
python main.py --enroll "Alice"

# Enroll a user from an image file
python main.py --enroll-img "Bob" /path/to/bob.jpg

# Real-time identification from webcam
python main.py --identify

# List all enrolled users in the database
python main.py --list

# Delete user with ID=2
python main.py --delete 2

# Run preprocessing demo (no webcam required)
python main.py --demo

# Set a custom matching threshold (default=8.0)
python main.py --identify --threshold 6.5

# Use a custom database path
python main.py --db /path/to/custom.db
```

### 4. Enrollment workflow

```
Launch App → Option 1 (Enroll) → Position face in webcam frame
→ Press SPACE to capture 5 samples → Template averaged & stored in DB
```

### 5. Identification workflow

```
Launch App → Option 3 (Identify) → System processes each frame:
Raw Frame → Detect Face → CLAHE + Blur → LBP+HOG+GradHist Features
         → Euclidean Distance to All DB Templates → Identity (or Unknown)
```

---

## MATLAB Usage (MATLAB Online)

### Steps to run in [MATLAB Online](https://matlab.mathworks.com):

1. **Upload** `matlab/FaceRecognitionSystem.m` to your MATLAB Online workspace
2. **Open** the file in the MATLAB editor
3. **Run** it (`F5` or click the **Run** button)
4. An interactive menu appears in the **Command Window**

### MATLAB Features:

| Menu Option | Description |
|-------------|-------------|
| 1. Enroll from image | Load `.jpg`/`.png` or use built-in synthetic face |
| 2. Identify from image | Run full pipeline + show ranked results |
| 3. Preprocessing demo | Visualize all 9 stages in figure windows |
| 4. List users | Show enrolled user table |
| 5. Delete user | Remove user by ID |

### MATLAB Notes:

- **No webcam support** in MATLAB Online — uses image files or synthetic faces
- Database stored as `face_db.mat` (auto-saved to workspace)
- Requires **Image Processing Toolbox** for `imresize()`
- **Computer Vision Toolbox** is optional (used for `vision.CascadeObjectDetector`); falls back to center-crop if unavailable
- All algorithms manually implemented (Sobel, CLAHE, LBP, HOG, Canny, Euclidean) — no black-box wrappers

---

## Classical Techniques Reference (Course Alignment)

| Course Topic | Technique Used |
|---|---|
| Color Spaces | RGB→Gray, RGB→YUV, RGB→HSI (manual formulas) |
| Spatial Filtering | Gaussian blur (separable kernel), Median filter (non-linear) |
| Histogram Processing | Global HE, CLAHE (local adaptive) |
| Edge Detection | Sobel (gradient magnitude), Canny (NMS + hysteresis) |
| Feature Extraction | LBP (texture), HOG (structure), Gradient Hist (shape) |
| Pattern Matching | Euclidean distance, Cosine similarity, Chi-squared, NCC |
| Thresholding | Distance threshold for accept/reject classification |

---

## Tuning the Matching Threshold

The default threshold is **8.0** (Euclidean distance). Adjust based on your results:

| Threshold | Effect |
|-----------|--------|
| Lower (e.g. 5.0) | Stricter — fewer false positives, more "Unknown" results |
| Higher (e.g. 12.0) | More lenient — more matches, higher false positive risk |

```bash
python main.py --identify --threshold 6.0
```

---

## Database Schema (SQLite)

```sql
CREATE TABLE users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    enrolled_at  TEXT    NOT NULL,     -- ISO-8601 timestamp
    template     BLOB    NOT NULL,     -- float32 bytes (NumPy serialized)
    template_len INTEGER NOT NULL      -- length of feature vector
);

CREATE TABLE enrollment_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL,
    action    TEXT    NOT NULL,        -- ENROLLED | TEMPLATE_UPDATED
    timestamp TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```
