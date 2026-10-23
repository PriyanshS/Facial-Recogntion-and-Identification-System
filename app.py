"""
=============================================================================
FastAPI Backend Server for Face Recognition & Identification System
=============================================================================
Exposes REST APIs for:
  - Face identification with full intermediate CV pipeline extraction
  - User enrollment (multipart upload or base64)
  - Pipeline inspection (Sobel, Canny, CLAHE, LBP, HOG)
  - User management (list, delete, reset)
  - Preloaded test dataset seeding
=============================================================================
"""

import os
import io
import base64
import time
import sqlite3
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from preprocessing import (
    full_preprocess_pipeline,
    rgb_to_grayscale,
    bgr_to_yuv,
    clahe_equalization,
    equalize_histogram,
    gaussian_blur,
    median_filter,
    sobel_edges,
    canny_edges,
    homomorphic_filter,
    detect_and_crop_face,
    _FACE_CASCADE,
)
from feature_extraction import (
    extract_feature_template,
    compute_lbp_fast,
    compute_gradient_histogram,
    compute_hog_features,
)
from database import FaceDatabase
from matching import FaceMatcher, euclidean_distance, cosine_similarity

# Initialize app
app = FastAPI(title="Classical Face Recognition & Identification System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "face_recognition.db")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

db = FaceDatabase(db_path=DB_FILE)
matcher = FaceMatcher(db=db, threshold=0.46)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def np_to_base64_png(img_np: np.ndarray) -> str:
    """Encode OpenCV NumPy image (grayscale or BGR) to base64 PNG data URL."""
    if img_np is None or img_np.size == 0:
        return ""
    success, buffer = cv2.imencode(".png", img_np)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG, PNG, etc.) to BGR NumPy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file or format.")
    return img


def decode_data_url(data_url: str) -> np.ndarray:
    """Decode a base64 data URL to BGR NumPy array."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    return decode_image_bytes(image_bytes)


def extract_pipeline_stages(bgr_img: np.ndarray):
    """
    Run classical image processing pipeline stages and return base64 images
    along with face bounding box detection info.
    """
    gray = rgb_to_grayscale(bgr_img)
    yuv = bgr_to_yuv(bgr_img)
    y_channel = yuv[:, :, 0]

    # Face detection
    faces = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50)
    )
    if len(faces) == 0:
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(35, 35)
        )

    annotated = bgr_img.copy()
    face_roi = None
    bbox_info = None

    if len(faces) > 0:
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces_sorted[0]
        bbox_info = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        # Draw bounding box and corner brackets
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 230, 115), 2)
        cv2.putText(
            annotated,
            f"Face ROI ({w}x{h})",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 230, 115),
            2,
            cv2.LINE_AA,
        )
        # Canonical inner facial crop (focus on eyes, nose, mouth; remove hair/collar)
        cy = int(max(0, y + h * 0.10))
        ch = int(min(gray.shape[0] - cy, h * 0.82))
        cx = int(max(0, x + w * 0.08))
        cw = int(min(gray.shape[1] - cx, w * 0.84))
        face_roi = gray[cy : cy + ch, cx : cx + cw]
    else:
        # Fallback to center crop if Haar cascade detects nothing (e.g. drawn face)
        h_img, w_img = gray.shape
        my, mx = int(h_img * 0.15), int(w_img * 0.15)
        face_roi = gray[my : h_img - my, mx : w_img - mx]
        bbox_info = {"x": mx, "y": my, "w": w_img - 2 * mx, "h": h_img - 2 * my}

    # Canonical preprocessed face ROI (with multi-angle fallback + eye de-rotation)
    canonical_face = detect_and_crop_face(bgr_img, target_size=(128, 128))
    if canonical_face is not None:
        blur_roi = canonical_face
        norm_roi = canonical_face
    else:
        norm_roi = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_AREA)
        clahe_roi = clahe_equalization(norm_roi, clip_limit=2.2)
        blur_roi = gaussian_blur(clahe_roi, kernel_size=3, sigma=0.7)

    # Visual intermediate pipeline stages for studio inspection
    homo_roi = homomorphic_filter(norm_roi, d0=20.0, gamma_low=0.75, gamma_high=1.25)
    eq_roi = equalize_histogram(norm_roi)
    clahe_roi = clahe_equalization(norm_roi, clip_limit=2.0)
    median_roi = median_filter(clahe_roi, kernel_size=3)
    sobel_roi = sobel_edges(blur_roi)
    canny_roi = canny_edges(blur_roi, 50, 150)

    # Feature descriptors
    lbp_hist = compute_lbp_fast(blur_roi, num_bins=256)
    grad_hist = compute_gradient_histogram(blur_roi, num_bins=36)
    template = extract_feature_template(blur_roi)

    stages = {
        "annotated_input": np_to_base64_png(annotated),
        "grayscale": np_to_base64_png(gray),
        "yuv_y": np_to_base64_png(y_channel),
        "raw_roi": np_to_base64_png(face_roi),
        "homomorphic": np_to_base64_png(homo_roi),
        "hist_equalized": np_to_base64_png(eq_roi),
        "clahe": np_to_base64_png(clahe_roi),
        "gaussian_blur": np_to_base64_png(blur_roi),
        "median_filter": np_to_base64_png(median_roi),
        "sobel_edges": np_to_base64_png(sobel_roi),
        "canny_edges": np_to_base64_png(canny_roi),
        "normalized_roi": np_to_base64_png(norm_roi),
        "bbox": bbox_info,
        "template": template,
        "lbp_hist": lbp_hist.tolist(),
        "grad_hist": grad_hist.tolist(),
        "hog_sample": template[:64].tolist(),
        "template_len": len(template),
    }
    return stages


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "enrolled_users": len(db.list_users())}


@app.get("/api/users")
def get_users():
    """Retrieve all enrolled users without full feature blobs."""
    users = db.list_users()
    return [
        {
            "id": u[0],
            "name": u[1],
            "enrolled_at": u[2],
            "template_len": u[3],
        }
        for u in users
    ]


@app.delete("/api/users/{user_id}")
def delete_user_endpoint(user_id: int):
    """Delete an enrolled user."""
    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"success": True, "message": f"User {user_id} deleted."}


@app.post("/api/users/clear")
def clear_all_users_endpoint():
    """Clear all enrolled users."""
    count = db.clear_all_users()
    return {"success": True, "cleared_count": count}


def build_gallery_template(face_roi: np.ndarray) -> np.ndarray:
    """
    Build multi-representation gallery templates for an enrolled face:
      1. Base canonical face
      2. Zoomed/scaled face (92% inner crop)
      3. Horizontally mirrored face (symmetry)
    Concatenated into a single compound BLOB for nearest-neighbor gallery matching.
    """
    t1 = extract_feature_template(face_roi)

    # Slight zoom/scale variation
    h, w = face_roi.shape
    dh, dw = int(h * 0.04), int(w * 0.04)
    zoomed = cv2.resize(face_roi[dh : h - dh, dw : w - dw], (w, h), interpolation=cv2.INTER_AREA)
    t2 = extract_feature_template(zoomed)

    # Horizontal flip (bilateral facial symmetry)
    flipped = cv2.flip(face_roi, 1)
    t3 = extract_feature_template(flipped)

    return np.concatenate([t1, t2, t3]).astype(np.float32)


@app.post("/api/enroll")
async def enroll_face(
    name: str = Form(...),
    file: UploadFile = File(None),
    data_url: str = Form(None),
):
    """Enroll a user using uploaded image file or base64 data URL."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    if file is not None:
        contents = await file.read()
        bgr = decode_image_bytes(contents)
    elif data_url:
        bgr = decode_data_url(data_url)
    else:
        raise HTTPException(status_code=400, detail="No image provided.")

    face_roi = full_preprocess_pipeline(bgr, (128, 128))
    if face_roi is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the provided image. Please use a clear, frontal headshot.",
        )

    gallery_template = build_gallery_template(face_roi)
    user_id = db.enroll_user(name, gallery_template)

    # Save thumbnail for UI
    thumb_path = os.path.join(STATIC_DIR, f"thumb_user_{user_id}.png")
    cv2.imwrite(thumb_path, face_roi)

    return {
        "success": True,
        "user_id": user_id,
        "name": name,
        "template_len": len(gallery_template),
        "thumbnail_url": f"/static/thumb_user_{user_id}.png",
    }


@app.post("/api/identify")
async def identify_face_endpoint(
    file: UploadFile = File(None),
    data_url: str = Form(None),
    threshold: float = Form(0.46),
    sample_key: str = Form(None),
):
    """
    Run full identification and classical CV pipeline on a query image.
    Returns matched identity, distance metrics, candidate rankings, and
    all visual pipeline stages.
    """
    start_time = time.time()

    # Load image from file, base64, or sample file
    if sample_key:
        sample_path = os.path.join(SAMPLE_DIR, f"{sample_key}.jpg")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(SAMPLE_DIR, sample_key)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample image not found.")
        bgr = cv2.imread(sample_path)
    elif file is not None:
        contents = await file.read()
        bgr = decode_image_bytes(contents)
    elif data_url:
        bgr = decode_data_url(data_url)
    else:
        raise HTTPException(status_code=400, detail="No image provided.")

    if bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    # Run classical CV pipeline stages
    pipeline = extract_pipeline_stages(bgr)
    query_template = pipeline["template"]

    # Match against database
    all_users = db.get_all_users()
    candidates = []

    matched = False
    best_match = None
    min_distance = float("inf")
    best_sim = 0.0

    if all_users:
        for uid, name, stored_template in all_users:
            try:
                dist = euclidean_distance(query_template, stored_template)
                sim = cosine_similarity(query_template, stored_template)
                candidates.append(
                    {
                        "user_id": uid,
                        "name": name,
                        "distance": round(dist, 4),
                        "similarity": round(sim, 4),
                        "is_within_threshold": bool(dist < threshold),
                    }
                )
            except Exception as e:
                continue

        candidates.sort(key=lambda c: c["distance"])

        if candidates:
            best = candidates[0]
            min_distance = best["distance"]
            best_sim = best["similarity"]
            matched = bool(min_distance < threshold)
            best_match = {
                "user_id": best["user_id"] if matched else None,
                "name": best["name"] if matched else "Unknown",
                "closest_name": best["name"],
                "distance": min_distance,
                "similarity": best_sim,
                "matched": matched,
            }

    if not best_match:
        best_match = {
            "user_id": None,
            "name": "Unknown (Empty DB)" if not all_users else "Unknown",
            "closest_name": "None",
            "distance": float("inf"),
            "similarity": 0.0,
            "matched": False,
        }

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "success": True,
        "elapsed_ms": elapsed_ms,
        "threshold": threshold,
        "match": best_match,
        "candidates": candidates,
        "total_enrolled": len(all_users),
        "pipeline": {
            "annotated_input": pipeline["annotated_input"],
            "grayscale": pipeline["grayscale"],
            "yuv_y": pipeline["yuv_y"],
            "raw_roi": pipeline["raw_roi"],
            "homomorphic": pipeline.get("homomorphic", ""),
            "hist_equalized": pipeline["hist_equalized"],
            "clahe": pipeline["clahe"],
            "gaussian_blur": pipeline["gaussian_blur"],
            "median_filter": pipeline["median_filter"],
            "sobel_edges": pipeline["sobel_edges"],
            "canny_edges": pipeline["canny_edges"],
            "normalized_roi": pipeline["normalized_roi"],
            "bbox": pipeline["bbox"],
            "template_len": pipeline["template_len"],
            "lbp_hist": pipeline["lbp_hist"],
            "grad_hist": pipeline["grad_hist"],
            "hog_sample": pipeline["hog_sample"],
        },
    }


@app.post("/api/seed_demo")
def seed_demo_database():
    """
    Automatically enroll Alice and Bob using sample portraits.
    Clears existing DB and sets up a clean, reliable demo dataset.
    """
    db.clear_all_users()

    enrolled = []
    # Enroll Alice
    alice_path = os.path.join(SAMPLE_DIR, "alice_enroll.jpg")
    if os.path.exists(alice_path):
        img_alice = cv2.imread(alice_path)
        roi_alice = full_preprocess_pipeline(img_alice, (128, 128))
        if roi_alice is not None:
            t_alice = build_gallery_template(roi_alice)
            uid = db.enroll_user("Alice", t_alice)
            cv2.imwrite(os.path.join(STATIC_DIR, f"thumb_user_{uid}.png"), roi_alice)
            enrolled.append({"id": uid, "name": "Alice"})

    # Enroll Bob
    bob_path = os.path.join(SAMPLE_DIR, "bob_enroll.jpg")
    if os.path.exists(bob_path):
        img_bob = cv2.imread(bob_path)
        roi_bob = full_preprocess_pipeline(img_bob, (128, 128))
        if roi_bob is not None:
            t_bob = build_gallery_template(roi_bob)
            uid = db.enroll_user("Bob", t_bob)
            cv2.imwrite(os.path.join(STATIC_DIR, f"thumb_user_{uid}.png"), roi_bob)
            enrolled.append({"id": uid, "name": "Bob"})

    return {
        "success": True,
        "message": "Demo database successfully initialized with Alice and Bob.",
        "enrolled": enrolled,
    }


@app.get("/api/samples")
def get_sample_images():
    """List built-in sample test images."""
    samples = [
        {
            "key": "bob_tilted_15deg",
            "name": "Bob (15° Tilted Head)",
            "role": "Head roll & rotation de-tilt test",
            "expected": "Match: Bob (d ~ 0.32)",
            "image_url": "/sample_data/bob_tilted_15deg.jpg",
        },
        {
            "key": "bob_bearded",
            "name": "Bob (Full Heavy Beard)",
            "role": "Facial hair occlusion test",
            "expected": "Match: Bob (d ~ 0.38)",
            "image_url": "/sample_data/bob_bearded.jpg",
        },
        {
            "key": "bob_query",
            "name": "Bob (Diff Lighting/Expression)",
            "role": "Genuine same person, different shot",
            "expected": "Match: Bob (d ~ 0.31)",
            "image_url": "/sample_data/bob_query.jpg",
        },
        {
            "key": "alice_query_genuine",
            "name": "Alice (Diff Photo - Tilt/Light)",
            "role": "Genuine same person, different shot",
            "expected": "Match: Alice (d ~ 0.33)",
            "image_url": "/sample_data/alice_query_genuine.jpg",
        },
        {
            "key": "charlie_unknown",
            "name": "Charlie (Different Person)",
            "role": "Unregistered person (Rejection Test)",
            "expected": "Flag: Unknown (d > 0.52)",
            "image_url": "/sample_data/charlie_unknown.jpg",
        },
        {
            "key": "alice_enroll",
            "name": "Alice (Master Photo)",
            "role": "Enrolled template source",
            "expected": "Match: Alice (d=0.0)",
            "image_url": "/sample_data/alice_enroll.jpg",
        },
        {
            "key": "bob_enroll",
            "name": "Bob (Master Photo)",
            "role": "Enrolled template source",
            "expected": "Match: Bob (d=0.0)",
            "image_url": "/sample_data/bob_enroll.jpg",
        },
    ]
    return samples


# Mount static assets and sample data
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/sample_data", StaticFiles(directory=SAMPLE_DIR), name="sample_data")


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Frontend index.html under construction"}
