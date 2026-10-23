"""
=============================================================================
MAIN APPLICATION — Face Recognition & Identification System
=============================================================================
Command-line interface that ties all 4 modules together:

  1. Preprocessing & Normalization   (preprocessing.py)
  2. Feature Extraction              (feature_extraction.py)
  3. Database & Storage              (database.py)
  4. Matching & Identification       (matching.py)

Usage:
    python main.py                   # Interactive menu
    python main.py --enroll "Alice"  # Enroll a new user via webcam
    python main.py --identify        # Identify face via webcam
    python main.py --list            # List all enrolled users
    python main.py --demo            # Run image-based demo (no webcam needed)

=============================================================================
"""

import argparse
import sys
import os
import time

import cv2
import numpy as np

from preprocessing import (
    full_preprocess_pipeline,
    sobel_edges,
    canny_edges,
    equalize_histogram,
    bgr_to_yuv,
)
from feature_extraction import extract_feature_template
from database import FaceDatabase
from matching import FaceMatcher


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEBCAM_INDEX     = 0          # Default camera device index
NUM_CAPTURE_FRAMES = 5        # Frames to capture per enrollment (averaged)
TARGET_FACE_SIZE = (100, 100) # Face ROI resize target (pixels)
DEFAULT_THRESHOLD = 8.0       # Euclidean distance matching threshold
DB_PATH          = os.path.join(os.path.dirname(__file__),
                                "face_recognition.db")

# Colors for OpenCV overlay (BGR format)
COLOR_GREEN  = (0, 220, 80)
COLOR_RED    = (0, 60, 220)
COLOR_YELLOW = (0, 200, 255)
COLOR_WHITE  = (240, 240, 240)
COLOR_TEAL   = (180, 200, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Utility: Draw HUD overlay on frame
# ---------------------------------------------------------------------------

def draw_hud(frame: np.ndarray, text_lines: list,
             detected: bool = True) -> np.ndarray:
    """
    Draw a semi-transparent heads-up display (HUD) overlay on the frame.

    Args:
        frame:     BGR video frame.
        text_lines: List of (text, color, scale) tuples to render.
        detected:  True if a face was detected in this frame.

    Returns:
        Frame with HUD overlay drawn.
    """
    overlay = frame.copy()
    # Top-left information panel
    cv2.rectangle(overlay, (0, 0), (420, 30 + len(text_lines) * 28), 
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (text, color, scale) in enumerate(text_lines):
        cv2.putText(frame, text, (8, 24 + i * 28),
                    FONT, scale, color, 1, cv2.LINE_AA)
    return frame


def draw_face_box(frame: np.ndarray, x: int, y: int,
                  w: int, h: int, label: str,
                  matched: bool) -> np.ndarray:
    """
    Draw a bounding box around the detected face with identity label.

    Args:
        frame:   BGR video frame.
        x,y,w,h: Face bounding box coordinates.
        label:   Identity text to display.
        matched: True for green box (matched), False for red (unknown).

    Returns:
        Annotated frame.
    """
    color = COLOR_GREEN if matched else COLOR_RED

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner markers (stylized box corners)
    corner_len = 15
    thick = 3
    # Top-left
    cv2.line(frame, (x, y), (x + corner_len, y), color, thick)
    cv2.line(frame, (x, y), (x, y + corner_len), color, thick)
    # Top-right
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thick)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thick)
    # Bottom-left
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thick)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thick)
    # Bottom-right
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thick)

    # Label background
    label_size, _ = cv2.getTextSize(label, FONT, 0.6, 2)
    cv2.rectangle(frame,
                  (x, y - label_size[1] - 10),
                  (x + label_size[0] + 6, y),
                  color, -1)
    cv2.putText(frame, label, (x + 3, y - 5),
                FONT, 0.6, (10, 10, 10), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Haar Cascade helper — reuse from preprocessing
# ---------------------------------------------------------------------------

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces_raw(gray: np.ndarray):
    """Return raw face detection rectangles (x,y,w,h) from grayscale frame."""
    return _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )


# ---------------------------------------------------------------------------
# Mode 1: ENROLL — Register a new user
# ---------------------------------------------------------------------------

def enroll_user_webcam(db: FaceDatabase, name: str) -> bool:
    """
    Enroll a new user by capturing face images from the webcam.

    Captures NUM_CAPTURE_FRAMES preprocessed face ROIs, extracts feature
    templates from each, and stores their AVERAGE template in the database.
    Averaging multiple captures makes the stored template more robust.

    Args:
        db:   FaceDatabase instance.
        name: Name of the user to enroll.

    Returns:
        True on success, False on failure.
    """
    print(f"\n[ENROLL] Starting enrollment for: '{name}'")
    print(f"[ENROLL] Will capture {NUM_CAPTURE_FRAMES} face samples.")
    print("[ENROLL] Press SPACE to capture a sample, Q to quit.\n")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check device index.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    templates = []
    captured = 0

    while captured < NUM_CAPTURE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_raw(gray)

        face_detected = len(faces) > 0
        for (x, y, w, h) in faces:
            draw_face_box(display, x, y, w, h,
                          f"Sample {captured+1}/{NUM_CAPTURE_FRAMES}",
                          matched=True)

        hud = [
            (f"ENROLL MODE: {name}", COLOR_YELLOW, 0.55),
            (f"Captured: {captured}/{NUM_CAPTURE_FRAMES}", COLOR_GREEN, 0.55),
            ("SPACE=Capture  Q=Quit", COLOR_WHITE, 0.45),
            ("Face detected!" if face_detected else "No face found",
             COLOR_GREEN if face_detected else COLOR_RED, 0.45),
        ]
        draw_hud(display, hud, detected=face_detected)

        cv2.imshow("Face Recognition — Enrollment", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("[ENROLL] Cancelled by user.")
            break

        if key == ord(" ") and face_detected:
            # Preprocess and extract template from this frame
            face_roi = full_preprocess_pipeline(frame, TARGET_FACE_SIZE)
            if face_roi is not None:
                template = extract_feature_template(face_roi)
                templates.append(template)
                captured += 1
                print(f"[ENROLL] Captured sample {captured}/{NUM_CAPTURE_FRAMES} "
                      f"(template len={len(template)})")
                # Flash green on capture
                cv2.rectangle(display, (0, 0),
                              (display.shape[1], display.shape[0]),
                              COLOR_GREEN, 8)
                cv2.imshow("Face Recognition — Enrollment", display)
                cv2.waitKey(300)  # Brief visual feedback

    cap.release()
    cv2.destroyAllWindows()

    if len(templates) == 0:
        print("[ENROLL] No valid face samples captured.")
        return False

    # Average all captured templates → more robust representative template
    avg_template = np.mean(np.stack(templates, axis=0), axis=0).astype(np.float32)
    user_id = db.enroll_user(name, avg_template)
    print(f"[ENROLL] SUCCESS: '{name}' enrolled (ID={user_id}) from "
          f"{len(templates)} samples.")
    return True


def enroll_user_from_image(db: FaceDatabase, name: str,
                           image_path: str) -> bool:
    """
    Enroll a user from a saved image file (alternative to webcam).

    Args:
        db:         FaceDatabase instance.
        name:       Name of the user.
        image_path: Path to the face image file (.jpg/.png).

    Returns:
        True on success, False on failure.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return False

    face_roi = full_preprocess_pipeline(image, TARGET_FACE_SIZE)
    if face_roi is None:
        print(f"[ERROR] No face detected in {image_path}")
        return False

    template = extract_feature_template(face_roi)
    user_id = db.enroll_user(name, template)
    print(f"[ENROLL] '{name}' enrolled from image (ID={user_id}).")
    return True


# ---------------------------------------------------------------------------
# Mode 2: IDENTIFY — Real-time face identification from webcam
# ---------------------------------------------------------------------------

def identify_webcam(db: FaceDatabase, matcher: FaceMatcher) -> None:
    """
    Run real-time face identification from the webcam feed.

    For each captured frame:
        1. Detect faces using Haar Cascade
        2. For each face, run preprocessing + feature extraction
        3. Match against the database using Euclidean distance
        4. Overlay identity label and distance on the live video

    Press Q or ESC to exit.

    Args:
        db:      FaceDatabase instance.
        matcher: FaceMatcher instance configured with threshold.
    """
    print("\n[IDENTIFY] Starting real-time identification.")
    print("[IDENTIFY] Press Q or ESC to quit.\n")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_time = time.time()
    fps = 0.0
    frame_count = 0

    # Cache for smoother display (avoid flicker on every frame)
    last_result = None
    result_cache_frames = 5  # Reuse result for N frames between computations

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam.")
            break

        frame_count += 1
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_raw(gray)

        if len(faces) > 0:
            # Process only the largest face
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces_sorted[0]

            # Run identification every N frames to maintain real-time FPS
            if frame_count % result_cache_frames == 0 or last_result is None:
                result = matcher.identify(frame)
                last_result = result

            if last_result is not None:
                label = (f"{last_result.name} "
                         f"d={last_result.distance:.2f}")
                draw_face_box(display, x, y, w, h,
                              label, last_result.matched)

        # FPS calculation
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - fps_time, 1e-5))
        fps_time = current_time

        user_count = len(db.list_users())
        result_text = (last_result.name if last_result else "Scanning...")
        dist_text = (f"{last_result.distance:.3f}" if last_result else "—")

        hud = [
            ("IDENTIFY MODE", COLOR_YELLOW, 0.55),
            (f"FPS: {fps:.1f}  |  Users in DB: {user_count}", COLOR_WHITE, 0.45),
            (f"Identity: {result_text}", COLOR_GREEN, 0.55),
            (f"Distance: {dist_text}  (threshold={matcher.threshold:.1f})",
             COLOR_TEAL, 0.45),
            ("Q / ESC = Quit", COLOR_WHITE, 0.40),
        ]
        draw_hud(display, hud)

        cv2.imshow("Face Recognition — Identification", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[IDENTIFY] Session ended.")


# ---------------------------------------------------------------------------
# Mode 3: LIST — Show all enrolled users
# ---------------------------------------------------------------------------

def list_users(db: FaceDatabase) -> None:
    """
    Print a formatted table of all enrolled users to the console.

    Args:
        db: FaceDatabase instance.
    """
    users = db.list_users()
    if not users:
        print("\n[DB] No users enrolled yet. Use --enroll to add users.\n")
        return

    print(f"\n{'='*65}")
    print(f"{'ID':>4}  {'Name':<22}  {'Enrolled At':<26}  {'Template Len':>12}")
    print(f"{'='*65}")
    for uid, name, enrolled_at, tlen in users:
        print(f"{uid:>4}  {name:<22}  {enrolled_at:<26}  {tlen:>12}")
    print(f"{'='*65}")
    print(f"Total enrolled: {len(users)} user(s)\n")


# ---------------------------------------------------------------------------
# Mode 4: DEMO — Image-based demo without webcam
# ---------------------------------------------------------------------------

def run_demo(db: FaceDatabase, matcher: FaceMatcher) -> None:
    """
    Run a non-webcam demonstration of the preprocessing pipeline.

    Generates a synthetic test face image, applies all preprocessing steps,
    extracts features, and shows intermediate results in separate windows.
    Useful for environments without a webcam (e.g., remote servers).

    Args:
        db:      FaceDatabase instance.
        matcher: FaceMatcher instance.
    """
    print("\n[DEMO] Generating synthetic face image demo...")

    # -------------------------------------------------------------------------
    # Create a synthetic face-like image for demonstration
    # -------------------------------------------------------------------------
    h, w = 200, 200
    # White background
    demo_img = np.ones((h, w, 3), dtype=np.uint8) * 220

    # Face oval
    cv2.ellipse(demo_img, (w//2, h//2), (70, 85), 0, 0, 360,
                (180, 160, 140), -1)
    # Left eye
    cv2.ellipse(demo_img, (w//2 - 22, h//2 - 20), (14, 9), 0, 0, 360,
                (60, 50, 40), -1)
    # Right eye
    cv2.ellipse(demo_img, (w//2 + 22, h//2 - 20), (14, 9), 0, 0, 360,
                (60, 50, 40), -1)
    # Nose
    pts = np.array([[w//2, h//2 + 5],
                    [w//2 - 8, h//2 + 25],
                    [w//2 + 8, h//2 + 25]], np.int32)
    cv2.fillPoly(demo_img, [pts], (150, 130, 110))
    # Mouth
    cv2.ellipse(demo_img, (w//2, h//2 + 40), (22, 10), 0, 0, 180,
                (100, 80, 60), -1)

    gray = cv2.cvtColor(demo_img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------------------
    # Apply preprocessing stages and visualize
    # -------------------------------------------------------------------------
    from preprocessing import (equalize_histogram, clahe_equalization,
                                gaussian_blur, median_filter,
                                sobel_edges, canny_edges)

    equalized   = equalize_histogram(gray)
    clahe_img   = clahe_equalization(gray)
    blurred     = gaussian_blur(gray, kernel_size=5, sigma=1.5)
    median_f    = median_filter(gray, kernel_size=5)
    sobel_img   = sobel_edges(gray)
    canny_img   = canny_edges(gray, 50, 150)

    # Stack for display: 3 rows × 2 cols
    row1 = np.hstack([
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    ])
    row2 = np.hstack([
        cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    ])
    row3 = np.hstack([
        cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    ])

    # Add labels
    labels_row1 = ["Grayscale", "Hist. Equalized"]
    labels_row2 = ["CLAHE", "Gaussian Blur"]
    labels_row3 = ["Sobel Edges", "Canny Edges"]

    for i, label in enumerate(labels_row1):
        cv2.putText(row1, label, (i * 200 + 5, 18),
                    FONT, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)
    for i, label in enumerate(labels_row2):
        cv2.putText(row2, label, (i * 200 + 5, 18),
                    FONT, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)
    for i, label in enumerate(labels_row3):
        cv2.putText(row3, label, (i * 200 + 5, 18),
                    FONT, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)

    full_display = np.vstack([row1, row2, row3])

    # Title bar
    title_bar = np.zeros((40, full_display.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (30, 30, 30)
    cv2.putText(title_bar,
                "Face Recognition — Preprocessing Pipeline Demo",
                (10, 26), FONT, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)
    full_display = np.vstack([title_bar, full_display])

    print("[DEMO] Displaying preprocessing pipeline — press any key to close.")
    cv2.imshow("Preprocessing Demo", full_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # -------------------------------------------------------------------------
    # Feature extraction demo
    # -------------------------------------------------------------------------
    face_roi = cv2.resize(gray, TARGET_FACE_SIZE)
    template = extract_feature_template(face_roi)
    print(f"\n[DEMO] Feature template extracted:")
    print(f"       Template length : {len(template)} dimensions")
    print(f"       Template min    : {template.min():.6f}")
    print(f"       Template max    : {template.max():.6f}")
    print(f"       Template mean   : {template.mean():.6f}")
    print(f"       Template L2 norm: {np.linalg.norm(template):.4f}")
    print("\n[DEMO] Demo complete.")


# ---------------------------------------------------------------------------
# Mode 5: DELETE — Remove an enrolled user
# ---------------------------------------------------------------------------

def delete_user(db: FaceDatabase, user_id: int) -> None:
    """
    Delete an enrolled user from the database by ID.

    Args:
        db:      FaceDatabase instance.
        user_id: The user's database ID to delete.
    """
    success = db.delete_user(user_id)
    if success:
        print(f"[DELETE] User ID={user_id} successfully removed.")
    else:
        print(f"[DELETE] User ID={user_id} not found in database.")


# ---------------------------------------------------------------------------
# Interactive Menu
# ---------------------------------------------------------------------------

def interactive_menu(db: FaceDatabase, matcher: FaceMatcher) -> None:
    """
    Display an interactive command-line menu for all operations.

    Args:
        db:      FaceDatabase instance.
        matcher: FaceMatcher instance.
    """
    while True:
        print("\n" + "=" * 55)
        print("   FACE RECOGNITION & IDENTIFICATION SYSTEM")
        print("=" * 55)
        print("  1. Enroll new user (webcam)")
        print("  2. Enroll new user (from image file)")
        print("  3. Identify face (real-time webcam)")
        print("  4. List all enrolled users")
        print("  5. Delete a user")
        print("  6. Run preprocessing demo (no webcam)")
        print("  7. Change matching threshold")
        print("  8. Exit")
        print("=" * 55)

        try:
            choice = input("  Select option [1-8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] Goodbye!")
            sys.exit(0)

        if choice == "1":
            name = input("  Enter user name: ").strip()
            if name:
                enroll_user_webcam(db, name)

        elif choice == "2":
            name = input("  Enter user name: ").strip()
            path = input("  Enter image file path: ").strip()
            if name and path:
                enroll_user_from_image(db, name, path)

        elif choice == "3":
            identify_webcam(db, matcher)

        elif choice == "4":
            list_users(db)

        elif choice == "5":
            list_users(db)
            try:
                uid = int(input("  Enter user ID to delete: ").strip())
                confirm = input(f"  Delete user ID={uid}? [y/N]: ").strip().lower()
                if confirm == "y":
                    delete_user(db, uid)
            except ValueError:
                print("  Invalid ID.")

        elif choice == "6":
            run_demo(db, matcher)

        elif choice == "7":
            try:
                t = float(input(f"  Current threshold={matcher.threshold:.2f}. "
                                "  Enter new threshold: ").strip())
                matcher.set_threshold(t)
            except ValueError:
                print("  Invalid threshold value.")

        elif choice == "8":
            print("[EXIT] Goodbye!")
            sys.exit(0)

        else:
            print("  Invalid choice. Please enter 1–8.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face Recognition & Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Launch interactive menu
  python main.py --enroll "Alice"       # Enroll Alice via webcam
  python main.py --enroll-img "Bob" img.jpg  # Enroll Bob from image
  python main.py --identify             # Real-time identification
  python main.py --list                 # List enrolled users
  python main.py --delete 3             # Delete user ID=3
  python main.py --demo                 # Preprocessing pipeline demo
  python main.py --threshold 6.0        # Set custom threshold
        """
    )
    parser.add_argument("--enroll",     metavar="NAME",
                        help="Enroll a new user via webcam")
    parser.add_argument("--enroll-img", nargs=2, metavar=("NAME", "IMAGE"),
                        help="Enroll a user from an image file")
    parser.add_argument("--identify",   action="store_true",
                        help="Run real-time face identification")
    parser.add_argument("--list",       action="store_true",
                        help="List all enrolled users")
    parser.add_argument("--delete",     type=int, metavar="USER_ID",
                        help="Delete a user by ID")
    parser.add_argument("--demo",       action="store_true",
                        help="Run preprocessing pipeline demo")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        metavar="DIST",
                        help=f"Matching threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--db",         default=DB_PATH, metavar="PATH",
                        help="Path to SQLite database file")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Initialize database and matcher
    db = FaceDatabase(db_path=args.db)
    matcher = FaceMatcher(db=db, threshold=args.threshold)

    # Route to requested mode
    if args.enroll:
        enroll_user_webcam(db, args.enroll)

    elif args.enroll_img:
        name, img_path = args.enroll_img
        enroll_user_from_image(db, name, img_path)

    elif args.identify:
        identify_webcam(db, matcher)

    elif args.list:
        list_users(db)

    elif args.delete is not None:
        delete_user(db, args.delete)

    elif args.demo:
        run_demo(db, matcher)

    else:
        # No flags provided → launch interactive menu
        interactive_menu(db, matcher)


if __name__ == "__main__":
    main()
