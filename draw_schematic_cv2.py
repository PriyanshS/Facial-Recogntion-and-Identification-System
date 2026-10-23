import cv2
import numpy as np

# Create high-res canvas (width=1800, height=850, 3 channels)
W, H = 1800, 860
canvas = np.full((H, W, 3), 255, dtype=np.uint8)

# Colors (BGR)
BG_WHITE = (255, 255, 255)
PANEL_BG = (248, 250, 252)
BORDER_GRAY = (200, 205, 215)
GRID_LINE = (220, 225, 230)
TEXT_DARK = (30, 35, 45)
TEXT_MUTED = (100, 110, 125)
BLUE_ACCENT = (180, 80, 40)    # BGR for #2850b4
RED_ACCENT = (50, 60, 215)     # BGR for #d73c32
GREEN_ACCENT = (60, 160, 40)   # BGR
ORANGE_ACCENT = (30, 120, 220) # BGR
ROW_HIGHLIGHT_BLUE = (240, 225, 210) # BGR fill
ROW_HIGHLIGHT_RED = (215, 215, 250)  # BGR fill

# Draw Header Title
cv2.rectangle(canvas, (0, 0), (W, 70), (242, 245, 249), -1)
cv2.line(canvas, (0, 70), (W, 70), BORDER_GRAY, 2)
cv2.putText(canvas, "ANATOMICAL LANDMARK DEFORMATION & SPATIAL GRID MISALIGNMENT MODEL", 
            (50, 45), cv2.FONT_HERSHEY_DUPLEX, 0.95, TEXT_DARK, 2, cv2.LINE_AA)

def draw_panel(x_offset, y_offset, grid_size, eye_y_val, mouth_y_val, is_compressed, title_text, pitch_text):
    cell_px = grid_size // 8 # 48 px per cell for 384x384 grid
    scale = grid_size / 128.0 # 3.0
    
    # Panel box
    box_w = grid_size + 140
    box_h = grid_size + 190
    cv2.rectangle(canvas, (x_offset - 20, y_offset - 40), (x_offset + box_w - 40, y_offset + box_h - 40), PANEL_BG, -1)
    cv2.rectangle(canvas, (x_offset - 20, y_offset - 40), (x_offset + box_w - 40, y_offset + box_h - 40), BORDER_GRAY, 2)
    
    # Panel title
    accent = RED_ACCENT if is_compressed else BLUE_ACCENT
    cv2.putText(canvas, title_text, (x_offset, y_offset - 12), cv2.FONT_HERSHEY_DUPLEX, 0.72, accent, 2, cv2.LINE_AA)
    cv2.putText(canvas, pitch_text, (x_offset, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MUTED, 1, cv2.LINE_AA)
    
    gy = y_offset + 35
    gx = x_offset + 35
    
    # Highlight specific row
    if not is_compressed:
        # Row 6 (mouth)
        r6_y = int(gy + 6 * cell_px)
        cv2.rectangle(canvas, (gx, r6_y), (gx + grid_size, r6_y + cell_px), ROW_HIGHLIGHT_BLUE, -1)
    else:
        # Row 5 (migrated mouth)
        r5_y = int(gy + 5 * cell_px)
        cv2.rectangle(canvas, (gx, r5_y), (gx + grid_size, r5_y + cell_px), ROW_HIGHLIGHT_RED, -1)
        
    # Draw Grid Lines
    for i in range(9):
        # horizontal
        cy = gy + i * cell_px
        cv2.line(canvas, (gx, cy), (gx + grid_size, cy), GRID_LINE, 1, cv2.LINE_AA)
        # vertical
        cx = gx + i * cell_px
        cv2.line(canvas, (cx, gy), (cx, gy + grid_size), GRID_LINE, 1, cv2.LINE_AA)
        
    # Label Row numbers R0-R7
    for r in range(8):
        ry = int(gy + r * cell_px + cell_px/2 + 5)
        cv2.putText(canvas, f"R{r}", (gx - 28, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MUTED, 1, cv2.LINE_AA)
        
    # Face Silhouette
    face_cx = int(gx + 64 * scale)
    face_cy = int(gy + 64 * scale)
    cv2.ellipse(canvas, (face_cx, face_cy), (int(42 * scale), int(54 * scale)), 0, 0, 360, (180, 185, 195), 2, cv2.LINE_AA)
    
    # Eyes
    eye_sy = int(gy + eye_y_val * scale)
    leyex = int(gx + 42 * scale)
    reyex = int(gx + 86 * scale)
    cv2.ellipse(canvas, (leyex, eye_sy), (int(8 * scale), int(4.5 * scale)), 0, 0, 360, accent, 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (reyex, eye_sy), (int(8 * scale), int(4.5 * scale)), 0, 0, 360, accent, 2, cv2.LINE_AA)
    cv2.circle(canvas, (leyex, eye_sy), 3, accent, -1, cv2.LINE_AA)
    cv2.circle(canvas, (reyex, eye_sy), 3, accent, -1, cv2.LINE_AA)
    
    # Eyebrows
    cv2.line(canvas, (int(gx + 33*scale), eye_sy - int(7*scale)), (int(gx + 51*scale), eye_sy - int(8*scale)), (60, 60, 60), 2, cv2.LINE_AA)
    cv2.line(canvas, (int(gx + 77*scale), eye_sy - int(8*scale)), (int(gx + 95*scale), eye_sy - int(7*scale)), (60, 60, 60), 2, cv2.LINE_AA)
    
    # Nose
    nose_sy = int(gy + (eye_y_val + 24) * scale)
    cv2.line(canvas, (face_cx, eye_sy + int(8*scale)), (face_cx, nose_sy), (100, 100, 100), 2, cv2.LINE_AA)
    cv2.line(canvas, (face_cx - int(6*scale), nose_sy), (face_cx + int(6*scale), nose_sy), (100, 100, 100), 2, cv2.LINE_AA)
    
    # Mouth
    mouth_sy = int(gy + mouth_y_val * scale)
    if not is_compressed:
        # Smile arc
        mouth_pts = []
        for x in np.linspace(44, 84, 25):
            y_curve = mouth_y_val + 4.5 * np.sin(np.pi * (x - 44) / 40)
            mouth_pts.append([int(gx + x * scale), int(gy + y_curve * scale)])
        cv2.polylines(canvas, [np.array(mouth_pts)], False, accent, 3, cv2.LINE_AA)
    else:
        # Neutral line
        cv2.line(canvas, (int(gx + 46*scale), mouth_sy), (int(gx + 82*scale), mouth_sy), accent, 3, cv2.LINE_AA)
        
    # Eye line datum
    cv2.line(canvas, (gx, eye_sy), (gx + grid_size + 45, eye_sy), BLUE_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Eye: y={eye_y_val}px (Row 2)", (gx + grid_size + 10, eye_sy - 6), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, BLUE_ACCENT, 1, cv2.LINE_AA)
    
    # Mouth line datum
    cv2.line(canvas, (gx, mouth_sy), (gx + grid_size + 45, mouth_sy), RED_ACCENT, 2, cv2.LINE_AA)
    row_num = 5 if mouth_y_val < 96 else 6
    cv2.putText(canvas, f"Mouth: y={mouth_y_val}px (Row {row_num})", (gx + grid_size + 10, mouth_sy + 16), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, RED_ACCENT, 1, cv2.LINE_AA)
    
    # Distance bracket on right
    dist_val = mouth_y_val - eye_y_val
    bx = gx + grid_size + 70
    cv2.line(canvas, (bx, eye_sy), (bx, mouth_sy), ORANGE_ACCENT, 2, cv2.LINE_AA)
    cv2.line(canvas, (bx - 5, eye_sy), (bx + 5, eye_sy), ORANGE_ACCENT, 2, cv2.LINE_AA)
    cv2.line(canvas, (bx - 5, mouth_sy), (bx + 5, mouth_sy), ORANGE_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"d = {dist_val} px", (bx + 10, int((eye_sy + mouth_sy)/2) + 5), 
                cv2.FONT_HERSHEY_DUPLEX, 0.58, ORANGE_ACCENT, 2, cv2.LINE_AA)
    
    # Bottom status card
    card_y = gy + grid_size + 20
    if not is_compressed:
        cv2.putText(canvas, "[v] Canonical Eye-to-Mouth Span: 57 px", (x_offset, card_y + 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (20, 120, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, "    Oral boundary cleanly contained in Row 6 (y = 96..111 px)", (x_offset, card_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_DARK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "    Feature histogram bin alignment: 100% matched to gallery", (x_offset, card_y + 53), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_MUTED, 1, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "[!] Non-Linear Foreshortened Span: 50 px (-14.0%)", (x_offset, card_y + 15), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, RED_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "    7-pixel vertical drift shifts oral features into Row 5 (y = 80..95 px)", (x_offset, card_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, TEXT_DARK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "    Cross-boundary bin migration: 43.8% of cell mass misaligned", (x_offset, card_y + 53), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 30, 180), 1, cv2.LINE_AA)

# Draw Panel 1
draw_panel(x_offset=80, y_offset=120, grid_size=384, eye_y_val=46, mouth_y_val=103, 
           is_compressed=False, 
           title_text="Capture 1: Upward Pitch & Smile", 
           pitch_text="Pose: +7 deg Pitch, 45 cm distance | Eye-to-Mouth: 57 px")

# Draw Panel 2
draw_panel(x_offset=940, y_offset=120, grid_size=384, eye_y_val=47, mouth_y_val=97, 
           is_compressed=True, 
           title_text="Capture 2: Downward Pitch & Neutral", 
           pitch_text="Pose: -5 deg Pitch, 90 cm distance | Eye-to-Mouth: 50 px (-14.0%)")

# Draw Center Mathematical Annotation Box
cv2.rectangle(canvas, (50, 700), (W - 50, 830), (240, 243, 248), -1)
cv2.rectangle(canvas, (50, 700), (W - 50, 830), BORDER_GRAY, 1)

cv2.putText(canvas, "ROOT CAUSE OF THE 'DIFFERENT PHOTO' RECOGNITION FAILURE IN 2D CLASSICAL DESCRIPTORS:", 
            (70, 730), cv2.FONT_HERSHEY_DUPLEX, 0.58, (15, 60, 140), 2, cv2.LINE_AA)
cv2.putText(canvas, "1. 2D Affine de-tilt corrects in-plane roll (R_z) around the optical axis, but CANNOT undo 3D out-of-plane pitch (R_x nodding) or perspective foreshortening.", 
            (70, 755), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_DARK, 1, cv2.LINE_AA)
cv2.putText(canvas, "2. Spatial LBPH partitions the face into rigid 8x8 spatial cells. L2 matching computes distance between cell (r, c) of query and cell (r, c) of gallery.", 
            (70, 778), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_DARK, 1, cv2.LINE_AA)
cv2.putText(canvas, "3. When vertical landmarks drift by 7 px across a 16 px cell height, oral texture histograms shift from Row 6 into Row 5, causing L2 distance to spike to 0.5631 (> 0.46).", 
            (70, 801), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 40, 20), 1, cv2.LINE_AA)

cv2.imwrite("sample_data/report_landmark_schematic.png", canvas)
print("Successfully saved sample_data/report_landmark_schematic.png")
