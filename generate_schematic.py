import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set publication style
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), dpi=300)
fig.patch.set_facecolor('#ffffff')

def draw_face_schematic(ax, eye_y, mouth_y, title, subtitle, color_accent, is_compressed=False):
    ax.set_facecolor('#fdfdfd')
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0) # Invert Y to match image coordinates
    
    # Draw 8x8 spatial grid
    for i in range(9):
        coord = i * 16
        lw = 1.0 if i % 2 == 0 else 0.5
        ax.axhline(coord, color='#e0e0e0', linestyle='--', linewidth=lw)
        ax.axvline(coord, color='#e0e0e0', linestyle='--', linewidth=lw)
    
    # Label Row numbers on the left
    for r in range(8):
        y_center = r * 16 + 8
        ax.text(2, y_center + 3, f"R{r}", color='#888888', fontsize=7, fontweight='bold', fontfamily='sans-serif')
        
    # Draw stylized oval face silhouette
    face_oval = patches.Ellipse((64, 64), 84, 108, edgecolor='#444444', facecolor='#f5f7fa', linewidth=1.5, alpha=0.7)
    ax.add_patch(face_oval)
    
    # Eyes
    left_eye = patches.Ellipse((42, eye_y), 14, 8, edgecolor=color_accent, facecolor='#ffffff', linewidth=1.4)
    right_eye = patches.Ellipse((86, eye_y), 14, 8, edgecolor=color_accent, facecolor='#ffffff', linewidth=1.4)
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    ax.plot(42, eye_y, 'o', color=color_accent, markersize=3)
    ax.plot(86, eye_y, 'o', color=color_accent, markersize=3)
    
    # Eyebrows
    ax.plot([33, 42, 51], [eye_y - 8, eye_y - 10, eye_y - 8], color='#333333', linewidth=1.8)
    ax.plot([77, 86, 95], [eye_y - 8, eye_y - 10, eye_y - 8], color='#333333', linewidth=1.8)
    
    # Nose
    ax.plot([64, 62, 66], [eye_y + 10, mouth_y - 14, mouth_y - 14], color='#666666', linewidth=1.2)
    
    # Mouth
    if not is_compressed:
        # Smile
        mouth_x = np.linspace(46, 82, 30)
        mouth_curve = mouth_y + 4 * np.sin(np.pi * (mouth_x - 46) / 36)
        ax.plot(mouth_x, mouth_curve, color=color_accent, linewidth=2.0)
        ax.plot([46, 82], [mouth_y, mouth_y], color=color_accent, linewidth=1.0, linestyle=':')
    else:
        # Neutral
        ax.plot([48, 80], [mouth_y, mouth_y], color=color_accent, linewidth=2.0)
        
    # Eye line datum
    ax.axhline(eye_y, color='#2b5c8f', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.text(100, eye_y - 2, f"y={eye_y}px", color='#2b5c8f', fontsize=8, fontweight='bold')
    
    # Mouth line datum
    ax.axhline(mouth_y, color='#c0392b', linestyle='-', linewidth=1.2, alpha=0.8)
    ax.text(100, mouth_y - 2, f"y={mouth_y}px", color='#c0392b', fontsize=8, fontweight='bold')
    
    # Landmark distance bracket
    dist = mouth_y - eye_y
    ax.annotate('', xy=(18, mouth_y), xytext=(18, eye_y),
                arrowprops=dict(arrowstyle='<->', color='#d35400', lw=1.6))
    ax.text(21, (eye_y + mouth_y)/2 + 3, f"{dist}px", color='#d35400', fontsize=9, fontweight='bold')
    
    # Highlight affected rows
    if is_compressed:
        # Highlight Row 5
        rect = patches.Rectangle((0, 80), 128, 16, linewidth=1, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.15)
        ax.add_patch(rect)
        ax.text(64, 89, "Row 5: Landmark Boundary Migrated Upward", ha='center', color='#c0392b', fontsize=7.5, fontweight='bold')
    else:
        # Highlight Row 6
        rect = patches.Rectangle((0, 96), 128, 16, linewidth=1, edgecolor='#2980b9', facecolor='#2980b9', alpha=0.15)
        ax.add_patch(rect)
        ax.text(64, 105, "Row 6: Canonical Mouth Location", ha='center', color='#2980b9', fontsize=7.5, fontweight='bold')

    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel(subtitle, fontsize=8, color='#333333', labelpad=6)
    ax.set_xticks(range(0, 129, 32))
    ax.set_yticks(range(0, 129, 32))

draw_face_schematic(ax1, eye_y=46, mouth_y=103, 
                    title="Capture 1: Upward Pitch & Smile", 
                    subtitle="Eye-to-Mouth: 57 px | Mouth in Row 6 (y=96..111)", 
                    color_accent='#2b5c8f', is_compressed=False)

draw_face_schematic(ax2, eye_y=47, mouth_y=97, 
                    title="Capture 2: Downward Pitch & Neutral", 
                    subtitle="Eye-to-Mouth: 50 px (-14.0%) | Mouth Migrated into Row 5", 
                    color_accent='#c0392b', is_compressed=True)

plt.suptitle("Schematic 2D Facial Grid Deformation Analysis (Pitch & Expression Foreshortening)", 
             fontsize=11.5, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("sample_data/report_landmark_schematic.png", dpi=300, bbox_inches='tight')
print("Successfully generated sample_data/report_landmark_schematic.png")
