import json
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- Settings ---
# Specify exact JSON files to load (one per camera)
JSON_FILES = {
    "8005": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8005\1\80300.json",
    "8006": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8006\1\80300.json",
    "8007": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8007\1\80000.json",
    "8008": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8008\1\80000.json"
}

# "C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8007\1\0.json"

VIEW_FPS = 10
grid_scale = 800  # working resolution = 800x800

print(f"[SETUP] Loaded {len(JSON_FILES)} JSON files for cameras.")

# --- Setup discrete colormap ---
colors = [
    (1.0, 1.0, 1.0),    # 0 overlaps → white
    (1.0, 0.75, 0.75),  # 1 overlap → light red
    (1.0, 0.5, 0.5),    # 2 overlaps → darker red
    (1.0, 0.25, 0.25),  # 3 overlaps → even darker
    (1.0, 0.0, 0.0),    # 4 overlaps → red
]
cmap = ListedColormap(colors)
bounds = [0, 1, 2, 3, 4, 5]
norm = BoundaryNorm(bounds, cmap.N)

# --- Setup matplotlib figure ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
img_display = ax.imshow(np.zeros((grid_scale, grid_scale)), origin='lower', cmap=cmap, norm=norm)
ax.set_xlim(0, grid_scale)
ax.set_ylim(0, grid_scale)
ax.set_aspect("equal")
ax.set_xlabel("X (grid units)")
ax.set_ylabel("Y (grid units)")
cbar = fig.colorbar(img_display, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
cbar.set_label("Overlap Count")
print("[SETUP] Matplotlib figure created for display.")

# --- Initialize overlap counter ---
overlap_count = np.zeros((grid_scale, grid_scale), dtype=np.uint8)

# --- Loop through specified JSON files ---
for cam_id, json_path in JSON_FILES.items():
    if not os.path.exists(json_path):
        print(f"[WARNING] JSON file for camera {cam_id} not found: {json_path}")
        continue

    with open(json_path, "r") as f:
        polygons = json.load(f)

    for poly in polygons:
        # Convert polygon coordinates to pixel indices
        pts = np.array([[int(x*grid_scale), int(y*grid_scale)] for x, y in poly], dtype=np.int32)

        # Create binary mask of polygon
        mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        # Increment overlap counter where polygon covers
        overlap_count += mask

# Cap values at 4
overlap_count = np.clip(overlap_count, 0, 4)

# --- Fit rectangles over shaded areas ---
rectangles = []
for overlap_val in range(1, 5):  # only consider non-zero overlaps
    binary_mask = (overlap_count == overlap_val).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangles.append((x, y, w, h))
        cv2.rectangle(overlap_count, (x, y), (x + w, y + h), color=5, thickness=1)  # max color index

# --- Display the result ---
img_display.set_data(overlap_count)
ax.set_title(f"Overlapped Polygons, Rects: {len(rectangles)}")
plt.pause(1 / VIEW_FPS)

plt.ioff()
plt.show()
print("[SUCCESS] Finished plotting specified JSON files.")
