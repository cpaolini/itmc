import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os

# --- Settings ---
INPUT_FOLDER = r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8005"  # folder containing 0.json, 100.json, ...
VIEW_FPS = 10  # Playback speed
grid_scale = 800

print(f"[SETUP] Input folder: {INPUT_FOLDER}")

# --- Gather and sort JSON files ---
json_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
print(f"[SETUP] Found {len(json_files)} JSON files in folder.")
# Sort by numeric value of filename before .json
json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
print("[SETUP] JSON files sorted numerically by filename.")

# --- Setup plot ---
fig, ax = plt.subplots(figsize=(6,6))
print("[SETUP] Matplotlib figure and axis created.")

for idx, json_file in enumerate(json_files):
    frame_path = os.path.join(INPUT_FOLDER, json_file)
    print(f"[PROCESSING] Loading frame {idx}: {json_file}")

    with open(frame_path, "r") as f:
        polygons = json.load(f)
    print(f"[PROCESSING] Loaded {len(polygons)} polygons from frame {json_file}")

    # Extract frame timestamp from filename
    timestamp_ms = int(os.path.splitext(json_file)[0])
    print(f"[PROCESSING] Frame timestamp (ms): {timestamp_ms}")

    ax.clear()
    ax.set_title(f"t={timestamp_ms} ms")
    ax.set_xlim(0, grid_scale)
    ax.set_ylim(0, grid_scale)
    ax.set_aspect("equal")
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")

    # Plot each polygon scaled to grid
    for poly_idx, poly in enumerate(polygons):
        scaled_poly = [[x * grid_scale, y * grid_scale] for x, y in poly]
        p = Polygon(scaled_poly, closed=True, edgecolor="blue", facecolor="cyan", alpha=0.4)
        ax.add_patch(p)
        print(f"    [DEBUG] Plotted polygon {poly_idx} with {len(poly)} points")

    plt.pause(1 / VIEW_FPS)  # Adjust for playback speed (~10 FPS)
    if idx % 10 == 0:
        print(f"[PROGRESS] Processed {idx+1}/{len(json_files)} frames")

print("[DONE] Finished plotting all frames.")
plt.show()