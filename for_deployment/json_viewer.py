import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- Settings ---
INPUT_JSON = "D:/Jetson Project Sample Videos/vehicle_polygons_8005.json"
grid_scale = 800

# --- Load JSON ---
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# Sort timestamps numerically
timestamps = sorted(data.keys(), key=lambda x: float(x))

# --- Setup plot ---
fig, ax = plt.subplots(figsize=(6,6))

for t in timestamps:
    ax.clear()
    ax.set_title(f"t={t}s")
    ax.set_xlim(0, grid_scale)
    ax.set_ylim(0, grid_scale)
    ax.set_aspect("equal")
    ax.set_xlabel("X (grid units)")
    ax.set_ylabel("Y (grid units)")

    # Plot each polygon scaled to 800x800 grid
    for poly in data[t]:
        scaled_poly = [[x * grid_scale, y * grid_scale] for x, y in poly]
        p = Polygon(scaled_poly, closed=True, edgecolor="blue", facecolor="cyan", alpha=0.4)
        ax.add_patch(p)

    plt.pause(0.01)  # Adjust for playback speed (~10 FPS)

plt.show()