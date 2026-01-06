import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider

# --- Path to CSV ---
CSV_PATH = r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\rectangles_log.csv"

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)

# Ensure numeric
for col in ["cx", "cy", "width", "height", "angle"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Get unique timestamps
timestamps = sorted(df["timestamp_ms"].unique())

# --- Set up Matplotlib figure ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.2)  # Space for slider

# Initial timestamp
t_idx = 0
current_t = timestamps[t_idx]

# Plot rectangles for given timestamp
def plot_frame(timestamp):
    ax.clear()
    ax.set_title(f"Rectangles at {timestamp} ms")
    ax.set_aspect("equal", "box")

    # --- Force fixed frame size 800x800 ---
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)

    subset = df[df["timestamp_ms"] == timestamp]

    for _, row in subset.iterrows():
        cx, cy = row["cx"], row["cy"]
        w, h, angle = row["width"], row["height"], row["angle"]

        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            angle=angle,
            linewidth=2,
            edgecolor="blue" if not row["snapped"] else "red",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Draw center point
        ax.plot(cx, cy, "go")

    plt.draw()

# --- Initial plot ---
plot_frame(current_t)

# --- Slider for timestamp ---
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, "Time idx", 0, len(timestamps) - 1,
                valinit=t_idx, valstep=1)

def update(val):
    idx = int(slider.val)
    timestamp = timestamps[idx]
    plot_frame(timestamp)

slider.on_changed(update)

plt.show()
