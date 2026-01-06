import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
from matplotlib.path import Path
from PIL import Image

# === CONSTANTS ===
CSV_FILE = 'D:/Jetson Project Sample Videos/rectangles_from_json4.csv'
GRID_WIDTH = 1600
GRID_HEIGHT = 1600
HEATMAP_WIDTH = 800
HEATMAP_HEIGHT = 800
CENTER_X = (GRID_WIDTH - HEATMAP_WIDTH) // 2
CENTER_Y = (GRID_HEIGHT - HEATMAP_HEIGHT) // 2
MIN_TIME = 0.2
MAX_TIMESTAMP = 179900.0
TIMESTEP = 0.1
BACKGROUND_IMAGE = 'C:/Users/shoun/Downloads/intersection.png'
PIXEL_TO_METER = 26.2 / 800  # meters per pixel

# === INIT ARRAYS ===
stopwatch = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
avg_pet = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
count_updates = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32)

# Load and filter data
df = pd.read_csv(CSV_FILE)
df = df[df['timestamp'] <= MAX_TIMESTAMP]
df.sort_values('timestamp', inplace=True)
grouped = df.groupby('timestamp')

# === MAIN LOOP ===
timestamp = 0.1
x0, x1 = CENTER_X, CENTER_X + HEATMAP_WIDTH
y0, y1 = CENTER_Y, CENTER_Y + HEATMAP_HEIGHT

while timestamp <= MAX_TIMESTAMP:
    if timestamp % 100 == 0:
        print(f"The current timestamp is {timestamp}")

    if timestamp not in grouped.groups:
        stopwatch[y0:y1, x0:x1] += TIMESTEP
        timestamp = round(timestamp + TIMESTEP, 10)
        continue

    rows = grouped.get_group(timestamp)
    occupied_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

    for _, row in rows.iterrows():
        corners = [
            (row['x1'] + CENTER_X, row['y1'] + CENTER_Y),
            (row['x2'] + CENTER_X, row['y2'] + CENTER_Y),
            (row['x3'] + CENTER_X, row['y3'] + CENTER_Y),
            (row['x4'] + CENTER_X, row['y4'] + CENTER_Y)
        ]
        poly_path = Path(corners)

        min_x = max(0, int(min(p[0] for p in corners)))
        max_x = min(GRID_WIDTH, int(max(p[0] for p in corners)) + 1)
        min_y = max(0, int(min(p[1] for p in corners)))
        max_y = min(GRID_HEIGHT, int(max(p[1] for p in corners)) + 1)

        xv, yv = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        coords = np.stack((xv.flatten(), yv.flatten()), axis=-1)
        mask = poly_path.contains_points(coords).reshape((max_y - min_y, max_x - min_x))
        occupied_mask[min_y:max_y, min_x:max_x] |= mask

    stationary_mask = ~occupied_mask[y0:y1, x0:x1]
    stopwatch[y0:y1, x0:x1][stationary_mask] += TIMESTEP

    update_mask = occupied_mask[y0:y1, x0:x1] & (stopwatch[y0:y1, x0:x1] >= MIN_TIME)
    count_updates[y0:y1, x0:x1][update_mask] += 1

    avg_pet[y0:y1, x0:x1][update_mask] = (
        (avg_pet[y0:y1, x0:x1][update_mask] * (count_updates[y0:y1, x0:x1][update_mask] - 1)) +
        stopwatch[y0:y1, x0:x1][update_mask]
    ) / count_updates[y0:y1, x0:x1][update_mask]

    stopwatch[y0:y1, x0:x1][occupied_mask[y0:y1, x0:x1]] = 0.0
    timestamp = round(timestamp + TIMESTEP, 10)

# === FINAL UPDATE ===
final_mask = stopwatch[y0:y1, x0:x1] > 0
count_updates[y0:y1, x0:x1][final_mask] += 1
avg_pet[y0:y1, x0:x1][final_mask] = (
    (avg_pet[y0:y1, x0:x1][final_mask] * (count_updates[y0:y1, x0:x1][final_mask] - 1)) +
    stopwatch[y0:y1, x0:x1][final_mask]
) / count_updates[y0:y1, x0:x1][final_mask]

# === LOAD BACKGROUND IMAGE ===
bg_img = Image.open(BACKGROUND_IMAGE).convert('RGBA')
bg_arr = np.array(bg_img)

# === CUSTOM COLORMAPS ===
base_colors_ax1 = [
    (0.0, (1, 0, 0, 1)),
    (0.5, (1, 0.65, 0, 1)),
    (1.0, (1, 1, 0, 1))
]
base_cmap_ax1 = LinearSegmentedColormap.from_list('red_orange_yellow', base_colors_ax1)

def transparent_yellow_at_highest_cmap(cmap):
    colors = cmap(np.linspace(0, 1, 256))
    alphas = np.linspace(0.6, 0.0, 256)
    colors[:, -1] = alphas
    return LinearSegmentedColormap.from_list('transparent_' + cmap.name, colors)

transparent_red_orange_yellow = transparent_yellow_at_highest_cmap(base_cmap_ax1)

transparent_to_dark_red = LinearSegmentedColormap.from_list(
    'transparent_to_dark_red',
    [
        (0.0, (1, 0, 0, 0)),
        (1.0, (0.5, 0, 0, 1))
    ]
)

# === PLOTTING ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

def overlay_heatmap_on_image(ax, data, cmap, title, colorbar_label, alpha=1.0, use_log=False):
    ax.imshow(bg_arr, extent=[0, 1600, 1600, 0])

    masked_data = np.zeros_like(data)
    masked_data[y0:y1, x0:x1] = data[y0:y1, x0:x1]

    vmin = np.min(masked_data[masked_data > 0]) if np.any(masked_data > 0) else 1e-3
    vmax = np.max(masked_data)

    norm = LogNorm(vmin=vmin, vmax=vmax) if use_log else Normalize(vmin=0, vmax=vmax)

    img = ax.imshow(
        masked_data,
        cmap=cmap,
        alpha=alpha,
        extent=[0, 1600, 1600, 0],
        norm=norm,
        interpolation='nearest'
    )

    ax.set_xlim(0, 1600)
    ax.set_ylim(1600, 0)

    tick_interval_px = 400
    ticks_px = np.arange(0, 1601, tick_interval_px)
    ticks_m = ticks_px * PIXEL_TO_METER

    ax.set_xticks(ticks_px)
    ax.set_xticklabels([f"{m:.1f} m" for m in ticks_m])
    ax.set_yticks(ticks_px)
    ax.set_yticklabels([f"{m:.1f} m" for m in ticks_m[::-1]])

    ax.set_title(title)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

# === PLOT PET HEATMAP ===
overlay_heatmap_on_image(
    ax1, avg_pet, cmap=transparent_red_orange_yellow,
    title='PET per Pixel',
    colorbar_label='seconds',
    alpha=1.0,
    use_log=True
)

# === PLOT UPDATE COUNT HEATMAP ===
overlay_heatmap_on_image(
    ax2, count_updates, cmap=transparent_to_dark_red,
    title='PET Update Count per Pixel',
    colorbar_label='vehicles',
    alpha=1.0,
    use_log=False
)

plt.tight_layout()
plt.show()