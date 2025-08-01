import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from collections import defaultdict
from matplotlib.colors import LogNorm

# === CONSTANTS ===
CSV_FILE = 'D:/Jetson Project Sample Videos/vehicle_rectangles_300sec2.csv'
NUM_ROWS = 300
NUM_COLS = 300
MIN_TIME = 0.2
MAX_TIMESTAMP = 300.0

# === INIT ===
CELL_WIDTH = 800 / NUM_COLS
CELL_HEIGHT = 800 / NUM_ROWS
TIMESTEP = 0.1

# Cell data: (stopwatch, avg_PET, num_updates)
cells = {
    (row, col): {'stopwatch': 0.0, 'avg_pet': 0.0, 'count': 0}
    for row in range(NUM_ROWS)
    for col in range(NUM_COLS)
}

# Load data
df = pd.read_csv(CSV_FILE)
df = df[df['timestamp'] <= MAX_TIMESTAMP]
df.sort_values('timestamp', inplace=True)

# Group rows by timestamp
grouped = df.groupby('timestamp')

# Create cell polygons
cell_polygons = {
    (r, c): box(
        c * CELL_WIDTH,
        r * CELL_HEIGHT,
        (c + 1) * CELL_WIDTH,
        (r + 1) * CELL_HEIGHT
    )
    for r in range(NUM_ROWS)
    for c in range(NUM_COLS)
}

# === MAIN LOOP ===
timestamp = 0.1
while timestamp <= MAX_TIMESTAMP:
    print(f"The current timestamp is {timestamp}")

    if timestamp not in grouped.groups:
        timestamp = round(timestamp + TIMESTEP, 10)
        continue

    rows = grouped.get_group(timestamp)
    cell_intersections = defaultdict(set)  # To track which cells were hit

    for _, row in rows.iterrows():
        corners = [
            (row['corner1_x'], row['corner1_y']),
            (row['corner2_x'], row['corner2_y']),
            (row['corner3_x'], row['corner3_y']),
            (row['corner4_x'], row['corner4_y'])
        ]
        car_poly = Polygon(corners)

        for cell_key, cell_poly in cell_polygons.items():
            if car_poly.intersects(cell_poly):
                cell_intersections[cell_key].add(row['color'])  # avoid duplicates

    # Update stopwatch and PET values
    for cell_key, cell_data in cells.items():
        if cell_key in cell_intersections:
            # Cell is occupied by at least one car
            if cell_data['stopwatch'] >= MIN_TIME:
                prev_total = cell_data['avg_pet'] * cell_data['count']
                cell_data['count'] += 1
                new_avg = (prev_total + cell_data['stopwatch']) / cell_data['count']
                cell_data['avg_pet'] = new_avg
            cell_data['stopwatch'] = 0.0
        else:
            # No car in this cell
            cell_data['stopwatch'] += TIMESTEP

    timestamp = round(timestamp + TIMESTEP, 10)

# Final update
for cell_key, cell_data in cells.items():
    if cell_data['stopwatch'] > 0:
        prev_total = cell_data['avg_pet'] * cell_data['count']
        cell_data['count'] += 1
        new_avg = (prev_total + cell_data['stopwatch']) / cell_data['count']
        cell_data['avg_pet'] = new_avg

# === HEATMAP DATA ===
pet_matrix = np.zeros((NUM_ROWS, NUM_COLS))
count_matrix = np.zeros((NUM_ROWS, NUM_COLS))
max_pet = max([cell['avg_pet'] for cell in cells.values()])
max_count = max([cell['count'] for cell in cells.values()])

for (r, c), data in cells.items():
    pet_matrix[r, c] = data['avg_pet']
    count_matrix[r, c] = data['count']

# === PLOTTING ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Set a small minimum for visualization (avoid log(0))
min_display_pet = np.min(pet_matrix[pet_matrix > 0]) if np.any(pet_matrix > 0) else 1e-2
max_display_pet = np.max(pet_matrix)

pet_img = ax1.imshow(
    pet_matrix,
    cmap='hot',
    norm=LogNorm(vmin=min_display_pet, vmax=max_display_pet if max_display_pet > 0 else 1)
)
ax1.set_title('Average PET per Cell')
fig.colorbar(pet_img, ax=ax1, fraction=0.046, pad=0.04)
if NUM_ROWS <= 50 and NUM_COLS <= 50:
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            ax1.text(c, r, f"{pet_matrix[r, c]:.1f}", ha='center', va='center', fontsize=7)

# Count Heatmap
count_img = ax2.imshow(
    count_matrix,
    cmap='Reds_r',
    vmin=1,
    vmax=max_count if max_count > 1 else 2
)
ax2.set_title('PET Update Count per Cell')
fig.colorbar(count_img, ax=ax2, fraction=0.046, pad=0.04)
if NUM_ROWS <= 50 and NUM_COLS <= 50:
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            ax2.text(c, r, f"{int(count_matrix[r, c])}", ha='center', va='center', fontsize=7)

plt.tight_layout()
plt.show()