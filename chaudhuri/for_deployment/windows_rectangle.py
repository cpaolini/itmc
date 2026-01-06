import json
import csv
import os

# --- User customizable parameters for overlap scoring ---
POINTS_FOR_1_OVERLAP = 1
POINTS_FOR_2_OVERLAP = 2
POINTS_FOR_3_OVERLAP = 6
POINTS_FOR_4_OVERLAP = 8

# --- User customizable parameters for rectangle display ---
RECTANGLE_BUFFER_PX = 0  # Buffer (in pixels) to expand rectangles
RECTANGLE_SNAP_DEGREES = 20  # Angle threshold (degrees) for snapping to axis

# --- Snapping ineligibility aspect ratio threshold ---
RECTANGLE_SNAP_MIN_RATIO = 1.5  # Minimum aspect ratio for snapping eligibility

# --- User customizable parameters for rectangle logic ---
RECTANGLE_MIN_AREA = 6000
RECTANGLE_EDGE_DIST_PX = 300
RECTANGLE_MAX_AREA = 25000
RECTANGLE_SQUARE_RATIO = 1.5
RECTANGLE_SLOT_WIDTH_PX = 20

grid_scale = 800

# Paths to the four Jetson .json files
json_files = [
    "for_deployment/jetson1_rectangles.json",
    "for_deployment/jetson2_rectangles.json",
    "for_deployment/jetson3_rectangles.json",
    "for_deployment/jetson4_rectangles.json"
]

# Output CSV path
csv_filename = "vehicle_rectangles.csv"
csv_header = [
    "timestamp", "camera_id", "color", "length", "width", "angle", "area",
    "corner1_x", "corner1_y", "corner2_x", "corner2_y", "corner3_x", "corner3_y", "corner4_x", "corner4_y"
]

# Load rectangles from all Jetson .json files
all_rectangles = {}  # {timestamp: [rects]}
for cam_idx, json_path in enumerate(json_files):
    camera_id = str(8005 + cam_idx)
    if not os.path.exists(json_path):
        print(f"Missing {json_path}")
        continue
    with open(json_path, "r") as f:
        data = json.load(f)
        for entry in data:
            timestamp = entry["timestamp"]
            rect = [
                timestamp,
                camera_id,
                entry.get("color", "red"),
                entry.get("length", 0),
                entry.get("width", 0),
                entry.get("angle", 0),
                entry.get("area", 0),
                *entry.get("corners", [0,0,0,0,0,0,0,0])  # [x1,y1,x2,y2,x3,y3,x4,y4]
            ]
            if timestamp not in all_rectangles:
                all_rectangles[timestamp] = []
            all_rectangles[timestamp].append(rect)

# Write combined rectangles to CSV, sorted by timestamp
with open(csv_filename, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)
    for timestamp in sorted(all_rectangles.keys()):
        for rect in all_rectangles[timestamp]:
            writer.writerow(rect)

print(f"CSV saved to {csv_filename}")
