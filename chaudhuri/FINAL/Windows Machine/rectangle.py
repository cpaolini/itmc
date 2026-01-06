# import cv2
# import numpy as np
# import json
# import os
# import csv

# # --- User customizable parameters for overlap scoring ---
# POINTS_FOR_1_OVERLAP = 1
# POINTS_FOR_2_OVERLAP = 2
# POINTS_FOR_3_OVERLAP = 6
# POINTS_FOR_4_OVERLAP = 8

# # --- User customizable parameters for rectangle display ---
# RECTANGLE_BUFFER_PX = 0  # Buffer (in pixels) to expand rectangles
# RECTANGLE_SNAP_DEGREES = 20  # Angle threshold (degrees) for snapping to axis

# # --- Snapping ineligibility aspect ratio threshold ---
# RECTANGLE_SNAP_MIN_RATIO = 1.5  # Minimum aspect ratio for snapping eligibility

# # --- User customizable parameters for rectangle logic ---
# RECTANGLE_MIN_AREA = 6000
# RECTANGLE_EDGE_DIST_PX = 300
# RECTANGLE_MAX_AREA = 25000
# RECTANGLE_SQUARE_RATIO = 1.5
# RECTANGLE_SLOT_WIDTH_PX = 20

# grid_scale = 800

# # Folders containing precomputed polygons
# POLYGON_PATH = [
#     r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8005", 
#     r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8006", 
#     r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8007", 
#     r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8008"
# ]

# # CSV output
# csv_filename = "vehicle_rectangles_json1.csv"
# csv_header = [
#     "timestamp", "color", "length", "width", "angle", "area",
#     "corner1_x", "corner1_y",
#     "corner2_x", "corner2_y",
#     "corner3_x", "corner3_y",
#     "corner4_x", "corner4_y"
# ]
# csv_file = open(csv_filename, mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(csv_header)
# print("[INFO] CSV file initialized and header written.")

# # AVI output
# avi_filename = "world_grid_json1.avi"
# avi_fps = 30
# avi_size = (grid_scale, grid_scale)
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# avi_writer = cv2.VideoWriter(avi_filename, fourcc, avi_fps, avi_size)
# print(f"[INFO] AVI writer initialized: {avi_filename}")

# def new_grid():
#     return np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255

# def points_for_overlap_count(count):
#     if count == 1:
#         return POINTS_FOR_1_OVERLAP
#     elif count == 2:
#         return POINTS_FOR_2_OVERLAP
#     elif count == 3:
#         return POINTS_FOR_3_OVERLAP
#     elif count >= 4:
#         return POINTS_FOR_4_OVERLAP
#     return 0

# def load_polygons_from_json(json_path):
#     """
#     Reads a JSON file containing contours and outputs a list of polygons
#     (each contour becomes its own polygon) formatted like OpenCV contours.

#     Parameters:
#         json_path (str): Path to the JSON file.

#     Returns:
#         list: A list of polygons (each is a numpy array of shape (N, 1, 2)).
#     """
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     # Expected structure:
#     # {
#     #   "contours": [
#     #       [[x1, y1], [x2, y2], ...],
#     #       [[x3, y3], [x4, y4], ...],
#     #       ...
#     #   ]
#     # }

#     contours = data.get("contours", [])

#     polygons = []
#     for contour in contours:
#         if contour:  # skip empty contours
#             polygon = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
#             polygons.append(polygon)

#     return polygons

# # Get all JSON filenames
# try:
#     json_files = sorted([f for f in os.listdir(POLYGON_PATH[0]) if f.endswith(".json")])
#     print(f"[INFO] Found {len(json_files)} JSON frames in {POLYGON_PATH[0]}")
# except Exception as e:
#     print(f"[ERROR] Failed to list JSON files: {e}")
#     json_files = []

# for frame_idx, filename in enumerate(json_files):
#     print(f"[INFO] Processing frame {frame_idx+1}/{len(json_files)}: {filename}")
#     world_grid = new_grid()
#     overlap_count = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
#     points_grid = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
#     cam_masks = []

#     # Load polygons for each camera
#     for cam_idx, cam in enumerate(POLYGON_PATH):
#         path = os.path.join(cam, filename)
#         if not os.path.exists(path):
#             cam_masks.append(np.zeros((grid_scale, grid_scale), dtype=np.uint8))
#             print(f"[WARN] File not found: {path}")
#             continue
#         with open(path, 'r') as f:
#             polygons = json.load(f)
#         mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
#         for poly_idx, poly in enumerate(polygons):
#             pts = np.array(poly, dtype=np.int32)
#             cv2.fillPoly(mask, [pts], 1)
#             print(f"[INFO] Camera {cam_idx}, loaded polygon {poly_idx}, min={pts.min(axis=0)}, max={pts.max(axis=0)}")
#         cam_masks.append(mask)

#     # Compute overlap count
#     for mask in cam_masks:
#         overlap_count += mask
#     print(f"[INFO] Overlap count computed. Max overlap = {overlap_count.max()}")

#     # Blend masks onto world grid for visualization
#     for mask in cam_masks:
#         idx = mask > 0
#         world_grid[idx] = (0.75 * world_grid[idx] + 0.25 * np.array([0,255,0])).astype(np.uint8)

#     # Compute points grid
#     vectorized_points_func = np.vectorize(points_for_overlap_count)
#     points_grid = vectorized_points_func(overlap_count)
#     max_points = max(POINTS_FOR_1_OVERLAP, POINTS_FOR_2_OVERLAP, POINTS_FOR_3_OVERLAP, POINTS_FOR_4_OVERLAP)
#     normalized = points_grid.astype(np.float32) / max_points

#     # Create colored visualization
#     World_Grid_2 = np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255
#     World_Grid_2[..., 1] = (255 * (1 - normalized)).astype(np.uint8)
#     World_Grid_2[..., 2] = (255 * (1 - normalized)).astype(np.uint8)

#     # --- NEW: Detect all polygons as potential rectangles ---
#     overlap_mask = (points_grid > 0).astype(np.uint8) * 255
#     contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print(f"[INFO] Found {len(contours)} contours in frame {filename}")

#     timestamp = int(filename.split('.')[0]) / 1000.0

#     for cnt_idx, cnt in enumerate(contours):
#         rect = cv2.minAreaRect(cnt)
#         (cx, cy), (w, h), angle = rect
#         w_exp = w + 2 * RECTANGLE_BUFFER_PX
#         h_exp = h + 2 * RECTANGLE_BUFFER_PX
#         area = w_exp * h_exp
#         snapped = False
#         angle_mod = angle % 180
#         aspect_ratio = max(w_exp, h_exp) / min(w_exp, h_exp) if min(w_exp, h_exp) > 0 else float('inf')
#         allow_snap = (area <= RECTANGLE_MAX_AREA) and (aspect_ratio >= RECTANGLE_SNAP_MIN_RATIO)
#         snap_angle = angle
#         if allow_snap:
#             if min(abs(angle_mod - 0), abs(angle_mod - 180)) <= RECTANGLE_SNAP_DEGREES:
#                 snap_angle = 0
#                 snapped = True
#             elif min(abs(angle_mod - 90), abs(angle_mod - 270)) <= RECTANGLE_SNAP_DEGREES:
#                 snap_angle = 90
#                 snapped = True

#         rect_exp = ((cx, cy), (w_exp, h_exp), snap_angle)
#         box = cv2.boxPoints(rect_exp)
#         box = np.intp(box)

#         # Compute coverage using points grid
#         rect_mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
#         cv2.drawContours(rect_mask, [box], 0, 1, -1)
#         rect_points = np.sum(points_grid[rect_mask > 0])
#         rect_area_px = np.sum(rect_mask > 0)
#         coverage = rect_points / rect_area_px if rect_area_px > 0 else 0
#         print(f"[INFO] Contour {cnt_idx}: points sum={rect_points}, coverage={coverage:.2f}, snapped={snapped}")

#         # Only skip if coverage = 0
#         if coverage == 0:
#             print(f"[INFO] Contour {cnt_idx} skipped due to zero points.")
#             continue

#         # Draw rectangle
#         color = (0, 128, 255) if snapped else (0, 0, 255)
#         cv2.drawContours(world_grid, [box], 0, color, 2)
#         rect_color = 'orange' if snapped else 'red'
#         corners = [tuple(pt) for pt in box]
#         csv_writer.writerow([
#             f"{timestamp:.3f}", rect_color,
#             f"{max(w_exp, h_exp):.2f}", f"{min(w_exp, h_exp):.2f}", f"{snap_angle:.2f}", f"{w_exp * h_exp:.2f}",
#             corners[0][0], corners[0][1],
#             corners[1][0], corners[1][1],
#             corners[2][0], corners[2][1],
#             corners[3][0], corners[3][1]
#         ])

#     avi_writer.write(world_grid)
#     if (frame_idx+1) % 10 == 0:
#         print(f"[INFO] Processed {frame_idx+1}/{len(json_files)} JSON frames.")

# csv_file.close()
# avi_writer.release()
# print("[INFO] Processing complete. CSV and AVI saved.")


import os
import cv2
import json
import numpy as np
import csv

print("[INFO] Starting polygon overlap + rectangle detection pipeline...")

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

# --- Display grid size ---
grid_scale = 800

# --- Iteration identifier for output files ---
ITERATION = 4

print("[INFO] Parameters set.")

# --- Paths to JSON polygon folders ---
POLYGON_PATHS = [
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8005", 
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8006", 
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8007", 
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8008"
]
print(f"[INFO] JSON polygon paths set: {POLYGON_PATHS}")

# --- Output setup ---
csv_filename = f"D:/Jetson Project Sample Videos/rectangles_from_json{ITERATION}.csv"
try:
    csv_file = open(csv_filename, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "rect_color", "w", "h", "angle", "area",
        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"
    ])
    print(f"[INFO] CSV output initialized at {csv_filename}")
except Exception as e:
    print(f"[ERROR] Failed to initialize CSV file: {e}")
    raise

avi_filename = f"D:/Jetson Project Sample Videos/world_grid_from_json{ITERATION}.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
try:
    avi_writer = cv2.VideoWriter(avi_filename, fourcc, 10, (grid_scale, grid_scale))
    print(f"[INFO] AVI writer initialized at {avi_filename}")
except Exception as e:
    print(f"[ERROR] Failed to initialize AVI writer: {e}")
    raise

# --- Utility functions ---
def points_for_overlap_count(count):
    if count == 1:
        return POINTS_FOR_1_OVERLAP
    elif count == 2:
        return POINTS_FOR_2_OVERLAP
    elif count == 3:
        return POINTS_FOR_3_OVERLAP
    elif count >= 4:
        return POINTS_FOR_4_OVERLAP
    else:
        return 0

def load_polygons_from_json(json_path, grid_scale=800):
    """
    Load polygons from a normalized JSON file and scale them to pixel coordinates.
    Each polygon is returned as an OpenCV-style array of shape (N, 1, 2), dtype=int32.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARNING] JSON is not a list in {json_path}, skipping.")
            return []

        polygons = []
        for poly_idx, polygon in enumerate(data):
            if not polygon:
                continue

            # Convert normalized coordinates to pixel coordinates
            scaled_pts = []
            for point in polygon:
                x_norm, y_norm = point
                # Clip small negative or >1 values to [0,1] to avoid going out of bounds
                x_norm = min(max(x_norm, 0.0), 1.0)
                y_norm = min(max(y_norm, 0.0), 1.0)
                x_px = int(x_norm * grid_scale)
                y_px = int(y_norm * grid_scale)
                scaled_pts.append([x_px, y_px])

            # Convert to OpenCV contour format
            poly_array = np.array(scaled_pts, dtype=np.int32).reshape((-1, 1, 2))
            polygons.append(poly_array)

        print(f"[INFO] Loaded {len(polygons)} polygons from {json_path}")
        return polygons

    except FileNotFoundError:
        print(f"[ERROR] JSON file not found: {json_path}")
        return []
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON file: {json_path}")
        return []
    except Exception as e:
        print(f"[ERROR] Unexpected error loading {json_path}: {e}")
        return []

# --- Main loop ---
frame_count = 0
timestamp = 0.0
fps = 10.0
max_seconds = 180
max_frames = int(fps * max_seconds)
print(f"[INFO] Entering main loop. Max frames: {max_frames}")

while True:
    # Build JSON filepaths for all 4 cameras at this timestamp
    filepaths = [os.path.join(path, f"{frame_count*100}.json") for path in POLYGON_PATHS]
    if not all(os.path.exists(fp) for fp in filepaths):
        print(f"[WARNING] Missing JSON files for timestamp {timestamp:.3f}, stopping loop.")
        break

    print(f"[INFO] Processing frame {frame_count + 1} at timestamp {timestamp:.3f}")

    world_grid = np.full((grid_scale, grid_scale, 3), 255, dtype=np.uint8)
    overlap_count = np.zeros((grid_scale, grid_scale), dtype=np.uint8)

    cam_polygons = []
    for filepath in filepaths:
        polygons = load_polygons_from_json(filepath)
        cam_polygons.append(polygons)

    cam_masks = []
    for idx, polygons in enumerate(cam_polygons):
        mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
        for poly in polygons:
            cv2.fillPoly(mask, [poly], 1)
        cam_masks.append(mask)
        print(f"[DEBUG] Camera {idx} mask built with {len(polygons)} polygons.")

    for mask in cam_masks:
        overlap_count += mask

    print(f"[INFO] Overlap count matrix updated. Max overlap value: {overlap_count.max()}")

    for mask in cam_masks:
        idx = mask > 0
        world_grid[idx] = (0.75 * world_grid[idx] + 0.25 * np.array([0, 255, 0])).astype(np.uint8)

    vectorized_points_func = np.vectorize(points_for_overlap_count)
    points_grid = vectorized_points_func(overlap_count)
    print(f"[DEBUG] Points grid created with shape {points_grid.shape}")

    max_points = max(POINTS_FOR_1_OVERLAP, POINTS_FOR_2_OVERLAP,
                     POINTS_FOR_3_OVERLAP, POINTS_FOR_4_OVERLAP)
    normalized = points_grid.astype(np.float32) / max_points

    World_Grid_2 = np.zeros((grid_scale, grid_scale, 3), dtype=np.uint8)
    World_Grid_2[..., 2] = 255
    World_Grid_2[..., 1] = (255 * (1 - normalized)).astype(np.uint8)
    World_Grid_2[..., 2] = (255 * (1 - normalized)).astype(np.uint8)
    print("[INFO] World_Grid_2 heatmap created.")

    # --- Detect overlap regions ---
    overlap_mask = ((overlap_count == 3) | (overlap_count == 4)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[INFO] Found {len(contours)} overlap contours.")

    # --- Draw rectangles + write CSV (same as your original code) ---
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        w_exp = w + 2 * RECTANGLE_BUFFER_PX
        h_exp = h + 2 * RECTANGLE_BUFFER_PX
        area = w_exp * h_exp
        snapped = False
        angle_mod = angle % 180
        aspect_ratio = max(w_exp, h_exp) / min(w_exp, h_exp) if min(w_exp, h_exp) > 0 else float("inf")
        allow_snap = (area <= RECTANGLE_MAX_AREA) and (aspect_ratio >= RECTANGLE_SNAP_MIN_RATIO)
        if allow_snap:
            if min(abs(angle_mod - 0), abs(angle_mod - 180)) <= RECTANGLE_SNAP_DEGREES:
                snap_angle = 0
                snapped = True
            elif min(abs(angle_mod - 90), abs(angle_mod - 270)) <= RECTANGLE_SNAP_DEGREES:
                snap_angle = 90
                snapped = True
            else:
                snap_angle = angle
        else:
            snap_angle = angle
        rect_exp = ((cx, cy), (w_exp, h_exp), snap_angle)
        box = cv2.boxPoints(rect_exp)
        box = np.intp(box)

        contour_area = cv2.contourArea(cnt)
        rect_area = w * h
        coverage = contour_area / rect_area if rect_area > 0 else 0

        print(f"[DEBUG] Rect center=({cx:.1f},{cy:.1f}), w={w_exp:.1f}, h={h_exp:.1f}, "
              f"angle={angle:.1f}, snapped={snapped}, coverage={coverage:.2f}")

        if coverage >= 0.4:
            color = (0, 128, 255) if snapped else (0, 0, 255)
            cv2.drawContours(world_grid, [box], 0, color, 2)
            rect_color = "orange" if snapped else "red"
            corners = [tuple(pt) for pt in box]
            csv_writer.writerow([
                f"{timestamp:.3f}", rect_color,
                f"{max(w_exp, h_exp):.2f}", f"{min(w_exp, h_exp):.2f}",
                f"{snap_angle:.2f}", f"{w_exp * h_exp:.2f}",
                corners[0][0], corners[0][1],
                corners[1][0], corners[1][1],
                corners[2][0], corners[2][1],
                corners[3][0], corners[3][1]
            ])
            print(f"[INFO] Rectangle written to CSV. Color={rect_color}, Area={area:.2f}")

    print(f"[INFO] Finished frame {frame_count + 1}")

    # --- Write to AVI ---
    avi_writer.write(world_grid)
    print(f"[INFO] Frame {frame_count + 1} written to AVI.")

    # --- Step frame/timestamp ---
    frame_count += 1
    timestamp += 1.0 / fps
    if frame_count >= max_frames:
        print(f"[WARNING] Reached {max_seconds} seconds ({max_frames} frames). Stopping.")
        break

csv_file.close()
avi_writer.release()
print("[INFO] Processing complete. Outputs saved.")