import numpy as np
import cv2
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

# Paths to the four Jetson .json files
json_files = [
    "for_deployment/vehicle_polygons_8005.json",
    "for_deployment/vehicle_polygons_8006.json",
    "for_deployment/vehicle_polygons_8007.json",
    "for_deployment/vehicle_polygons_8008.json"
]

# Output CSV path
csv_filename = "vehicle_rectangles.csv"
csv_header = [
    "timestamp", "color", "length", "width", "angle", "area",
    "corner1_x", "corner1_y", "corner2_x", "corner2_y", "corner3_x", "corner3_y", "corner4_x", "corner4_y"
]

# Load rectangles from all Jetson .json files
rects_by_timestamp = {}  # {timestamp: [rects]}
for json_path in json_files:
    if not os.path.exists(json_path):
        print(f"Missing {json_path}")
        continue
    with open(json_path, "r") as f:
        data = json.load(f)
        for entry in data:
            timestamp = entry["timestamp"]
            corners = entry.get("corners", [0,0,0,0,0,0,0,0])
            rect = {
                "timestamp": timestamp,
                "color": entry.get("color", "red"),
                "length": entry.get("length", 0),
                "width": entry.get("width", 0),
                "angle": entry.get("angle", 0),
                "area": entry.get("area", 0),
                "corners": corners
            }
            if timestamp not in rects_by_timestamp:
                rects_by_timestamp[timestamp] = []
            rects_by_timestamp[timestamp].append(rect)

# Process each timestamp
with open(csv_filename, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_header)
    for timestamp in sorted(rects_by_timestamp.keys()):
        # Create masks for each camera
        cam_masks = []
        for rect in rects_by_timestamp[timestamp]:
            mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
            corners = np.array(rect["corners"], dtype=np.int32).reshape(4,2)
            cv2.fillPoly(mask, [corners], 1)
            cam_masks.append(mask)
        # Overlap logic
        overlap_count = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
        for mask in cam_masks:
            overlap_count += mask
        vectorized_points_func = np.vectorize(points_for_overlap_count)
        points_grid = vectorized_points_func(overlap_count)
        max_points = max(POINTS_FOR_1_OVERLAP, POINTS_FOR_2_OVERLAP, POINTS_FOR_3_OVERLAP, POINTS_FOR_4_OVERLAP)
        normalized = points_grid.astype(np.float32) / max_points  # 0 to 1

        # Detect overlap regions (pixels with 3 or 4 overlaps)
        overlap_mask = ((overlap_count == 3) | (overlap_count == 4)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            w_exp = w + 2 * RECTANGLE_BUFFER_PX
            h_exp = h + 2 * RECTANGLE_BUFFER_PX
            area = w_exp * h_exp
            snapped = False
            angle_mod = angle % 180
            aspect_ratio = max(w_exp, h_exp) / min(w_exp, h_exp) if min(w_exp, h_exp) > 0 else float('inf')
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
            # --- Clause 1: Extend small rectangles near edge ---
            if area < RECTANGLE_MIN_AREA:
                close_to_edge = np.all((box[:,0] < RECTANGLE_EDGE_DIST_PX) | (box[:,0] > grid_scale - RECTANGLE_EDGE_DIST_PX) |
                                       (box[:,1] < RECTANGLE_EDGE_DIST_PX) | (box[:,1] > grid_scale - RECTANGLE_EDGE_DIST_PX))
                if close_to_edge:
                    dists = np.stack([box[:,0], grid_scale - box[:,0], box[:,1], grid_scale - box[:,1]], axis=1)
                    min_dist = np.min(dists)
                    long_side = max(w_exp, h_exp)
                    short_side = min(w_exp, h_exp)
                    edge_axis = np.argmin(np.min(dists, axis=0)) # 0:left, 1:right, 2:top, 3:bottom
                    if ((edge_axis in [2,3] and w_exp >= h_exp) or (edge_axis in [0,1] and h_exp > w_exp)):
                        if w_exp >= h_exp:
                            if edge_axis == 3:  # bottom
                                max_y = np.max(box[:,1])
                                for i in range(4):
                                    if box[i][1] == max_y:
                                        box[i][1] = grid_scale-1
                            elif edge_axis == 2:  # top
                                min_y = np.min(box[:,1])
                                for i in range(4):
                                    if box[i][1] == min_y:
                                        box[i][1] = 0
                        else:
                            if edge_axis == 0:  # left
                                min_x = np.min(box[:,0])
                                for i in range(4):
                                    if box[i][0] == min_x:
                                        box[i][0] = 0
                            elif edge_axis == 1:  # right
                                max_x = np.max(box[:,0])
                                for i in range(4):
                                    if box[i][0] == max_x:
                                        box[i][0] = grid_scale-1
            split_done = False
            if area > RECTANGLE_MAX_AREA:
                side_ratio = max(w_exp, h_exp) / min(w_exp, h_exp)
                if side_ratio < RECTANGLE_SQUARE_RATIO:
                    for axis in [0,1]:
                        min_axis = np.min(box[:,axis])
                        max_axis = np.max(box[:,axis])
                        slot_start = int(min_axis + (max_axis - min_axis)/2 - RECTANGLE_SLOT_WIDTH_PX/2)
                        slot_end = int(min_axis + (max_axis - min_axis)/2 + RECTANGLE_SLOT_WIDTH_PX/2)
                        if slot_start < 0 or slot_end > grid_scale:
                            continue
                        slot_mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
                        if axis == 0:
                            slot_mask[:, slot_start:slot_end] = 1
                        else:
                            slot_mask[slot_start:slot_end, :] = 1
                        rect_mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
                        cv2.drawContours(rect_mask, [box], 0, 1, -1)
                        slot_points = np.sum(points_grid[slot_mask & rect_mask > 0])
                        slot_area = np.sum(slot_mask & rect_mask > 0)
                        rect_points = np.sum(points_grid[rect_mask > 0])
                        rect_area_px = np.sum(rect_mask > 0)
                        if slot_area > 0 and rect_area_px > 0:
                            slot_density = slot_points / slot_area
                            rect_density = rect_points / rect_area_px
                            if slot_density < rect_density:
                                if axis == 0:
                                    left_box = box.copy()
                                    right_box = box.copy()
                                    left_box[left_box[:,0] > (min_axis + max_axis)/2, 0] = int((min_axis + max_axis)/2)
                                    right_box[right_box[:,0] < (min_axis + max_axis)/2, 0] = int((min_axis + max_axis)/2)
                                    # Write both boxes to CSV
                                    for b in [left_box, right_box]:
                                        corners = [tuple(pt) for pt in b]
                                        writer.writerow([
                                            f\"{timestamp:.3f}\", 'red',
                                            f\"{max(w_exp, h_exp):.2f}\", f\"{min(w_exp, h_exp):.2f}\", f\"{snap_angle:.2f}\", f\"{w_exp * h_exp:.2f}\",
                                            corners[0][0], corners[0][1],
                                            corners[1][0], corners[1][1],
                                            corners[2][0], corners[2][1],
                                            corners[3][0], corners[3][1]])
                                else:
                                    top_box = box.copy()
                                    bottom_box = box.copy()
                                    top_box[top_box[:,1] > (min_axis + max_axis)/2, 1] = int((min_axis + max_axis)/2)
                                    bottom_box[bottom_box[:,1] < (min_axis + max_axis)/2, 1] = int((min_axis + max_axis)/2)
                                    for b in [top_box, bottom_box]:
                                        corners = [tuple(pt) for pt in b]
                                        writer.writerow([
                                            f\"{timestamp:.3f}\", 'red',
                                            f\"{max(w_exp, h_exp):.2f}\", f\"{min(w_exp, h_exp):.2f}\", f\"{snap_angle:.2f}\", f\"{w_exp * h_exp:.2f}\",
                                            corners[0][0], corners[0][1],
                                            corners[1][0], corners[1][1],
                                            corners[2][0], corners[2][1],
                                            corners[3][0], corners[3][1]])
                                split_done = True
                                break
            if split_done:
                continue
            if coverage >= 0.4:
                rect_color = 'orange' if snapped else 'red'
                corners = [tuple(pt) for pt in box]
                writer.writerow([
                    f\"{timestamp:.3f}\", rect_color,
                    f\"{max(w_exp, h_exp):.2f}\", f\"{min(w_exp, h_exp):.2f}\", f\"{snap_angle:.2f}\", f\"{w_exp * h_exp:.2f}\",
                    corners[0][0], corners[0][1],
                    corners[1][0], corners[1][1],
                    corners[2][0], corners[2][1],
                    corners[3][0], corners[3][1]])

print(f\"CSV saved to {csv_filename}\")