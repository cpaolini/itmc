import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import csv

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

# === Model Setup === (same as your existing setup)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO vehicle class IDs for vehicles
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda")
model.to(device)

# === Single video with 4 cameras side by side ===
VIDEO_PATH = r"D:\Jetson Project Sample Videos\8000_2.avi"
cap = cv2.VideoCapture(VIDEO_PATH)

# Get FPS for timestamp calculation
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback if FPS is not available

# Output CSV setup
csv_filename = "D:/Jetson Project Sample Videos/vehicle_rectangles_2460sec5.csv"
csv_header = [
    "timestamp", "color", "length", "width", "angle", "area",
    "corner1_x", "corner1_y", "corner2_x", "corner2_y", "corner3_x", "corner3_y", "corner4_x", "corner4_y"
]
csv_file = open(csv_filename, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_header)

# Output AVI setup
avi_filename = "D:/Jetson Project Sample Videos/world_grid_2460sec5.avi"
avi_fps = fps
avi_size = (grid_scale, grid_scale)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
avi_writer = cv2.VideoWriter(avi_filename, fourcc, avi_fps, avi_size)

# Each camera frame size
CAM_W, CAM_H = 1280, 720

# Homography data for the 4 cameras as before, but only coords and offsets here simplified
# Offsets in the big frame: cameras arranged horizontally
CAMERAS = {
    "8005": {
        "video": "C:/Users/shoun/Downloads/camera_8005_video_10.mkv",
        "camera_coords": np.array([
            [157, 16], [1149, 18], [728, 1], [742, 19], [876, 26], [758, 36], [846, 41],
            [779, 55], [810, 56], [764, 70], [813, 75], [83, 92], [710, 86], [150, 96],
            [854, 99], [217, 104], [645, 102], [283, 115], [571, 118], [905, 123],
            [350, 131], [486, 134], [387, 151], [968, 150], [417, 152], [1253, 178],
            [174, 178], [1046, 179], [488, 177], [1173, 183], [63, 189], [1092, 194],
            [1012, 209], [1138, 209], [559, 209], [931, 228], [1245, 242], [635, 249],
            [851, 256], [771, 292], [711, 296], [697, 338], [792, 352], [628, 401],
            [876, 423], [564, 479], [966, 509], [501, 581], [1061, 618], [434, 715]
        ], dtype=np.float32),
        "world_coords": np.array([
            [0, 0], [0, 1], [0.011, 0.559], [0.076, 0.56], [0.106, 0.681], [0.143, 0.568], [0.159, 0.643],
            [0.205, 0.576], [0.21, 0.601], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
            [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
            [0.41, 0.257], [0.419, 0.347], [0.45, 0.287], [0.45, 0.664], [0.454, 0.307], [0.494, 0.844],
            [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.515, 0.099], [0.528, 0.719],
            [0.554, 0.659], [0.554, 0.738], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435],
            [0.62, 0.548], [0.66, 0.5], [0.663, 0.47], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424],
            [0.781, 0.526], [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357]
        ], dtype=np.float32),
        "offset": (0, 0),  # no offset needed here
    },
    "8006": {
        "video": "C:/Users/shoun/Downloads/camera_8006_video_10.mkv",
        "camera_coords": np.array([
            [886, 619], [655, 608], [816, 512], [749, 507], [833, 430], [740, 421],
            [912, 366], [671, 348], [988, 316], [1057, 275], [607, 292], [1279, 241],
            [1126, 242], [542, 246], [1197, 215], [1164, 213], [1265, 194], [92, 250],
            [212, 234], [475, 210], [1055, 183], [327, 216], [439, 197], [406, 183],
            [541, 175], [958, 157], [628, 155], [336, 163], [875, 133], [704, 134],
            [270, 148], [763, 113], [804, 111], [198, 138], [127, 130], [804, 93],
            [745, 89], [831, 73], [696, 70], [846, 55], [658, 50], [843, 37],
            [630, 33], [852, 20], [611, 16], [135, 39], [1187, 6]
        ], dtype=np.float32),
        "world_coords": np.array([
            [0.143, 0.568], [0.159, 0.643], [0.205, 0.576], [0.21, 0.601], [0.258, 0.556], [0.27, 0.593],
            [0.303, 0.507], [0.333, 0.612], [0.344, 0.456], [0.383, 0.403], [0.392, 0.634], [0.41, 0.257],
            [0.419, 0.347], [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226], [0.485, 0.909],
            [0.494, 0.844], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.528, 0.719], [0.554, 0.738],
            [0.554, 0.659], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435], [0.62, 0.548],
            [0.639, 0.831], [0.66, 0.5], [0.663, 0.47], [0.675, 0.886], [0.705, 0.942], [0.708, 0.459],
            [0.722, 0.5], [0.76, 0.424], [0.781, 0.526], [0.816, 0.396], [0.843, 0.547], [0.877, 0.375],
            [0.906, 0.562], [0.94, 0.357], [0.97, 0.573], [1, 1], [1, 0]
        ], dtype=np.float32),
        "offset": (1280, 0),
    },
    "8007": {
        "video": "C:/Users/shoun/Downloads/camera_8007_video_10.mkv",
        "camera_coords": np.array([
            [79, 60], [27, 239], [122, 162], [148, 224], [197, 176], [259, 209],
            [273, 193], [350, 214], [373, 189], [432, 241], [482, 170], [559, 718],
            [518, 274], [646, 589], [611, 316], [577, 149], [733, 486], [708, 366],
            [662, 129], [818, 433], [816, 407], [934, 515], [636, 14], [655, 33],
            [682, 53], [1065, 619], [733, 108], [897, 343], [716, 74], [761, 97],
            [791, 87], [821, 123], [969, 293], [892, 150], [835, 70], [1040, 257],
            [978, 178], [866, 52], [1077, 207], [1106, 227], [887, 35], [899, 19],
            [1188, 232], [1172, 203], [1237, 185], [1176, 7]
        ], dtype=np.float32),
        "world_coords": np.array([
            [0, 1], [0.076, 0.56], [0.106, 0.681], [0.143, 0.568], [0.159, 0.643], [0.205, 0.576],
            [0.21, 0.601], [0.258, 0.556], [0.27, 0.593], [0.303, 0.507], [0.333, 0.612], [0.337, 0.151],
            [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634], [0.41, 0.257], [0.419, 0.347],
            [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226], [0.479, 0.976], [0.485, 0.909],
            [0.494, 0.844], [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.528, 0.719],
            [0.554, 0.738], [0.554, 0.659], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435],
            [0.62, 0.548], [0.639, 0.831], [0.66, 0.5], [0.663, 0.47], [0.675, 0.886], [0.705, 0.942],
            [0.708, 0.459], [0.722, 0.5], [0.781, 0.526], [1, 1]
        ], dtype=np.float32),
        "offset": (0, 720),
    },
    "8008": {
        "video": "C:/Users/shoun/Downloads/camera_8008_video_10.mkv",
        "camera_coords": np.array([
            [1163, 30], [1217, 241], [1232, 267], [902, 21], [1110, 210], [898, 35],
            [1142, 283], [886, 50], [1014, 179], [865, 67], [928, 149], [1057, 302],
            [834, 34], [856, 122], [968, 331], [790, 95], [792, 103], [741, 70],
            [1125, 698], [1020, 572], [699, 49], [878, 368], [735, 123], [926, 473],
            [666, 29], [836, 394], [642, 11], [787, 421], [751, 331], [669, 144],
            [669, 282], [693, 493], [589, 168], [588, 242], [600, 578], [508, 211],
            [501, 193], [499, 696], [426, 189], [398, 215], [341, 174], [286, 239],
            [256, 163], [164, 264], [168, 156], [37, 279], [79, 152], [122, 27]
        ], dtype=np.float32),
        "world_coords": np.array([
            [0, 0], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
            [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
            [0.41, 0.257], [0.419, 0.347], [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226],
            [0.485, 0.909], [0.494, 0.844], [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782],
            [0.515, 0.099], [0.528, 0.719], [0.529, 0.035], [0.554, 0.738], [0.554, 0.659], [0.555, 0.396],
            [0.583, 0.602], [0.599, 0.785], [0.608, 0.435], [0.62, 0.548], [0.639, 0.831], [0.66, 0.5],
            [0.663, 0.47], [0.675, 0.886], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424], [0.781, 0.526],
            [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357], [1, 0]
        ], dtype=np.float32),
        "offset": (1280, 720),
    }
}

def new_grid():
    return np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255

def pixel_to_world(u, v, H):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return world_point[0, 0], world_point[1, 0]

def get_vehicle_polygons(frame, H):
    """Detect vehicles and return list of projected world grid polygons for each car."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)
    with torch.no_grad():
        predictions = model([img_tensor])
    pred = predictions[0]
    masks = pred['masks']
    labels = pred['labels']
    scores = pred['scores']
    threshold = 0.5
    polygons = []
    for i in range(len(labels)):
        if scores[i] >= threshold and labels[i].item() in VEHICLE_CLASS_IDS:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            # Project contour points to world grid
            pts = largest.squeeze(1)
            if len(pts.shape) != 2 or pts.shape[0] < 3:
                continue
            world_pts = []
            for px, py in pts:
                wx, wy = pixel_to_world(px, py, H)
                gx = int(wx * grid_scale)
                gy = grid_scale - int(wy * grid_scale)
                world_pts.append([gx, gy])
            world_pts = np.array(world_pts, dtype=np.int32)
            # Remove out-of-bounds points
            world_pts = world_pts[(world_pts[:,0] >= 0) & (world_pts[:,0] < grid_scale) & (world_pts[:,1] >= 0) & (world_pts[:,1] < grid_scale)]
            if len(world_pts) >= 3:
                polygons.append(world_pts)
    return polygons

def blend_pixel(img, x, y, color, alpha=0.25):
    """Blend color onto img at (x,y) with given alpha [0-1]."""
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        bg_color = img[y, x].astype(np.float32)
        fg_color = np.array(color, dtype=np.float32)
        blended = (1 - alpha) * bg_color + alpha * fg_color
        img[y, x] = blended.astype(np.uint8)

def points_for_overlap_count(count):
    """Return the point value for a pixel given its overlap count."""
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

# --- Replace per-pixel mapping with polygon filling and overlap counting ---
for cam_id, cam in CAMERAS.items():
    H, _ = cv2.findHomography(cam["camera_coords"], cam["world_coords"])
    cam["H"] = H

CAM_DISPLAY_INDEX = 0
camera_ids = list(CAMERAS.keys())

world_grid = new_grid()

start_time_seconds = 1920
start_frame = int(start_time_seconds * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# --- Main processing loop ---
timestamp = 1920.0
frame_count = start_frame
max_seconds = 2460
max_frames = int(max_seconds * fps)

paused = False  # not used, but kept for compatibility

while True:
    ret, full_frame = cap.read()
    if not ret or full_frame is None:
        print("No frame captured or frame is None, exiting loop.")
        break
    # Print current frame number
    print(f"Processing frame {frame_count + 1}")
    expected_width = 2560
    expected_height = 1440
    if full_frame.shape[1] != expected_width or full_frame.shape[0] != expected_height:
        print(f"Unexpected frame size: {full_frame.shape}, exiting loop.")
        break
    world_grid[:] = 255
    overlap_count = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
    points_grid = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
    # Split frame into 4 camera frames
    cameras_frames = []
    for row in range(2):
        for col in range(2):
            x_start = col * CAM_W
            y_start = row * CAM_H
            cam_frame = full_frame[y_start:y_start + CAM_H, x_start:x_start + CAM_W]
            cameras_frames.append(cam_frame)
    cam_polygons = []
    for i, cam_id in enumerate(camera_ids):
        cam = CAMERAS[cam_id]
        cam_frame = cameras_frames[i]
        polygons = get_vehicle_polygons(cam_frame, cam["H"])
        cam_polygons.append(polygons)
    cam_masks = []
    for polygons in cam_polygons:
        mask = np.zeros((grid_scale, grid_scale), dtype=np.uint8)
        for poly in polygons:
            cv2.fillPoly(mask, [poly], 1)
        cam_masks.append(mask)
    for mask in cam_masks:
        overlap_count += mask
    for mask in cam_masks:
        idx = mask > 0
        world_grid[idx] = (0.75 * world_grid[idx] + 0.25 * np.array([0,255,0])).astype(np.uint8)
    vectorized_points_func = np.vectorize(points_for_overlap_count)
    points_grid = vectorized_points_func(overlap_count)
    max_points = max(POINTS_FOR_1_OVERLAP, POINTS_FOR_2_OVERLAP, POINTS_FOR_3_OVERLAP, POINTS_FOR_4_OVERLAP)
    normalized = points_grid.astype(np.float32) / max_points  # 0 to 1
    World_Grid_2 = np.zeros((grid_scale, grid_scale, 3), dtype=np.uint8)
    World_Grid_2[..., 2] = 255  # Blue channel fixed to 255 initially for white base
    World_Grid_2[..., 1] = 255  # Green channel fixed to 255 initially for white base
    World_Grid_2[..., 0] = 255  # Red channel fixed to 255 initially for white base
    World_Grid_2[..., 1] = (255 * (1 - normalized)).astype(np.uint8)  # green channel decreases as red increases
    World_Grid_2[..., 2] = (255 * (1 - normalized)).astype(np.uint8)  # blue channel decreases as red increases
    # Detect overlap regions (pixels with 3 or 4 overlaps)
    overlap_mask = ((overlap_count == 3) | (overlap_count == 4)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw rotated rectangles for overlap regions
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
                                cv2.drawContours(world_grid, [left_box], 0, (0, 0, 255), 2)
                                cv2.drawContours(world_grid, [right_box], 0, (0, 0, 255), 2)
                            else:
                                top_box = box.copy()
                                bottom_box = box.copy()
                                top_box[top_box[:,1] > (min_axis + max_axis)/2, 1] = int((min_axis + max_axis)/2)
                                bottom_box[bottom_box[:,1] < (min_axis + max_axis)/2, 1] = int((min_axis + max_axis)/2)
                                cv2.drawContours(world_grid, [top_box], 0, (0, 0, 255), 2)
                                cv2.drawContours(world_grid, [bottom_box], 0, (0, 0, 255), 2)
                            split_done = True
                            break
        if split_done:
            continue
        if coverage >= 0.4:
            color = (0, 128, 255) if snapped else (0, 0, 255)  # orange if snapped, else red
            cv2.drawContours(world_grid, [box], 0, color, 2)
            rect_color = 'orange' if snapped else 'red'
            corners = [tuple(pt) for pt in box]
            # Write to CSV (only for red/orange rectangles)
            csv_writer.writerow([
                f"{timestamp:.3f}", rect_color,
                f"{max(w_exp, h_exp):.2f}", f"{min(w_exp, h_exp):.2f}", f"{snap_angle:.2f}", f"{w_exp * h_exp:.2f}",
                corners[0][0], corners[0][1],
                corners[1][0], corners[1][1],
                corners[2][0], corners[2][1],
                corners[3][0], corners[3][1]
            ])
    # Print CSV row count after each frame
    print(f"CSV row count (including header): {csv_file.tell() // 2}")  # Approximate, since each row is a line
    # Write World Grid to AVI
    avi_writer.write(world_grid)
    frame_count += 1
    timestamp += 1.0 / fps
    if frame_count >= max_frames:
        print(f"Reached {max_seconds} seconds ({max_frames} frames). Stopping.")
        break
cap.release()
avi_writer.release()
csv_file.close()