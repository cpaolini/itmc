import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

# === Model Setup ===
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO vehicle class IDs for vehicles
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Single video with 4 cameras side by side ===
VIDEO_PATH = r"D:\Jetson Project Sample Videos\8000.avi"
cap = cv2.VideoCapture(VIDEO_PATH)

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

# Compute homographies once
for cam_id, cam in CAMERAS.items():
    H, _ = cv2.findHomography(cam["camera_coords"], cam["world_coords"])
    cam["H"] = H

grid_scale = 800

def new_grid():
    return np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255

def pixel_to_world(u, v, H):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return world_point[0, 0], world_point[1, 0]

def get_vehicle_detections(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)
    with torch.no_grad():
        predictions = model([img_tensor])
    pred = predictions[0]
    masks = pred['masks']
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']
    threshold = 0.5

    vehicle_pixels = []
    keypoints = []
    overlays = frame.copy()

    for i in range(len(labels)):
        if scores[i] >= threshold and labels[i].item() in VEHICLE_CLASS_IDS:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            idx = mask > 128
            ys, xs = np.where(idx)
            vehicle_pixels.extend(zip(xs, ys))

            box = boxes[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bx = cx
            by = y2
            mx = (cx + bx) // 2
            my = (cy + by) // 2

            keypoints.append({
                "centroid": (cx, cy),
                "bottom": (bx, by),
                "midpoint": (mx, my)
            })

            color_mask = np.zeros_like(frame)
            color_mask[idx] = (0, 255, 0)
            overlays = cv2.addWeighted(overlays, 1.0, color_mask, 0.5, 0)

            cv2.circle(overlays, (cx, cy), 4, (0, 0, 255), -1)  # red centroid
            cv2.circle(overlays, (bx, by), 4, (0, 128, 255), -1)  # orange bottom
            cv2.circle(overlays, (mx, my), 4, (255, 0, 0), -1)  # blue midpoint

    return overlays, vehicle_pixels, keypoints

def blend_pixel(img, x, y, color, alpha=0.25):
    """Blend color onto img at (x,y) with given alpha [0-1]."""
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        bg_color = img[y, x].astype(np.float32)
        fg_color = np.array(color, dtype=np.float32)
        blended = (1 - alpha) * bg_color + alpha * fg_color
        img[y, x] = blended.astype(np.uint8)

# Choose which camera index to display (0 to 3)
CAM_DISPLAY_INDEX = 0
camera_ids = list(CAMERAS.keys())

world_grid = new_grid()

while True:
    ret, full_frame = cap.read()
    if not ret or full_frame is None:
        print("No frame captured or frame is None, exiting loop.")
        break

    expected_width = 2560
    expected_height = 1440
    if full_frame.shape[1] != expected_width or full_frame.shape[0] != expected_height:
        print(f"Unexpected frame size: {full_frame.shape}, exiting loop.")
        break

    world_grid[:] = 255
    overlap_count = np.zeros((grid_scale, grid_scale), dtype=np.uint8)  # NEW: track overlaps

    # Split frame into 2 rows x 2 cols = 4 cameras
    cameras_frames = []
    for row in range(2):
        for col in range(2):
            x_start = col * CAM_W   # CAM_W = 1280
            y_start = row * CAM_H   # CAM_H = 720
            cam_frame = full_frame[y_start:y_start + CAM_H, x_start:x_start + CAM_W]
            cameras_frames.append(cam_frame)

    display_frame = None

    for i, cam_id in enumerate(camera_ids):
        cam = CAMERAS[cam_id]
        cam_frame = cameras_frames[i]

        overlays, vehicle_pixels, keypoints = get_vehicle_detections(cam_frame)

        # Draw vehicle pixels on world grid and increment overlap count
        for px, py in vehicle_pixels:
            wx, wy = pixel_to_world(px, py, cam["H"])
            gx = int(wx * grid_scale)
            gy = grid_scale - int(wy * grid_scale)
            if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
                blend_pixel(world_grid, gx, gy, (0, 255, 0), alpha=0.25)
                overlap_count[gy, gx] += 1

        # Draw keypoints on world grid
        for pt in keypoints:
            for color, key in [((0, 0, 255), "centroid"), ((0, 128, 255), "bottom"), ((255, 0, 0), "midpoint")]:
                px, py = pt[key]
                wx, wy = pixel_to_world(px, py, cam["H"])
                gx = int(wx * grid_scale)
                gy = grid_scale - int(wy * grid_scale)
                if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
                    cv2.circle(world_grid, (gx, gy), 4, color, -1)

        if i == CAM_DISPLAY_INDEX:
            display_frame = overlays

    # Detect overlap regions where 2 or more vehicle pixels overlap
    overlap_mask = (overlap_count >= 1).astype(np.uint8) * 255
    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("Overlap Mask", overlap_mask)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_area = cv2.contourArea(cnt)
        rect_area = w * h
        coverage = contour_area / rect_area if rect_area > 0 else 0

        # Only draw rectangles if they cover at least 40% of the overlap area
        if coverage >= 0.4:
            cv2.rectangle(world_grid, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red rectangle

    MAX_W, MAX_H = 1280, 720
    height, width = full_frame.shape[:2]
    scale_w = MAX_W / width
    scale_h = MAX_H / height
    scale = min(scale_w, scale_h, 1.0)

    if scale < 1.0:
        new_w = int(width * scale)
        new_h = int(height * scale)
        resized_frame = cv2.resize(full_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_frame = full_frame

    cv2.imshow("Full 8000.avi Frame", resized_frame)
    cv2.imshow("World Grid", world_grid)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()