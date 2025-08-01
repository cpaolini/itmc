import cv2
import numpy as np
import time
import torch
import torchvision
from torchvision.transforms import functional as F
import datetime

# === Load Model ===
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    raise RuntimeError("CUDA device not available. A GPU with CUDA is required.")
model.to(device)

# === Camera Setup ===
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
        ], dtype=np.float32),  # use your data here
        "world_coords": np.array([
            [0, 0], [0, 1], [0.011, 0.559], [0.076, 0.56], [0.106, 0.681], [0.143, 0.568], [0.159, 0.643],
            [0.205, 0.576], [0.21, 0.601], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
            [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
            [0.41, 0.257], [0.419, 0.347], [0.45, 0.287], [0.45, 0.664], [0.454, 0.307], [0.494, 0.844],
            [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.515, 0.099], [0.528, 0.719],
            [0.554, 0.659], [0.554, 0.738], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435],
            [0.62, 0.548], [0.66, 0.5], [0.663, 0.47], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424],
            [0.781, 0.526], [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357]
        ], dtype=np.float32),   # use your data here
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
    }
}

# === Settings ===
grid_scale = 800
FPS = 10
frame_interval = int(1000 / FPS)
DELAY_8008_FRAMES = 20  # 2 seconds * 10 FPS

def new_grid():
    return np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255

def pixel_to_world(u, v, H):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return world_point[0, 0], world_point[1, 0]

def get_vehicle_overlay(frame):
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

    overlay = frame.copy()
    vehicle_pixel_coords = []
    keypoints = []

    for i in range(len(labels)):
        if scores[i] >= threshold and labels[i].item() in VEHICLE_CLASS_IDS:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            idx = mask > 128
            ys, xs = np.where(idx)
            vehicle_pixel_coords.extend(zip(xs, ys))

            # Bounding box & keypoints
            box = boxes[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bx = (x1 + x2) // 2
            by = y2
            mx = (cx + bx) // 2
            my = (cy + by) // 2

            keypoints.append({
                "centroid": (cx, cy),
                "bottom": (bx, by),
                "midpoint": (mx, my)
            })

            # Green mask overlay
            colored_mask = cv2.merge([mask // 2, mask, mask // 2])
            overlay[idx] = cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0)[idx]

    return overlay, vehicle_pixel_coords, keypoints

# === Initialize Video Feeds and Homographies ===
for cam_id, cam_data in CAMERAS.items():
    cap = cv2.VideoCapture(cam_data["video"])
    if not cap.isOpened():
        print(f"❌ Could not open video for camera {cam_id}")
        exit()
    cam_data["cap"] = cap
    cam_data["H"], _ = cv2.findHomography(cam_data["camera_coords"], cam_data["world_coords"])
    cam_data["frame_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_data["frame_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow(f"Camera {cam_id}", cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f"Camera {cam_id}", cam_data["frame_width"], cam_data["frame_height"])

cv2.namedWindow("World Grid", cv2.WINDOW_NORMAL)
cv2.resizeWindow("World Grid", grid_scale, grid_scale)

# Define real video start/end times
video_times = {
    "8005": ("2025-07-01T18:36:33.917337", "2025-07-01T18:36:48.931052"),
    "8006": ("2025-07-01T18:36:34.158518", "2025-07-01T18:36:52.066948"),
    "8007": ("2025-07-01T18:36:34.331567", "2025-07-01T18:36:50.699164"),
    "8008": ("2025-07-01T18:36:36.310494", "2025-07-01T18:36:51.325068"),
}

# Build staggered timeline
frame_interval_ms = 100
master_timeline = []

for cam_id, (start_str, end_str) in video_times.items():
    start = datetime.datetime.fromisoformat(start_str)
    end = datetime.datetime.fromisoformat(end_str)
    t = start
    while t <= end:
        master_timeline.append((t, cam_id))
        t += datetime.timedelta(milliseconds=frame_interval_ms)

master_timeline.sort()
frame_indices = {cam_id: 0 for cam_id in CAMERAS}

world_grid = new_grid()

# Define colors per camera for world grid shading (BGR format)
CAM_COLORS = {
    "8005": (0, 0, 255),      # Red
    "8006": (0, 165, 255),    # Orange
    "8007": (0, 255, 0),      # Green
    "8008": (255, 0, 0),      # Blue
}

# Pre-run 20 frames for all except 8008
for _ in range(DELAY_8008_FRAMES):
    for cam_id, cam_data in CAMERAS.items():
        if cam_id == "8008":
            continue  # don't touch 8008 yet
        cam_data["cap"].read()
        frame_indices[cam_id] += 1

# Reset 8008 to beginning, just in case
CAMERAS["8008"]["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_indices["8008"] = 0
delay_8008_passed = False

# === Main Loop ===
while True:
    world_grid = new_grid()

    all_done = True
    for cam_id, cam_data in CAMERAS.items():
        # Wait for 2 seconds before activating 8008
        if cam_id == "8008" and not delay_8008_passed:
            delay_8008_passed = True
            continue

        ret, frame = cam_data["cap"].read()
        if not ret:
            continue
        all_done = False

        H = cam_data["H"]
        overlay, pixel_coords, keypoints = get_vehicle_overlay(frame)

        cam_color = CAM_COLORS.get(cam_id, (0, 255, 0))

        for px, py in pixel_coords:
            wx, wy = pixel_to_world(px, py, H)
            gx = int(wx * grid_scale)
            gy = grid_scale - int(wy * grid_scale)
            if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
                cv2.circle(world_grid, (gx, gy), 1, cam_color, -1)

        for pt in keypoints:
            for color, key in [((0, 0, 255), "centroid"), ((0, 128, 255), "bottom"), ((255, 0, 0), "midpoint")]:
                px, py = pt[key]
                cv2.circle(overlay, (px, py), 4, color, -1)

                wx, wy = pixel_to_world(px, py, H)
                gx = int(wx * grid_scale)
                gy = grid_scale - int(wy * grid_scale)
                if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
                    cv2.circle(world_grid, (gx, gy), 4, color, -1)

        cv2.imshow(f"Camera {cam_id}", overlay)
        frame_indices[cam_id] += 1

    cv2.imshow("World Grid", world_grid)
    if cv2.waitKey(frame_interval) & 0xFF == ord('q') or all_done:
        break


# === Cleanup ===
for cam_data in CAMERAS.values():
    cam_data["cap"].release()
cv2.destroyAllWindows()