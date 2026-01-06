import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import json

# === Model Setup ===
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO vehicle classes
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda")
model.to(device)

# === Video Input ===
VIDEO_PATH = r"D:\Jetson Project Sample Videos\8006.mkv"
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback

# Camera 8005 calibration data
CAM_W, CAM_H = 1280, 720
CAMERA_8005 = {
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
}

H, _ = cv2.findHomography(CAMERA_8005["camera_coords"], CAMERA_8005["world_coords"])

# === Helper Functions ===
def pixel_to_world(u, v, H):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return float(world_point[0, 0]), float(world_point[1, 0])

def get_vehicle_polygons(frame, H):
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
            pts = largest.squeeze(1)
            if len(pts.shape) != 2 or pts.shape[0] < 3:
                continue
            world_pts = [pixel_to_world(px, py, H) for px, py in pts]
            polygons.append(world_pts)
    return polygons

# === Output Storage ===
results = {}
frame_idx = 0
max_frames = int(fps * 60 * 5)  # first 5 minutes
next_print_time = 10  # seconds

while frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
        print(f"Unexpected frame size {frame.shape}, expected {CAM_W}x{CAM_H}")
        break

    timestamp = frame_idx / fps
    polygons = get_vehicle_polygons(frame, H)
    results[f"{timestamp:.2f}"] = polygons

    # Print progress every 10 seconds
    if timestamp >= next_print_time:
        print(f"Processed {int(timestamp)} seconds of video...")
        next_print_time += 10

    frame_idx += 1

cap.release()

# === Save JSON ===
OUTPUT_JSON = r"D:\Jetson Project Sample Videos\vehicle_polygons_8006.json"
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} frames of polygons to {OUTPUT_JSON}")