import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
import os

# === User Config ===
NUM_FRAMES_TO_PROCESS = 1800  # number of frames to process before stopping
VIDEO_PATH = r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 2\8006.avi"
MODEL_PATH = "yolo11m-seg.pt"  # or "yolo11m-seg.pt" if desired
YOLO_SIZE = 640  # input size for YOLO

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO vehicle classes

print("=== Starting Script ===")

# === Model Setup ===
print("[INFO] Loading YOLOv11 segmentation model...")
model = YOLO(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"[INFO] Using device: {device}")
print("[INFO] Model loaded and ready.")

# === Video Input ===
print(f"[INFO] Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 10
frame_delay = int(1000 / fps)
print(f"[INFO] Video FPS: {fps}, frame delay set to {frame_delay} ms")

# Camera 8006 calibration data
CAM_W, CAM_H = 1280, 720
CAMERA_8006 = {
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
H, _ = cv2.findHomography(CAMERA_8006["camera_coords"], CAMERA_8006["world_coords"])
print("[INFO] Homography matrix calculated.")

# === Helper Functions ===
def pixel_to_world(u, v, H):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return float(world_point[0, 0]), float(world_point[1, 0])

def get_vehicle_polygons(frame, H):
    # Squish to YOLO input size for correct mask detection
    frame_squished = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
    results = model(frame_squished, verbose=False)

    polygons = []
    for result in results:
        if result.masks is None:
            continue

        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy()

        for mask, cls_id, conf in zip(masks, cls_ids, confs):
            if conf < 0.3 or cls_id not in VEHICLE_CLASS_IDS:
                continue

            # Resize mask back to original frame
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                pts = largest.squeeze(1)
                if pts.shape[0] >= 3:
                    world_pts = [pixel_to_world(px, py, H) for px, py in pts]
                    polygons.append(world_pts)
    return polygons

# === Frame Processing ===
frame_idx = 0
output_dir = r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 2\Outputs"
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output directory set: {output_dir}")

while frame_idx < NUM_FRAMES_TO_PROCESS:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break
    if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
        print(f"[ERROR] Unexpected frame size {frame.shape}, expected {CAM_W}x{CAM_H}")
        break

    polygons = get_vehicle_polygons(frame, H)

    # Save polygons to JSON
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    output_path = os.path.join(output_dir, f"{timestamp_ms}.json")
    with open(output_path, "w") as f:
        json.dump(polygons, f, indent=2)

    print(f"[INFO] Processed frame {frame_idx}")

    frame_idx += 1

cap.release()
print("\n=== Processing complete. ===")