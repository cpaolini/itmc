import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
import os
import time

# === User Config ===
VIDEO_PATH = "rtsp://admin:Computervision20@192.168.1.65"
MODEL_PATH = "yolo11m-seg.pt"  # or "yolo11m-seg.pt" if desired
YOLO_SIZE = 640  # input size for YOLO

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO vehicle classes
BATCH_DURATION_SEC = 10
MAX_FRAMES_PER_BATCH = 100

print("=== Starting Script ===")

# === Model Setup ===
print("[INFO] Loading YOLOv11 segmentation model...")
model = YOLO(MODEL_PATH)
model.eval()
device = torch.device("cuda")
model.to(device)
print(f"[INFO] Using device: {device}")
print("[INFO] Model loaded and ready.")

# === Video Input ===
def open_stream():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream. Retrying...")
        time.sleep(2)
        return open_stream()
    return cap

print(f"[INFO] Opening video: {VIDEO_PATH}")
cap = open_stream()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 10
frame_delay = int(1000 / fps)
print(f"[INFO] Video FPS: {fps}, frame delay set to {frame_delay} ms")

# Camera 8005 calibration data
CAM_W, CAM_H = 1280, 720
CAMERA_8005 = {
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
}
H, _ = cv2.findHomography(CAMERA_8005["camera_coords"], CAMERA_8005["world_coords"])
print("[INFO] Homography matrix calculated.")

# === Helper Functions ===
def pixel_to_world(u, v, H):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return float(world_point[0, 0]), float(world_point[1, 0])

def get_vehicle_polygons(frame, H):
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
output_dir = "/home/jetson/FINAL/Outputs/"
os.makedirs(output_dir, exist_ok=True)
print(f"[INFO] Output directory set: {output_dir}")

batch_data = {}  # store multiple frames
batch_start_time = time.time()
frame_count = 0
frames_since_last_msg = 0  # counter for printing every 10 frames

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame. Reconnecting...")
            cap.release()
            cap = open_stream()
            continue

        if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
            print(f"[ERROR] Unexpected frame size {frame.shape}, expected {CAM_W}x{CAM_H}")
            continue

        polygons = get_vehicle_polygons(frame, H)
        timestamp_ms = int(time.time() * 1000)

        # Save polygons under timestamp key in batch
        batch_data[str(timestamp_ms)] = polygons
        frame_count += 1
        frames_since_last_msg += 1

        # Print message every 10 frames
        if frames_since_last_msg >= 10:
            print(f"[INFO] {frame_count} frames collected in current batch so far...")
            frames_since_last_msg = 0

        # Check if batch is ready to save
        if frame_count >= MAX_FRAMES_PER_BATCH or (time.time() - batch_start_time) >= BATCH_DURATION_SEC:
            batch_filename = os.path.join(output_dir, f"{int(time.time() * 1000)}.json")
            with open(batch_filename, "w") as f:
                json.dump(batch_data, f, indent=2)
            print(f"[INFO] JSON file created: {batch_filename} ({frame_count} frames)")

            # Reset for next batch
            batch_data = {}
            frame_count = 0
            frames_since_last_msg = 0
            batch_start_time = time.time()

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}. Continuing...")
        time.sleep(0.01)
        continue

cap.release()
print("\n=== Processing complete. ===")