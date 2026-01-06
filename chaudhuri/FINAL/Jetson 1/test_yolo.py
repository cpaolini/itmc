import cv2
import torch
import numpy as np
from ultralytics import YOLO

# === Config ===
VIDEO_PATH = r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 1\8005.avi"
MODEL_PATH = "yolo11m-seg.pt"
MASK_COLOR = (0, 255, 0)  # Green for all masks
YOLO_SIZE = 640  # YOLO input size

# Load model
print("[INFO] Loading YOLOv11 segmentation model...")
model = YOLO(MODEL_PATH)
device = "cuda"
model.to(device)
print(f"[INFO] Using device: {device}")

# Open video
print(f"[INFO] Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Failed to open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 10
frame_delay = int(1000 / fps)
print(f"[INFO] Video FPS: {fps}, frame delay set to {frame_delay} ms")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached or failed to read frame")
        break

    h_orig, w_orig = frame.shape[:2]
    print(f"[INFO] Processing frame {frame_idx} (original size: {w_orig}x{h_orig})")

    # Squish frame to YOLO input size
    frame_squished = cv2.resize(frame, (YOLO_SIZE, YOLO_SIZE))
    overlay = frame.copy()  # will overlay masks on original frame

    # Run YOLO segmentation
    results = model(frame_squished, verbose=False)

    for result in results:
        if result.masks is not None:
            num_masks = result.masks.data.shape[0]
            print(f"[INFO] Frame {frame_idx}: {num_masks} masks detected")
            masks = result.masks.data.cpu().numpy()  # [N, H_model, W_model]

            for i in range(num_masks):
                # Resize mask back to original frame size
                mask_resized = cv2.resize(masks[i], (w_orig, h_orig))
                mask_bool = mask_resized > 0.5
                overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(MASK_COLOR, dtype=np.uint8) * 0.5

    # Display original video with masks
    cv2.imshow("YOLOv11 Segmentation Overlay", overlay.astype(np.uint8))

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        print("[INFO] Quitting due to user input")
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print("[INFO] Video processing complete")