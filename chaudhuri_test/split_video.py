import cv2
import os

# === Input/Output Config ===
INPUT_VIDEO = r"D:\Jetson Project Sample Videos\8000_2.avi"
OUTPUT_FOLDER = r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"[SETUP] Output folder ensured at: {OUTPUT_FOLDER}")

OUTPUT_NAMES = {
    "top_left": "8005.avi",
    "top_right": "8006.avi",
    "bottom_left": "8007.avi",
    "bottom_right": "8008.avi"
}

# === Open Video ===
print(f"[SETUP] Opening input video: {INPUT_VIDEO}")
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise Exception(f"[ERROR] Could not open video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30  # fallback
print(f"[SETUP] Video FPS detected: {fps}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[SETUP] Input video size: {frame_width}x{frame_height}")

# Each sub-video dimensions
sub_w, sub_h = frame_width // 2, frame_height // 2
print(f"[SETUP] Each sub-video will be: {sub_w}x{sub_h}")

# === Create VideoWriters for each quadrant ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI-friendly codec
writers = {
    "top_left": cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, OUTPUT_NAMES["top_left"]), fourcc, fps, (sub_w, sub_h)),
    "top_right": cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, OUTPUT_NAMES["top_right"]), fourcc, fps, (sub_w, sub_h)),
    "bottom_left": cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, OUTPUT_NAMES["bottom_left"]), fourcc, fps, (sub_w, sub_h)),
    "bottom_right": cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, OUTPUT_NAMES["bottom_right"]), fourcc, fps, (sub_w, sub_h))
}
print("[SETUP] VideoWriters created for all four quadrants.")

# === Process Frames ===
frame_idx = 0
print("[PROCESSING] Starting frame processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[PROCESSING] End of video reached or cannot read frame.")
        break

    # Split into quadrants
    tl = frame[0:sub_h, 0:sub_w]
    tr = frame[0:sub_h, sub_w:frame_width]
    bl = frame[sub_h:frame_height, 0:sub_w]
    br = frame[sub_h:frame_height, sub_w:frame_width]

    # Write each quadrant
    writers["top_left"].write(tl)
    writers["top_right"].write(tr)
    writers["bottom_left"].write(bl)
    writers["bottom_right"].write(br)

    if frame_idx % 100 == 0:
        print(f"[PROCESSING] Processed frame {frame_idx}")

    frame_idx += 1

# === Release Resources ===
cap.release()
for name, w in writers.items():
    w.release()
    print(f"[FINALIZE] Released writer for {name} ({OUTPUT_NAMES[name]})")

print("[COMPLETE] All four videos saved successfully.")
