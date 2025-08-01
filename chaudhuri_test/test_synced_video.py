import cv2
import datetime
import numpy as np
import os

# === Config ===
video_paths = {
    "8005": r"D:\Jetson Project Sample Videos\8005.mkv",
    "8006": r"D:\Jetson Project Sample Videos\8006.mkv",
    "8007": r"D:\Jetson Project Sample Videos\8007.mkv",  # reference
    "8008": r"D:\Jetson Project Sample Videos\8008.mkv",
}

start_times = {
    "8005": datetime.datetime.fromisoformat("2025-07-01T18:45:40.944337"),
    "8006": datetime.datetime.fromisoformat("2025-07-01T18:45:39.957667"),
    "8007": datetime.datetime.fromisoformat("2025-07-01T18:45:42.231810"),
    "8008": datetime.datetime.fromisoformat("2025-07-01T18:45:41.103654"),
}

target_time = start_times["8007"] + datetime.timedelta(minutes=29)
frame_size = (1280, 720)
output_size = (2560, 1440)
output_path = r"D:\Jetson Project Sample Videos\8000.avi"

# === Load videos and seek to target time ===
caps = {}
fps_values = {}

for key, path in video_paths.items():
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_values[key] = fps

    delta = (target_time - start_times[key]).total_seconds()
    frame_offset = max(int(delta * fps), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)

    print(f"{key}: Seek to frame {frame_offset} (Δt = {delta:.3f}s)")
    caps[key] = cap

# === Determine output parameters ===
min_fps = int(min(fps_values.values()))
total_frames = min_fps * 60 * 41  # 70 minutes worth of frames

# === VideoWriter setup ===
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_path = r"D:\Jetson Project Sample Videos\8000_2.avi"
out = cv2.VideoWriter(output_path, fourcc, min_fps, output_size)

# === Frame processing loop ===
for frame_idx in range(total_frames):
    frames = {}
    for key in video_paths.keys():
        ret, frame = caps[key].read()
        if not ret:
            print(f"{key}: End of video or error.")
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # fallback: black frame
        else:
            frame = cv2.resize(frame, frame_size)
        frames[key] = frame

    # Compose 2×2 grid
    top = np.hstack([frames["8005"], frames["8006"]])
    bottom = np.hstack([frames["8007"], frames["8008"]])
    full_frame = np.vstack([top, bottom])

    out.write(full_frame)

    print(f"Writing frame {frame_idx + 1} / {total_frames}", end='\r')

# === Cleanup ===
out.release()
for cap in caps.values():
    cap.release()

print(f"\nSaved 41-minute combined clip to: {output_path}")
