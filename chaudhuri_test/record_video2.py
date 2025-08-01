import sys
import cv2
import time
import subprocess
import os
from datetime import datetime

# Map of camera ID to IP suffix
camera_ip_map = {
    8005: '65',
    8006: '66',
    8007: '67',
    8008: '68'
}

# Validate command-line arguments
if len(sys.argv) != 6:
    print("Usage: python script.py <cameraID> <password> <count> <jetson_sudo_password> <duration_minutes>")
    print("Allowed camera IDs: 8005, 8006, 8007, 8008")
    sys.exit(1)

# Parse arguments
try:
    cameraID = int(sys.argv[1])
    count = int(sys.argv[3])
    duration_minutes = int(sys.argv[5])
except ValueError:
    print("Camera ID, count, and duration must be integers.")
    sys.exit(1)

if cameraID not in camera_ip_map:
    print("Invalid camera ID. Must be one of: 8005, 8006, 8007, 8008")
    sys.exit(1)

password = sys.argv[2]
jetson_sudo_password = sys.argv[4]

# Sync system clock
print("Syncing system clock...")
try:
    sync_cmd = f"echo {jetson_sudo_password} | sudo -S ntpdate -u pool.ntp.org"
    result = subprocess.run(sync_cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("System clock synced successfully.")
    else:
        print(f"Clock sync failed: {result.stderr.strip()}")
except Exception as e:
    print(f"Exception during clock sync: {e}")

# Build the RTSP URL
ip_suffix = camera_ip_map[cameraID]
rtsp_url = f'rtsp://admin:Computervision20@192.168.1.{ip_suffix}'

def open_stream():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open RTSP stream for camera {cameraID} at IP ending .{ip_suffix}")
        return None
    return cap

start_time = time.time()
start_time_human = datetime.now()

cap = open_stream()
if cap is None:
    sys.exit(1)

# Get resolution and FPS
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120:
    fps = 10  # fallback FPS

# Use a codec that works reliably across platforms
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG works well with .mkv

output_dir = "/mnt/drive"
os.makedirs(output_dir, exist_ok=True)
output_filename = f"{output_dir}/camera_{cameraID}_video_{count}.mkv"

out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Verify that VideoWriter was initialized
if not out.isOpened():
    print(f"❌ ERROR: Failed to initialize VideoWriter. Check codec, path, and permissions.")
    cap.release()
    sys.exit(1)

interval_start = start_time
interval_frames = 0

print(f"Recording {duration_minutes} minutes of footage from camera {cameraID} with auto-reconnect...")

max_read_failures = 10
read_failures = 0

while time.time() - start_time < duration_minutes * 60:
    ret, frame = cap.read()
    if not ret:
        read_failures += 1
        print(f"Frame read failed ({read_failures}/{max_read_failures}), trying to reconnect...")
        cap.release()
        time.sleep(2)
        cap = open_stream()
        if cap is None:
            print("Reconnect failed, retrying in 5 seconds...")
            time.sleep(5)
            continue
        if read_failures >= max_read_failures:
            print("Max read failures reached, exiting recording loop.")
            break
        continue
    read_failures = 0

    out.write(frame)
    interval_frames += 1

    current_time = time.time()
    if current_time - interval_start >= 10:
        interval_duration = current_time - interval_start
        interval_fps = interval_frames / interval_duration
        print(f"[{int(current_time - start_time)}s] - Avg FPS: {interval_fps:.2f}, Frames in last 10s: {interval_frames}")
        interval_start = current_time
        interval_frames = 0

end_time_human = datetime.now()

# Clean up
cap.release()
out.release()
print(f"Recording saved to {output_filename}")

# Save timestamps
timestamp_filename = output_filename.replace(".mkv", "_times.txt")
with open(timestamp_filename, "w") as f:
    f.write(f"Start time: {start_time_human.isoformat()}\n")
    f.write(f"End time:   {end_time_human.isoformat()}\n")
print(f"Timestamps saved to {timestamp_filename}")

# Get local file size
local_file_size = os.path.getsize(output_filename)

# SCP video
scp_process = subprocess.Popen([
    'sshpass', '-p', password,
    'scp', output_filename,
    f'chaudhuri@notos.sdsu.edu:~/'
])

print("Uploading video via SCP...")

remote_path = os.path.basename(output_filename)
progress_check_interval = 5
start_upload_time = time.time()

while scp_process.poll() is None:
    try:
        ssh_cmd = [
            'sshpass', '-p', password,
            'ssh', 'chaudhuri@notos.sdsu.edu',
            f'stat -c%s {remote_path}'
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            remote_size = int(result.stdout.strip())
            progress = (remote_size / local_file_size) * 100
            elapsed = int(time.time() - start_upload_time)
            print(f"[{elapsed}s] Upload Progress: {progress:.1f}%")
        else:
            print("Waiting for upload to begin...")

    except Exception as e:
        print(f"Could not check remote file size: {e}")

    time.sleep(progress_check_interval)

scp_process.wait()
if scp_process.returncode == 0:
    print("✅ Video transferred successfully to chaudhuri@notos.sdsu.edu")
else:
    print("❌ SCP upload failed.")

# Upload timestamp file
print("Uploading timestamp file via SCP...")
scp_txt = subprocess.run([
    'sshpass', '-p', password,
    'scp', timestamp_filename,
    f'chaudhuri@notos.sdsu.edu:~/'
])

if scp_txt.returncode == 0:
    print("✅ Timestamp file transferred successfully.")
else:
    print("❌ Failed to transfer timestamp file.")
