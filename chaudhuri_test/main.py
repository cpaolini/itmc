import cv2
import os
import pymysql
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from chaudhuri_test.functions import get_coordinates

# Get secrets from dedicated file, not to be accessed in GitHub
secrets_filepath = "C:/Users/shoun/Documents/GitHub/B - Projects/2 - SDSU/Jetson Project/Jetson Code/secrets.txt" # CHANGE THIS!
with open(secrets_filepath, 'r') as file:
    # RTSP_STREAM: RTSP IP to access the live video feed, which should be replaced with the live IP from the server
    # SQL_PASSWORD: The password for the itmc SQL database through notos
    secrets = {}
    for line in file:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            secrets[key] = value.strip('"')  # Remove surrounding quotes if present

# Get options from dedicated file, changeable for each Jetson
with open(os.abspath("options.txt"), 'r') as file:
    # SAVE_VIDEO: Decide whether to write a video with the live boxes to the hard drive or not
    # SAVE_MAP: Decide whether to write a video with the live map view and centroids
    # CAMERA: The code for which camera to run for this Jetson (8005, 8006, 8007, or 8008)
    # CONFIDENCE: Required confidence to allow a YOLO bounding box
    # SAVE_ITERATION: The iteration of videos taken used to name videos without making duplicates or replacing other videos
    lines = file.readlines()

for i, line in enumerate(lines):
    options = {}
    if '=' in line:
        key, value = line.strip().split('=', 1)
        options[key] = value.strip('"')
        if line.startswith('"SAVE_ITERATION"'):
            lines[i] = f'{key} = {options[key] + 1}\n'

with open(os.abspath("options.txt"), 'w') as file:
    file.writelines(lines)

connection = pymysql.connect(host="notos.sdsu.edu", user="itmc", password=secrets["SQL_PASSWORD"], database="itmc")

model = YOLO('yolov8x.pt')
model.overrides['classes'] = [0, 1, 2, 3, 5]

tracker = DeepSort(max_age=5, nn_budget=50)
rtsp_url = secrets["RTSP_STREAM"] + options["CAMERA"]
cap = cv2.VideoCapture(rtsp_url)

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if options["SAVE_MAP"]:
    map_image = cv2.imread(os.abspath("intersection.png"))
    map_scaled = cv2.resize(map_image, (628, 424))
    output_filename = f"map_save_{options["CAMERA"]}_{options["SAVE_ITERATION"]}"
    map_video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (628, 424))
    if not map_video_writer.isOpened(): raise RuntimeError("Map video writer could not be opened properly.")
    else: print("Map video writer created successfully.")

if options["SAVE_VIDEO"]:
    output_filename = f"video-save_{options["CAMERA"]}_{options["SAVE_ITERATION"]}"
    video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    if not map_video_writer.isOpened(): raise RuntimeError("Video writer could not be opened properly.")
    else: print("Video writer created successfully.")

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    map_live = map_scaled.copy()

    results = model(frame)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = box.cls.cpu().numpy()[0]
            if cls in model.overrides['classes']:
                confidence = box.conf.cpu().numpy()[0]
                if confidence >= options["CONFIDENCE"]:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy[0]
                    w, h = int(x2 - x1), int(y2 - y1)
                    detections.append([[int(x1), int(y1), w, h], confidence, int(cls)])

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id
        centroid_x, centroid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if options["SAVE_VIDEO"]: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if options['SAVE_MAP']: 
            global_x, global_y = get_coordinates(options["CAMERA"], 0, (centroid_x, centroid_y))
            map_x, map_y = get_coordinates(0, 1, (global_x, global_y))
            cv2.circle(map_live, (int(map_x), int(map_y)), 5, (0, 0, 255), -1)

    if options["SAVE_VIDEO"]:


    if options["SAVE_MAP"]:

    
    
cap.release()