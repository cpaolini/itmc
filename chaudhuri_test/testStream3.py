# testStream3.py - Clean object tracking implementation

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from coordinates import get_coordinates  # Custom function to map coordinates

def global_to_map(x: float, y: float):
    xnew = (0.2 * (((1981-1022) * x) + 1022))
    ynew = (0.2 * (((596-1555) * y) + 1555))
    return (xnew, ynew)

# Load YOLO model
model = YOLO('yolov8x.pt')
model.overrides['classes'] = [0, 1, 2, 3, 5]  # Filter for specific classes (e.g., vehicles)

# # Initialize DeepSORT tracker
# tracker = DeepSort(max_age=5, nn_budget=500)

# Set RTSP camera URL
cameras = [8005, 8006, 8007]
trackers = [DeepSort(max_age=5, nn_budget=50) for _ in cameras]
# camera = 8007
# rtsp_url = f'rtsp://admin:Computervision20@63.42.242.124:{camera}'
# cap = cv2.VideoCapture(rtsp_url)
rtsp_urls = [f'rtsp://admin:Computervision20@63.42.242.124:{cameras[i]}' for i in range(len(cameras))]
caps = [cv2.VideoCapture(rtsp_url) for rtsp_url in rtsp_urls]

# Load and scale down the map
map_image = cv2.imread("intersection.png")  # Replace with your map file path
map_scaled = cv2.resize(map_image, (628, 424))  # Adjust as needed for display

# Set confidence threshold for detections
confidence_threshold = 0.4

while True:

    frames = []
    for i in range(len(cameras)):
        ret, frame = caps[i].read()
        if not ret:
            break
        frames.append(frame)

    # Create a copy of the scaled-down map for live plotting
    map_live = map_scaled.copy()

    for count, frame in enumerate(frames):
        # Run YOLO model on the frame
        results = model(frame)
        print(f"count: {count}, frame: {frame}")

        # Prepare detections for DeepSORT (format: [x1, y1, w, h, confidence, class_id])
        detections = []
        for result in results:
            for box in result.boxes:
                cls = box.cls.cpu().numpy()[0]  # Class ID
                if cls in model.overrides['classes']:  # Filter only specified classes
                    confidence = box.conf.cpu().numpy()[0]  # Confidence score
                    if confidence >= confidence_threshold:  # Apply confidence threshold
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        w, h = int(x2 - x1), int(y2 - y1)  # Width and height of the bounding box
                        detections.append([[int(x1), int(y1), w, h], confidence, int(cls)])

        # Update tracks with DeepSORT
        tracks = trackers[count].update_tracks(detections, frame=frame)

        # Visualize the tracks and plot centroids on the map
        for track in tracks:
            if not track.is_confirmed():
                continue

            # Extract track coordinates and ID
            x1, y1, x2, y2 = track.to_ltrb()
            track_id = track.track_id
            centroid_x, centroid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Convert centroid to map coordinates
            map_x, map_y = get_coordinates(cameras[count], 0, (centroid_x, centroid_y))
            xnew, ynew = global_to_map(map_x, map_y)

            # Draw bounding box and track ID on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Plot the centroid on the map
            cv2.circle(map_live, (int(xnew), int(ynew)), 5, (0, 0, 255), -1)

        # Display the main frame with tracked objects
        # cv2.imshow(f'Tracked Vehicles {count}]', frame)

    # Display the map with plotted centroids
    cv2.imshow('Map with Tracked Centroids', map_live)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()

cv2.destroyAllWindows()
