import cv2
import numpy as np
from coordinates import get_coordinates

# RTSP stream URL
camera = 8008
rtsp_url = f'rtsp://admin:Computervision20@63.42.242.124:{camera}'

# Initialize global variables
clicked_points = []
cursor_position = (0, 0)
zoom_level = 1.0
min_zoom = 1.0
max_zoom = 16.0
offset_x, offset_y = 0, 0

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global clicked_points, cursor_position, zoom_level, offset_x, offset_y
    frame_width, frame_height = param

    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        # Use the snapped (rounded) coordinates of the blue circle
        adjusted_cursor_x = round((x / zoom_level) + offset_x)
        adjusted_cursor_y = round((y / zoom_level) + offset_y)
        clicked_points.append((adjusted_cursor_x, adjusted_cursor_y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click
        # Use the snapped coordinates of the blue circle for detection
        adjusted_cursor_x = round((x / zoom_level) + offset_x)
        adjusted_cursor_y = round((y / zoom_level) + offset_y)

        # Determine deletion radius based on zoom level
        if zoom_level == min_zoom:  # Maximum zoom out
            delete_radius = 6
        elif zoom_level == max_zoom / 2:  # One click zoomed in
            delete_radius = 4
        else:  # Other zoom levels
            delete_radius = 2
        
        delete_radius_squared = delete_radius ** 2

        # Remove points within the determined radius
        clicked_points = [
            point for point in clicked_points
            if (point[0] - adjusted_cursor_x) ** 2 + (point[1] - adjusted_cursor_y) ** 2 > delete_radius_squared
        ]
    elif event == cv2.EVENT_MOUSEMOVE:  # Update cursor position
        cursor_position = (x, y)
    elif event == cv2.EVENT_MOUSEWHEEL:  # Handle zooming
        cursor_x, cursor_y = cursor_position
        if flags > 0:  # Scroll up (zoom in)
            new_zoom = min(zoom_level * 2, max_zoom)
        elif flags < 0:  # Scroll down (zoom out)
            new_zoom = max(zoom_level / 2, min_zoom)

        # Calculate the offsets to maintain cursor position
        scale_factor = new_zoom / zoom_level
        cursor_frame_x = (cursor_x / zoom_level) + offset_x
        cursor_frame_y = (cursor_y / zoom_level) + offset_y
        offset_x = max(0, min(frame_width - frame_width / new_zoom, cursor_frame_x - (cursor_x / new_zoom)))
        offset_y = max(0, min(frame_height - frame_height / new_zoom, cursor_frame_y - (cursor_y / new_zoom)))

        # Update the zoom level
        zoom_level = new_zoom

plot = []
for xin in range(0, 11):
    for yin in range(0, 11):
        plot.append((get_coordinates(0, camera, (xin/10, yin/10)), (xin/10, yin/10)))

# Initialize the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"Failed to connect to RTSP stream at {rtsp_url}")
    exit()

# Get initial frame dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to retrieve initial frame. Exiting...")
    cap.release()
    exit()

frame_height, frame_width = frame.shape[:2]
window_width, window_height = frame_width, frame_height  # Match window size to video resolution

# Set up the display window and mouse callback
cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP Stream", window_width, window_height)
cv2.setMouseCallback("RTSP Stream", mouse_callback, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # Read a frame from the RTSP stream
    if not ret:
        print("Failed to retrieve frame. Exiting...")
        break

    # Calculate the visible area based on zoom and offsets
    visible_width = int(frame_width / zoom_level)
    visible_height = int(frame_height / zoom_level)
    visible_frame = frame[int(offset_y):int(offset_y + visible_height), int(offset_x):int(offset_x + visible_width)]

    # Resize the visible area to fit the window size
    resized_frame = cv2.resize(visible_frame, (window_width, window_height))

    # Draw clicked points
    for point in clicked_points:
        # Adjust points for zoom and offsets
        screen_x = int((point[0] - offset_x) * zoom_level)
        screen_y = int((point[1] - offset_y) * zoom_level)
        if 0 <= screen_x < window_width and 0 <= screen_y < window_height:
            cv2.circle(resized_frame, (screen_x, screen_y), 3, (0, 0, 255), -1)  # Small red dot
            cv2.putText(resized_frame, f"{point}", (screen_x + 5, screen_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)  # Coordinates in red
            
    # Draw plot points
    for points in plot:
        point = points[0]
        # Adjust points for zoom and offsets
        screen_x = int((point[0] - offset_x) * zoom_level)
        screen_y = int((point[1] - offset_y) * zoom_level)
        if 0 <= screen_x < window_width and 0 <= screen_y < window_height:
            cv2.circle(resized_frame, (screen_x, screen_y), 3, (0, 255, 0), -1)  # Small green dot
            cv2.putText(resized_frame, f"{points[1]}", (screen_x + 5, screen_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)  # Coordinates in green

    # Update the current cursor position
    cursor_x, cursor_y = cursor_position
    adjusted_cursor_x = round((cursor_x / zoom_level) + offset_x)
    adjusted_cursor_y = round((cursor_y / zoom_level) + offset_y)

    # Convert back to screen coordinates for display
    screen_cursor_x = int((adjusted_cursor_x - offset_x) * zoom_level)
    screen_cursor_y = int((adjusted_cursor_y - offset_y) * zoom_level)

    # Draw the blue circle at the nearest full pixel
    if 0 <= screen_cursor_x < window_width and 0 <= screen_cursor_y < window_height:
        cv2.circle(resized_frame, (screen_cursor_x, screen_cursor_y), 3, (255, 0, 0), -1)  # Small blue dot
        cv2.putText(resized_frame, f"({adjusted_cursor_x}, {adjusted_cursor_y})", 
                    (screen_cursor_x + 5, screen_cursor_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)  # Coordinates in blue

    # Display the video frame
    cv2.imshow("RTSP Stream", resized_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()