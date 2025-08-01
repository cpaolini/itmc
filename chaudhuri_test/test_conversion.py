import cv2
import numpy as np
import time

# # === Calibration Data === (8005)
# camera_coords = np.array([
#     [157, 16], [1149, 18], [728, 1], [742, 19], [876, 26], [758, 36], [846, 41],
#     [779, 55], [810, 56], [764, 70], [813, 75], [83, 92], [710, 86], [150, 96],
#     [854, 99], [217, 104], [645, 102], [283, 115], [571, 118], [905, 123],
#     [350, 131], [486, 134], [387, 151], [968, 150], [417, 152], [1253, 178],
#     [174, 178], [1046, 179], [488, 177], [1173, 183], [63, 189], [1092, 194],
#     [1012, 209], [1138, 209], [559, 209], [931, 228], [1245, 242], [635, 249],
#     [851, 256], [771, 292], [711, 296], [697, 338], [792, 352], [628, 401],
#     [876, 423], [564, 479], [966, 509], [501, 581], [1061, 618], [434, 715]
# ], dtype=np.float32)

# world_coords = np.array([
#     [0, 0], [0, 1], [0.011, 0.559], [0.076, 0.56], [0.106, 0.681], [0.143, 0.568], [0.159, 0.643],
#     [0.205, 0.576], [0.21, 0.601], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
#     [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
#     [0.41, 0.257], [0.419, 0.347], [0.45, 0.287], [0.45, 0.664], [0.454, 0.307], [0.494, 0.844],
#     [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.515, 0.099], [0.528, 0.719],
#     [0.554, 0.659], [0.554, 0.738], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435],
#     [0.62, 0.548], [0.66, 0.5], [0.663, 0.47], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424],
#     [0.781, 0.526], [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357]
# ], dtype=np.float32)

# # === Calibration Data === (8006)
# camera_coords = np.array([
#     [886, 619], [655, 608], [816, 512], [749, 507], [833, 430], [740, 421],
#     [912, 366], [671, 348], [988, 316], [1057, 275], [607, 292], [1279, 241],
#     [1126, 242], [542, 246], [1197, 215], [1164, 213], [1265, 194], [92, 250],
#     [212, 234], [475, 210], [1055, 183], [327, 216], [439, 197], [406, 183],
#     [541, 175], [958, 157], [628, 155], [336, 163], [875, 133], [704, 134],
#     [270, 148], [763, 113], [804, 111], [198, 138], [127, 130], [804, 93],
#     [745, 89], [831, 73], [696, 70], [846, 55], [658, 50], [843, 37],
#     [630, 33], [852, 20], [611, 16], [135, 39], [1187, 6]
# ], dtype=np.float32)

# world_coords = np.array([
#     [0.143, 0.568], [0.159, 0.643], [0.205, 0.576], [0.21, 0.601], [0.258, 0.556], [0.27, 0.593],
#     [0.303, 0.507], [0.333, 0.612], [0.344, 0.456], [0.383, 0.403], [0.392, 0.634], [0.41, 0.257],
#     [0.419, 0.347], [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226], [0.485, 0.909],
#     [0.494, 0.844], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.528, 0.719], [0.554, 0.738],
#     [0.554, 0.659], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435], [0.62, 0.548],
#     [0.639, 0.831], [0.66, 0.5], [0.663, 0.47], [0.675, 0.886], [0.705, 0.942], [0.708, 0.459],
#     [0.722, 0.5], [0.76, 0.424], [0.781, 0.526], [0.816, 0.396], [0.843, 0.547], [0.877, 0.375],
#     [0.906, 0.562], [0.94, 0.357], [0.97, 0.573], [1, 1], [1, 0]
# ], dtype=np.float32)

# # === Calibration Data === (8007)
# camera_coords = np.array([
#     [79, 60], [27, 239], [122, 162], [148, 224], [197, 176], [259, 209],
#     [273, 193], [350, 214], [373, 189], [432, 241], [482, 170], [559, 718],
#     [518, 274], [646, 589], [611, 316], [577, 149], [733, 486], [708, 366],
#     [662, 129], [818, 433], [816, 407], [934, 515], [636, 14], [655, 33],
#     [682, 53], [1065, 619], [733, 108], [897, 343], [716, 74], [761, 97],
#     [791, 87], [821, 123], [969, 293], [892, 150], [835, 70], [1040, 257],
#     [978, 178], [866, 52], [1077, 207], [1106, 227], [887, 35], [899, 19],
#     [1188, 232], [1172, 203], [1237, 185], [1176, 7]
# ], dtype=np.float32)

# world_coords = np.array([
#     [0, 1], [0.076, 0.56], [0.106, 0.681], [0.143, 0.568], [0.159, 0.643], [0.205, 0.576],
#     [0.21, 0.601], [0.258, 0.556], [0.27, 0.593], [0.303, 0.507], [0.333, 0.612], [0.337, 0.151],
#     [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634], [0.41, 0.257], [0.419, 0.347],
#     [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226], [0.479, 0.976], [0.485, 0.909],
#     [0.494, 0.844], [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.528, 0.719],
#     [0.554, 0.738], [0.554, 0.659], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435],
#     [0.62, 0.548], [0.639, 0.831], [0.66, 0.5], [0.663, 0.47], [0.675, 0.886], [0.705, 0.942],
#     [0.708, 0.459], [0.722, 0.5], [0.781, 0.526], [1, 1]
# ], dtype=np.float32)

# === Calibration Data === (8008)
camera_coords = np.array([
    [1163, 30], [1217, 241], [1232, 267], [902, 21], [1110, 210], [898, 35],
    [1142, 283], [886, 50], [1014, 179], [865, 67], [928, 149], [1057, 302],
    [834, 34], [856, 122], [968, 331], [790, 95], [792, 103], [741, 70],
    [1125, 698], [1020, 572], [699, 49], [878, 368], [735, 123], [926, 473],
    [666, 29], [836, 394], [642, 11], [787, 421], [751, 331], [669, 144],
    [669, 282], [693, 493], [589, 168], [588, 242], [600, 578], [508, 211],
    [501, 193], [499, 696], [426, 189], [398, 215], [341, 174], [286, 239],
    [256, 163], [164, 264], [168, 156], [37, 279], [79, 152], [122, 27]
], dtype=np.float32)

world_coords = np.array([
    [0, 0], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
    [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
    [0.41, 0.257], [0.419, 0.347], [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226],
    [0.485, 0.909], [0.494, 0.844], [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782],
    [0.515, 0.099], [0.528, 0.719], [0.529, 0.035], [0.554, 0.738], [0.554, 0.659], [0.555, 0.396],
    [0.583, 0.602], [0.599, 0.785], [0.608, 0.435], [0.62, 0.548], [0.639, 0.831], [0.66, 0.5],
    [0.663, 0.47], [0.675, 0.886], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424], [0.781, 0.526],
    [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357], [1, 0]
], dtype=np.float32)

# === Compute Homography ===
H, _ = cv2.findHomography(camera_coords, world_coords)

# Compute inverse homography: world → camera
H_inv = np.linalg.inv(H)

def world_to_pixel(x, y):
    pt = np.array([x, y, 1.0]).reshape(3, 1)
    cam_pt = H_inv @ pt
    cam_pt /= cam_pt[2, 0]
    return int(cam_pt[0, 0]), int(cam_pt[1, 0])

# Precompute grid world points spaced by 0.05 in both directions
grid_points_world = []
step = 0.1
max_val = 1.0 + 1e-6
x_vals = np.arange(0, max_val, step)
y_vals = np.arange(0, max_val, step)
for x in x_vals:
    for y in y_vals:
        grid_points_world.append((x, y))

def pixel_to_world(u, v):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return world_point[0, 0], world_point[1, 0]

def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    if dist == 0:
        return
    num_dots = int(dist / gap)
    for i in range(0, num_dots, 2):  # step by 2 to create the gaps
        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * i / num_dots),
            int(pt1[1] + (pt2[1] - pt1[1]) * i / num_dots)
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (i + 1) / num_dots),
            int(pt1[1] + (pt2[1] - pt1[1]) * (i + 1) / num_dots)
        )
        cv2.line(img, start, end, color, thickness)

# === Load Video ===
video_path = "D:/camera_8005_video_11.mkv"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read the first frame of the video.")

frame_height, frame_width = frame.shape[:2]
zoom_level = 1.0
min_zoom = 1.0
max_zoom = 16.0
offset_x, offset_y = 0, 0
cursor_position = (0, 0)
clicked_points = []
is_mouse_down = False
last_click_time = 0
CLICK_INTERVAL = 0.1  # 10 times per second

# Project to camera pixel coordinates
grid_points_camera = []
for wx, wy in grid_points_world:
    try:
        cx, cy = world_to_pixel(wx, wy)
        grid_points_camera.append((cx, cy, wx, wy))  # remove frame bounds check
    except:
        continue

# Reshape grid_points_camera into 2D for drawing lines
grid_width = len(x_vals)
grid_height = len(y_vals)

# Create a 2D grid of camera coordinates with same structure
grid_points_2d = [[None for _ in range(grid_width)] for _ in range(grid_height)]
for wx, wy in grid_points_world:
    try:
        cx, cy = world_to_pixel(wx, wy)
        i = int(round((wy) / step))  # row index
        j = int(round((wx) / step))  # col index
        if 0 <= i < grid_height and 0 <= j < grid_width:
            grid_points_2d[i][j] = (cx, cy)
    except:
        continue

# === Grid Image ===
grid_scale = 800
def new_grid():
    return np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255
grid_image = new_grid()

def mouse_callback(event, x, y, flags, param):
    global cursor_position, zoom_level, offset_x, offset_y, clicked_points, is_mouse_down, last_click_time
    frame_width, frame_height = param

    if event == cv2.EVENT_MOUSEMOVE:
        cursor_position = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_down = True

    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        cursor_x, cursor_y = cursor_position
        if flags > 0:
            new_zoom = min(zoom_level * 2, max_zoom)
        else:
            new_zoom = max(zoom_level / 2, min_zoom)
        scale_factor = new_zoom / zoom_level
        cursor_frame_x = (cursor_x / zoom_level) + offset_x
        cursor_frame_y = (cursor_y / zoom_level) + offset_y
        offset_x = max(0, min(frame_width - frame_width / new_zoom, cursor_frame_x - (cursor_x / new_zoom)))
        offset_y = max(0, min(frame_height - frame_height / new_zoom, cursor_frame_y - (cursor_y / new_zoom)))
        zoom_level = new_zoom

cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera View", frame_width, frame_height)
cv2.setMouseCallback("Camera View", mouse_callback, (frame_width, frame_height))

cv2.namedWindow("World Grid", cv2.WINDOW_NORMAL)
cv2.resizeWindow("World Grid", grid_scale, grid_scale)

while True:
    visible_width = int(frame_width / zoom_level)
    visible_height = int(frame_height / zoom_level)
    visible_frame = frame[int(offset_y):int(offset_y + visible_height), int(offset_x):int(offset_x + visible_width)]
    display_frame = cv2.resize(visible_frame, (frame_width, frame_height))
    grid_copy = new_grid()

    current_time = time.time()
    if is_mouse_down and (current_time - last_click_time >= CLICK_INTERVAL):
        adjusted_x = round((cursor_position[0] / zoom_level) + offset_x)
        adjusted_y = round((cursor_position[1] / zoom_level) + offset_y)
        clicked_points.append((adjusted_x, adjusted_y))
        last_click_time = current_time

    # Draw clicked points
    for pt in clicked_points:
        screen_x = int((pt[0] - offset_x) * zoom_level)
        screen_y = int((pt[1] - offset_y) * zoom_level)
        cv2.circle(display_frame, (screen_x, screen_y), 4, (0, 0, 255), -1)
        cv2.putText(display_frame, f"{pt}", (screen_x + 5, screen_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        wx, wy = pixel_to_world(pt[0], pt[1])
        gx = int(wx * grid_scale)
        gy = grid_scale - int(wy * grid_scale)
        if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
            cv2.circle(grid_copy, (gx, gy), 5, (0, 0, 255), -1)
            cv2.putText(grid_copy, f"({wx:.2f},{wy:.2f})", (gx + 5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw cursor hover point
    x, y = cursor_position
    adj_x = int((x / zoom_level) + offset_x)
    adj_y = int((y / zoom_level) + offset_y)
    if 0 <= adj_x < frame_width and 0 <= adj_y < frame_height:
        cv2.circle(display_frame, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(display_frame, f"({adj_x}, {adj_y})", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        wx, wy = pixel_to_world(adj_x, adj_y)
        gx = int(wx * grid_scale)
        gy = grid_scale - int(wy * grid_scale)
        if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
            cv2.circle(grid_copy, (gx, gy), 5, (255, 0, 0), -1)
            cv2.putText(grid_copy, f"({wx:.2f},{wy:.2f})", (gx + 5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Draw world grid points on camera frame
    for cx, cy, wx, wy in grid_points_camera:
        screen_x = int((cx - offset_x) * zoom_level)
        screen_y = int((cy - offset_y) * zoom_level)
        if 0 <= screen_x < frame_width and 0 <= screen_y < frame_height:
            cv2.circle(display_frame, (screen_x, screen_y), 2, (0, 255, 0), -1)
            # Optional: show (x, y) text
            # cv2.putText(display_frame, f"{wx:.2f},{wy:.2f}", (screen_x + 2, screen_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    # Draw faint green grid lines between adjacent camera points (even if off screen)
    for i in range(grid_height):
        for j in range(grid_width):
            pt = grid_points_2d[i][j]
            if pt is None:
                continue
            cx, cy = pt
            screen_x = int((cx - offset_x) * zoom_level)
            screen_y = int((cy - offset_y) * zoom_level)

            # Right neighbor
            if j + 1 < grid_width and grid_points_2d[i][j + 1] is not None:
                cx2, cy2 = grid_points_2d[i][j + 1]
                screen_x2 = int((cx2 - offset_x) * zoom_level)
                screen_y2 = int((cy2 - offset_y) * zoom_level)
                draw_dotted_line(display_frame, (screen_x, screen_y), (screen_x2, screen_y2), (0, 255, 0), 1)

            # Bottom neighbor
            if i + 1 < grid_height and grid_points_2d[i + 1][j] is not None:
                cx2, cy2 = grid_points_2d[i + 1][j]
                screen_x2 = int((cx2 - offset_x) * zoom_level)
                screen_y2 = int((cy2 - offset_y) * zoom_level)
                draw_dotted_line(display_frame, (screen_x, screen_y), (screen_x2, screen_y2), (0, 255, 0), 1)

    # Define the four corners of the camera frame
    camera_corners = np.array([
        [0, 0],
        [1280, 0],
        [1280, 720],
        [0, 720]
    ], dtype=np.float32)

    # Convert camera corners to world coordinates
    world_corners = []
    for (cx, cy) in camera_corners:
        wx, wy = pixel_to_world(cx, cy)
        world_corners.append((wx, wy))

    # Draw polygon connecting those corners on grid_copy
    for i in range(len(world_corners)):
        (wx1, wy1) = world_corners[i]
        (wx2, wy2) = world_corners[(i + 1) % len(world_corners)]
        
        gx1 = int(wx1 * grid_scale)
        gy1 = grid_scale - int(wy1 * grid_scale)
        gx2 = int(wx2 * grid_scale)
        gy2 = grid_scale - int(wy2 * grid_scale)
        
        cv2.line(grid_copy, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)  # Red lines, thickness 2

    cv2.imshow("Camera View", display_frame)
    cv2.imshow("World Grid", grid_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()