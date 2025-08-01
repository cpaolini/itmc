import cv2
import numpy as np
import time
import torch
import torchvision
from torchvision.transforms import functional as F

# === Vehicle Detection Model Setup ===
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car=2, motorcycle=3, bus=5, truck=7
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # === Calibration Data === (8005)
camera_coords = np.array([
    [157, 16], [1149, 18], [728, 1], [742, 19], [876, 26], [758, 36], [846, 41],
    [779, 55], [810, 56], [764, 70], [813, 75], [83, 92], [710, 86], [150, 96],
    [854, 99], [217, 104], [645, 102], [283, 115], [571, 118], [905, 123],
    [350, 131], [486, 134], [387, 151], [968, 150], [417, 152], [1253, 178],
    [174, 178], [1046, 179], [488, 177], [1173, 183], [63, 189], [1092, 194],
    [1012, 209], [1138, 209], [559, 209], [931, 228], [1245, 242], [635, 249],
    [851, 256], [771, 292], [711, 296], [697, 338], [792, 352], [628, 401],
    [876, 423], [564, 479], [966, 509], [501, 581], [1061, 618], [434, 715]
], dtype=np.float32)

world_coords = np.array([
    [0, 0], [0, 1], [0.011, 0.559], [0.076, 0.56], [0.106, 0.681], [0.143, 0.568], [0.159, 0.643],
    [0.205, 0.576], [0.21, 0.601], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
    [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
    [0.41, 0.257], [0.419, 0.347], [0.45, 0.287], [0.45, 0.664], [0.454, 0.307], [0.494, 0.844],
    [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782], [0.515, 0.099], [0.528, 0.719],
    [0.554, 0.659], [0.554, 0.738], [0.555, 0.396], [0.583, 0.602], [0.599, 0.785], [0.608, 0.435],
    [0.62, 0.548], [0.66, 0.5], [0.663, 0.47], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424],
    [0.781, 0.526], [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357]
], dtype=np.float32)

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
# camera_coords = np.array([
#     [1163, 30], [1217, 241], [1232, 267], [902, 21], [1110, 210], [898, 35],
#     [1142, 283], [886, 50], [1014, 179], [865, 67], [928, 149], [1057, 302],
#     [834, 34], [856, 122], [968, 331], [790, 95], [792, 103], [741, 70],
#     [1125, 698], [1020, 572], [699, 49], [878, 368], [735, 123], [926, 473],
#     [666, 29], [836, 394], [642, 11], [787, 421], [751, 331], [669, 144],
#     [669, 282], [693, 493], [589, 168], [588, 242], [600, 578], [508, 211],
#     [501, 193], [499, 696], [426, 189], [398, 215], [341, 174], [286, 239],
#     [256, 163], [164, 264], [168, 156], [37, 279], [79, 152], [122, 27]
# ], dtype=np.float32)

# world_coords = np.array([
#     [0, 0], [0.258, 0.556], [0.27, 0.593], [0.286, 0.032], [0.303, 0.507], [0.309, 0.094],
#     [0.333, 0.612], [0.337, 0.151], [0.344, 0.456], [0.371, 0.206], [0.383, 0.403], [0.392, 0.634],
#     [0.41, 0.257], [0.419, 0.347], [0.45, 0.664], [0.45, 0.287], [0.454, 0.307], [0.475, 0.226],
#     [0.485, 0.909], [0.494, 0.844], [0.496, 0.163], [0.503, 0.698], [0.505, 0.354], [0.508, 0.782],
#     [0.515, 0.099], [0.528, 0.719], [0.529, 0.035], [0.554, 0.738], [0.554, 0.659], [0.555, 0.396],
#     [0.583, 0.602], [0.599, 0.785], [0.608, 0.435], [0.62, 0.548], [0.639, 0.831], [0.66, 0.5],
#     [0.663, 0.47], [0.675, 0.886], [0.708, 0.459], [0.722, 0.5], [0.76, 0.424], [0.781, 0.526],
#     [0.816, 0.396], [0.843, 0.547], [0.877, 0.375], [0.906, 0.562], [0.94, 0.357], [1, 0]
# ], dtype=np.float32)

# === Compute Homography ===
H, _ = cv2.findHomography(camera_coords, world_coords)

def pixel_to_world(u, v):
    point = np.array([u, v, 1.0]).reshape(3, 1)
    world_point = H @ point
    world_point /= world_point[2, 0]
    return world_point[0, 0], world_point[1, 0]

# === Load Video ===
video_path = "C:/Users/shoun/Downloads/camera_8005_video_2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Failed to open video file.")
    exit()

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

zoom_level = 1.0
min_zoom = 1.0
max_zoom = 16.0
offset_x, offset_y = 0, 0
cursor_position = (0, 0)
clicked_points = []
is_mouse_down = False
last_click_time = 0
CLICK_INTERVAL = 0.1  # 10 times per second
FPS = 10
frame_interval = int(1000 / FPS)  # milliseconds

# === Grid Image ===
grid_scale = 800
def new_grid():
    return np.ones((grid_scale, grid_scale, 3), dtype=np.uint8) * 255

def get_vehicle_mask_overlay_with_visual(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(device)
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    masks = pred['masks']
    boxes = pred['boxes']
    labels = pred['labels']
    scores = pred['scores']
    threshold = 0.5

    overlay = frame.copy()
    vehicle_pixel_coords = []
    box_centroids = []  # For storing visual keypoints

    for i in range(len(labels)):
        if scores[i] >= threshold and labels[i].item() in VEHICLE_CLASS_IDS:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            idx = mask > 128
            ys, xs = np.where(idx)
            vehicle_pixel_coords.extend(zip(xs, ys))

            box = boxes[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box

            # Centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Bottom midpoint
            bx = (x1 + x2) // 2
            by = y2

            # Midpoint between centroid and bottom midpoint
            mx = (cx + bx) // 2
            my = (cy + by) // 2

            # Save for later drawing
            box_centroids.append({
                "centroid": (cx, cy),
                "bottom": (bx, by),
                "midpoint": (mx, my)
            })

            # Mask overlay (green tint)
            colored_mask = cv2.merge([mask // 2, mask, mask // 2])
            overlay[idx] = cv2.addWeighted(overlay, 0.5, colored_mask, 0.5, 0)[idx]

    return overlay, vehicle_pixel_coords, box_centroids

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

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    visible_width = int(frame_width / zoom_level)
    visible_height = int(frame_height / zoom_level)
    visible_frame = frame[int(offset_y):int(offset_y + visible_height), int(offset_x):int(offset_x + visible_width)]

    # Get overlay frame with mask visualization and mask pixel coords
    overlay_frame, vehicle_pixel_coords, box_centroids = get_vehicle_mask_overlay_with_visual(visible_frame)

    # Resize overlay frame to display size
    display_frame = cv2.resize(overlay_frame, (frame_width, frame_height))

    grid_copy = new_grid()

    current_time = time.time()
    if is_mouse_down and (current_time - last_click_time >= CLICK_INTERVAL):
        adjusted_x = round((cursor_position[0] / zoom_level) + offset_x)
        adjusted_y = round((cursor_position[1] / zoom_level) + offset_y)
        clicked_points.append((adjusted_x, adjusted_y))
        last_click_time = current_time

    # Draw clicked points on Camera View and World Grid
    for pt in clicked_points:
        screen_x = int((pt[0] - offset_x) * zoom_level)
        screen_y = int((pt[1] - offset_y) * zoom_level)
        cv2.circle(display_frame, (screen_x, screen_y), 4, (0, 0, 255), -1)
        wx, wy = pixel_to_world(pt[0], pt[1])
        gx = int(wx * grid_scale)
        gy = grid_scale - int(wy * grid_scale)
        if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
            cv2.circle(grid_copy, (gx, gy), 5, (0, 0, 255), -1)

    # Draw vehicle segmentation points on World Grid
    for px, py in vehicle_pixel_coords:
        wx, wy = pixel_to_world(px, py)
        gx = int(wx * grid_scale)
        gy = grid_scale - int(wy * grid_scale)
        if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
            cv2.circle(grid_copy, (gx, gy), 1, (0, 255, 0), -1)
    
    # Draw centroids and midpoints on both Camera View and World Grid
    for pt in box_centroids:
        for color, key in [((0, 0, 255), "centroid"), ((0, 128, 255), "bottom"), ((255, 0, 0), "midpoint")]:
            px, py = pt[key]

            # Camera View (adjust for zoom and offset)
            screen_x = int((px - offset_x) * zoom_level)
            screen_y = int((py - offset_y) * zoom_level)
            if 0 <= screen_x < frame_width and 0 <= screen_y < frame_height:
                cv2.circle(display_frame, (screen_x, screen_y), 4, color, -1)

            # World Grid
            wx, wy = pixel_to_world(px, py)
            gx = int(wx * grid_scale)
            gy = grid_scale - int(wy * grid_scale)
            if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
                cv2.circle(grid_copy, (gx, gy), 4, color, -1)

    # Draw hover cursor
    x, y = cursor_position
    adj_x = int((x / zoom_level) + offset_x)
    adj_y = int((y / zoom_level) + offset_y)
    if 0 <= adj_x < frame_width and 0 <= adj_y < frame_height:
        cv2.circle(display_frame, (x, y), 4, (255, 0, 0), -1)
        wx, wy = pixel_to_world(adj_x, adj_y)
        gx = int(wx * grid_scale)
        gy = grid_scale - int(wy * grid_scale)
        if 0 <= gx < grid_scale and 0 <= gy < grid_scale:
            cv2.circle(grid_copy, (gx, gy), 5, (255, 0, 0), -1)

    cv2.imshow("Camera View", display_frame)
    cv2.imshow("World Grid", grid_copy)

    if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
