import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

# COCO class IDs for vehicles
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car=2, motorcycle=3, bus=5, truck=7

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_frame(frame):
    # Convert BGR to RGB and to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    masks = pred['masks']  # [N, 1, H, W]
    labels = pred['labels']
    scores = pred['scores']

    # Threshold for confidence
    threshold = 0.6

    # Prepare overlay mask
    overlay = frame.copy()

    for i in range(len(labels)):
        if scores[i] >= threshold and labels[i].item() in VEHICLE_CLASS_IDS:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            colored_mask = cv2.merge([mask // 2, mask, mask // 2])  # green tint

            idx = mask > 128
            overlay[idx] = cv2.addWeighted(frame, 0.5, colored_mask, 0.5, 0)[idx]

            # Draw bounding box
            box = pred['boxes'][i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label text
            cv2.putText(overlay, f"Vehicle {scores[i]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Centroid of bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)  # red dot

            # Midpoint of bottom edge
            bx = (x1 + x2) // 2
            by = y2
            cv2.circle(overlay, (bx, by), 5, (0, 255, 255), -1)  # yellow dot

            # Midpoint between centroid and bottom midpoint
            mx = (cx + bx) // 2
            my = (cy + by) // 2
            cv2.circle(overlay, (mx, my), 5, (255, 0, 0), -1)  # blue dot

    return overlay

window_name = "Vehicle Segmentation"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 2560, 1440)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = process_frame(frame)
        cv2.imshow(window_name, output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "D:/Jetson Project Sample Videos/8000.avi"  # <- Replace with your video path
    main(video_path)