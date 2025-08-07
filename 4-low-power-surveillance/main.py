import cv2
import yaml
from ultralytics import YOLO
from utils.alert import AlertManager
from utils.draw import draw_boxes

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

video_path = config["video_path"]
model_path = config["model_path"]
alert_classes = config["alert_classes"]
alert_once_per = config["alert_once_per"]

# Load model and names
model = YOLO(model_path)
names = model.names

# Alert manager
alert_manager = AlertManager(alert_classes, alert_once_per)

# Video input
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        cls_name = names[cls_id]

        # Alert check
        if alert_manager.should_alert(cls_name, frame_count):
            alert_manager.trigger_alert(cls_name)

        detections.append((x1, y1, x2, y2, conf, cls_id))

    # Draw boxes
    frame = draw_boxes(frame, detections, names)

    # Show frame
    cv2.imshow("Smart Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
