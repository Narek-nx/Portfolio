import cv2
from ultralytics import YOLO

# Paths to the ONNX models
backbone_path = '../onnx/nanotrack_backbone_sim.onnx'
neckhead_path = '../onnx/nanotrack_head_sim.onnx'

# Load YOLO model
model = YOLO("../weights/best.pt" , task="detect")
# Capture video
cap = cv2.VideoCapture("/home/jetson/Downloads/edit.mp4")

trackers = []
det_interval = 0
# Start tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if det_interval % 200 == 0:
        trackers = []  # Reset old trackers
        results = model.predict(frame, conf=0.5, imgsz=480, iou=0.5)
        detections = results[0].boxes.data 
        # Draw detections (green color)
        for det in detections:
            x1, y1, x2, y2, score = det[:5]  # Detections are in format [x1, y1, x2, y2, score]
            bbox = [x1, y1, x2 - x1, y2 - y1]
            color = (0, 255, 0)  # Green color for detections
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{score:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Load TrackerNano with ONNX models
            params = cv2.TrackerNano_Params()
            params.backbone = backbone_path
            params.neckhead = neckhead_path
            tracker = cv2.TrackerNano_create(params)
            tracker.init(frame, bbox)
            trackers.append(tracker)

    if len(trackers) > 0:
        for i, tracker in enumerate(trackers):
        success, new_bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in new_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
      
        else:
            cv2.putText(frame, "Tracking failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("TrackerNano", frame)
    det_interval += 1
    # Break on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
cv2.destroyAllWindows()

