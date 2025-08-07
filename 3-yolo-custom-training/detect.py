from ultralytics import YOLO

# Load trained model
model = YOLO("runs/train/custom_yolov8/weights/best.pt")

# Run detection
results = model("test.jpg", show=True, conf=0.25)
