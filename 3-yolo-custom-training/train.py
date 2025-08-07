from ultralytics import YOLO

# Load a model (use 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="custom.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="/path/to/save/model",
    name="custom_yolov8"
)
