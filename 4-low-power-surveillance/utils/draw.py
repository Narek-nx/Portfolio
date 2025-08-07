import cv2

def draw_boxes(frame, detections, names):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        label = f"{names[int(cls_id)]} {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
