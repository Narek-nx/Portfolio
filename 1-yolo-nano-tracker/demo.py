
import cv2

# Paths to the ONNX models
backbone_path = '/home/jetson/Downloads/nanotrack_backbone_sim.onnx'
neckhead_path = '/home/jetson/Downloads/nanotrack_head_sim.onnx'


#backbone_path = '/home/jetson/Desktop/nano_tracker/SiamTrackers/NanoTrack/models/nanotrackv3/nanotrack_backbone.onnx'
#neckhead_path = '/home/jetson/Desktop/nano_tracker/SiamTrackers/NanoTrack/models/nanotrackv3/nanotrack_head.onnx'
# Load TrackerNano with ONNX models
params = cv2.TrackerNano_Params()
params.backbone = backbone_path
params.neckhead = neckhead_path
tracker = cv2.TrackerNano_create(params)

# Capture video
cap = cv2.VideoCapture("/home/jetson/Downloads/edit.mp4")
ret, frame = cap.read()
if not ret:
    print("Failed to read frame.")
    exit()

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up the VideoWriter to save the output video
output_path = "/home/jetson/Desktop/output_video9.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec to 'MJPG' or others as needed
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Select region to track
bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# Initialize tracker
tracker.init(frame, bbox)

# Start tracking
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update tracker
    success, bbox = tracker.update(frame)
    
    if success:
        # Draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Show the frame
    cv2.imshow("TrackerNano", frame)
    
    # Break on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

