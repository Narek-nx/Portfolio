# 🚀 Real-Time Object Detection + Tracking with YOLOv8 and Nano Tracker on Jetson Nano

This project combines YOLOv8 object detection with OpenCV’s Nano Tracker to achieve fast, real-time object tracking on low-power devices like Jetson Nano. It’s designed for surveillance, robotics, and embedded AI systems.

![Tracking Demo](objectdetection.gif)

---

## 🎯 Objective

To track moving objects (people, vehicles, etc.) in real-time using lightweight models suitable for devices like Jetson Nano, raspberry pi5 without compromising speed.

---

## 🛠️ Tools & Frameworks

- YOLOv8n (Ultralytics)
- OpenCV (Nano Tracker)
- Python 3.8+
- Jetson Nano 4GB
- Raspberry Pi Camera or USB webcam

---

## ⚙️ How It Works

1. YOLOv8 detects objects in the frame
2. Each detection is passed to Nano Tracker
3. Tracker updates object position across frames
4. Bounding boxes and IDs are displayed live


