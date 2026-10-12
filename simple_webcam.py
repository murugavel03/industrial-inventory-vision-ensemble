"""
simple_webcam.py — Minimal YOLO webcam test. Press Q to quit.
"""
from ultralytics import YOLO
import cv2

print("Loading model...")
model = YOLO("runs/detect/runs/train/irregular_specialized_v1/weights/best.pt")
print("Model loaded!")

print("Opening camera 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: No camera found!")
    exit()

print("Camera opened! Starting detection... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame error")
        break

    results = model(frame, conf=0.25, verbose=False)
    frame = results[0].plot()

    cv2.imshow("YOLO11 - Model 1 (press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
