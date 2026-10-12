from ultralytics import YOLO
import cv2

# 1. Load your custom model
model = YOLO(r"C:\5_model\runs\detect\runs\train\small_parts_v1\weights\best.pt")

# 2. Initialize the camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

# 3. Create the display window upfront (required on some Windows setups)
cv2.namedWindow("Custom Model - Live Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Custom Model - Live Detection", 800, 600)

print("✅ Camera open. Press 'q' to quit.")

# 4. Enter the live processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame could not be read.")
        break

    # 5. Run inference (stream=False returns a list directly, not a generator)
    results = model.predict(frame, conf=0.5, verbose=False)

    # 6. Annotate and display the frame
    annotated_frame = results[0].plot()
    cv2.imshow("Custom Model - Live Detection", annotated_frame)

    # 7. waitKey MUST be called every loop iteration to paint the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quit signal received.")
        break

cap.release()
cv2.destroyAllWindows()