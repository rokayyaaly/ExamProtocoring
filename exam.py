import cvzone
from ultralytics import YOLO
import cv2

# Load models
face_model = YOLO("yolov8n-face.pt")   # Face detection model
object_model = YOLO("yolov8n.pt")      # Object detection model

# COCO classes we want
# 63 = laptop, 67 = cell phone, 73 = book
allowed_classes = [63, 67, 73]

# Start webcam
cap = cv2.VideoCapture(0)

print("Press Q or ESC to exit")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (700, 500))

    # =========================
    # FACE DETECTION
    # =========================
    face_results = face_model.track(frame, conf=0.5, persist=True)

    face_count = 0

    for result in face_results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:

                face_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Draw face box
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=3)

    # Warning if more than one face
    if face_count > 1:
        cv2.putText(
            frame,
            "WARNING: Multiple Faces Detected!",
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            3
        )

    # =========================
    # OBJECT DETECTION
    # =========================
    results = object_model(frame, classes=allowed_classes)

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Draw object box
                cvzone.cornerRect(frame, (x1, y1, w, h), l=6, rt=2)

                # Confidence
                conf = float(box.conf[0])

                # Class id
                cls = int(box.cls[0])

                names = object_model.names
                label = f"{names[cls]} {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Show frame
    cv2.imshow("Detection System", frame)

    # Exit keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or ESC
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)