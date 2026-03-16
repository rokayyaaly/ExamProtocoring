from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

allowed_classes = [63, 67, 73]  # laptop, cell phone, book

cap = cv2.VideoCapture(0)

print("Press q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=allowed_classes)
    annotated_frame = results[0].plot()

    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()