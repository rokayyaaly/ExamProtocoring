import cvzone  # For drawing  rectangles around detected faces
from ultralytics import YOLO
import cv2     # OpenCV for webcam capture and image processing

# Use webcam (0 = default camera)
cap = cv2.VideoCapture(0)
#persist=True in tracking keeps memory of faces across frames
facemodel = YOLO('yolov8n-face.pt')   

while cap.isOpened():
    rt, frame = cap.read()   
    if not rt:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (700, 500))

    # Use tracking instead of predict
    face_result = facemodel.track(frame, conf=0.5, persist=True)
    face_count = 0

    # Iterate over detection results
    for result in face_result:
        boxes = result.boxes
        for box in boxes:
            face_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Draw box
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    # Alert if multiple faces
    if face_count > 1:
        cv2.putText(frame, "WARNING: Multiple Faces Detected!", 
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 255), 3)
    # Show the frame in a window
    cv2.imshow("Webcam", frame)
    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()