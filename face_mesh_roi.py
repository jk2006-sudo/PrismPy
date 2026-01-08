import cv2
import numpy as np

# Load face detector
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")

            face = frame[y1:y2, x1:x2]

            # Estimate eye region (upper part of face)
            fh, fw = face.shape[:2]
            eye_y1 = int(fh * 0.2)
            eye_y2 = int(fh * 0.45)

            left_eye = face[eye_y1:eye_y2, int(fw * 0.1):int(fw * 0.45)]
            right_eye = face[eye_y1:eye_y2, int(fw * 0.55):int(fw * 0.9)]

            # Draw boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame,
                          (x1 + int(fw*0.1), y1 + eye_y1),
                          (x1 + int(fw*0.45), y1 + eye_y2),
                          (255, 0, 0), 2)

            cv2.rectangle(frame,
                          (x1 + int(fw*0.55), y1 + eye_y1),
                          (x1 + int(fw*0.9), y1 + eye_y2),
                          (255, 0, 0), 2)

            break

    cv2.imshow("Face + Eye ROI (Python 3.13)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

