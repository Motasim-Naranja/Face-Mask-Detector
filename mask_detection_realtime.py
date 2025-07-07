# mask_detection_realtime.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("mask_detection_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face region
        face = frame[y:y+h, x:x+w]
        # Resize to match training size
        face_resized = cv2.resize(face, (100, 100))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)

        # Predict
        # prediction = model.predict(face_input)[0][0]
        # label = "With Mask" if prediction < 0.5 else "Without Mask"
        # color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
        prediction = model.predict(face_input)[0][0]
        label = "Without Mask" if prediction < 0.5 else "With Mask"
        color = (0, 0, 255) if label == "Without Mask" else (0, 255, 0)

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show result
    cv2.imshow("Mask Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
