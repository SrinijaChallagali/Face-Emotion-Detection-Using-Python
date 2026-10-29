"""Real-time facial emotion detection from a webcam feed.

Detects faces with an OpenCV Haar cascade and classifies each face into one of
seven emotions with a pre-trained CNN (model.h5). Press 'q' to quit.
"""
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
INPUT_SIZE = 48  # the model expects 48x48 grayscale face images


def main():
    model = load_model(MODEL_PATH, compile=False)  # inference only, no training setup needed
    face_classifier = cv2.CascadeClassifier(CASCADE_PATH)
    if face_classifier.empty():
        raise RuntimeError(f"Could not load face detector from {CASCADE_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            cv2.putText(frame, "No Faces", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Crop the face, resize to the model's input size and normalize to 0-1
            roi = cv2.resize(gray[y:y + h, x:x + w], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
            roi = img_to_array(roi.astype("float32") / 255.0)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi, verbose=0)[0]
            label = f"{EMOTION_LABELS[prediction.argmax()]} ({prediction.max():.0%})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
