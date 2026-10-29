# Face Emotion Detection using Python

Real-time facial emotion recognition from a webcam feed. OpenCV detects faces in each frame, and a convolutional neural network (CNN) built with TensorFlow/Keras classifies each face into one of seven emotions:

**Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise**

## How it works

1. **Face detection:** each webcam frame is converted to grayscale and faces are located with an OpenCV Haar cascade.
2. **Preprocessing:** each face is cropped, resized to 48x48 pixels and normalized to the 0–1 range.
3. **Classification:** the CNN predicts the most likely emotion, which is drawn on the frame with its confidence.

## Model training

`train_emotion_cnn.ipynb` trains the CNN on the [Face Expression Recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) (48x48 grayscale faces, 7 classes):

| | |
|---|---|
| Training images | 28,821 |
| Validation images | 7,066 |
| Architecture | 4 convolutional blocks (64, 128, 512, 512 filters) with batch normalization, max pooling and dropout, followed by two dense layers |
| Optimizer | Adam, with early stopping and learning-rate reduction |
| Best validation accuracy | ~62% |

For reference, emotion recognition on 48x48 faces is a hard problem; human accuracy on similar datasets is around 65%.

The real-time demo uses the pre-trained `model.h5` included in this repository.

## Getting started

```bash
pip install -r requirements.txt
python detect_emotions.py
```

A webcam window opens with each detected face labelled. Press **q** to quit.

## Project structure

```
detect_emotions.py                   Real-time webcam emotion detection
train_emotion_cnn.ipynb              CNN training and evaluation notebook
model.h5                             Pre-trained emotion classification model
haarcascade_frontalface_default.xml  OpenCV face detector
requirements.txt                     Python dependencies
```

## Tech stack

Python, TensorFlow/Keras, OpenCV, NumPy
