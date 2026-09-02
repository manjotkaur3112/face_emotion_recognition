import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
from keras.models import load_model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "emotiondetector.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH,compile=False)

emotion_labels = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]


def predict_emotion(face):

    face = cv2.resize(face,(48, 48),interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    face = clahe.apply(face)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face,axis=-1)
    face = np.expand_dims(face,axis=0)

    prediction = model.predict(face,verbose=0)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    confidence = float(np.max(prediction) * 100)
    return emotion, round(confidence, 2)