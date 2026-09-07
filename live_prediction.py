import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("model/emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("model/emotiondetector.h5")
print("Emotion model loaded successfully.")

YUNET_MODEL = "model/face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000)
print("YuNet face detector loaded successfully.")

labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

def extract_features(image):
    feature = np.array(image)
    feature = cv2.resize(feature, (48, 48))
    if len(feature.shape) == 3:
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
    feature = clahe.apply(feature)
    feature = feature.astype("float32") / 255.0
    feature = feature.reshape(1, 48, 48, 1)
    return feature

webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Could not read webcam frame.")
        break

    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0:
                continue

            face_region = frame[y:y + h, x:x + w]
            if face_region.size == 0:
                continue

            img = extract_features(face_region)
            pred = model.predict(img, verbose=0)
            prediction_index = pred.argmax()
            prediction_label = labels[prediction_index]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

    cv2.imshow("Facial Emotion Recognition - YuNet", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
print("Webcam prediction stopped.")