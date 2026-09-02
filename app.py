import os
import tempfile
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from utils import predict_emotion

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
register_heif_opener()

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "face_detection_yunet_2023mar.onnx")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("YuNet model not found")

face_detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), 0.5, 0.3, 5000)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(),"face_emotion_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "heic", "heif", "bmp", "tiff", "tif"}

def allowed_file(filename):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def load_image(file):
    try:
        file.seek(0)
        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    except (UnidentifiedImageError, OSError, ValueError):
        return None

def detect_faces(image):
    height, width = image.shape[:2]
    new_width = 1000
    scale = new_width / width
    new_height = int(height * scale)
    resized = cv2.resize(image,(new_width, new_height))
    face_detector.setInputSize((new_width, new_height))

    _, faces = face_detector.detect(resized)
    if faces is None:
        return []
    results = []

    for face in faces:
        x = face[0] / scale
        y = face[1] / scale
        w = face[2] / scale
        h = face[3] / scale
        confidence = face[-1]
        results.append([x, y, w, h, confidence])
    return results

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/uploaded/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],filename)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("upload.html",error="Please select an image.")
    
    file = request.files["image"]
    if file.filename == "":
        return render_template("upload.html",error="Please select an image.")

    if not allowed_file(file.filename):
        return render_template("upload.html", error="Unsupported image format.")
    
    image = load_image(file)
    if image is None:
        return render_template("upload.html",error="Unable to read image.")

    faces = detect_faces(image)
    emotion = "No Face Found"
    confidence = 0

    for face in faces:
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])
        face_confidence = float(face[4])
        if face_confidence < 0.60:
            continue

        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        if w <= 0 or h <= 0:
            continue

        face_image = image[y:y + h,x:x + w]
        if face_image.size == 0:
            continue

        gray_face = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
        emotion, confidence = predict_emotion(gray_face)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),3)

    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "result.jpg")
    cv2.imwrite(output_path, image)
    return render_template("upload.html",emotion=emotion,confidence=confidence,image="result.jpg")

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    if "image" not in request.files:
        return jsonify({"error": "No image received."}), 400

    file = request.files["image"]
    image = load_image(file)
    if image is None:
        return jsonify({"error": "Unable to read image."}), 400

    faces = detect_faces(image)
    results = []
    for face in faces:
        x = int(face[0])
        y = int(face[1])
        w = int(face[2])
        h = int(face[3])

        face_confidence = float(face[4])
        if face_confidence < 0.50:
            continue

        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        if w <= 0 or h <= 0:
            continue

        face_image = image[y:y + h,x:x + w]
        if face_image.size == 0:
            continue

        gray_face = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
        emotion, confidence = predict_emotion(gray_face)
        results.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "emotion": emotion,
            "confidence": confidence,
            "face_confidence": round(face_confidence * 100,2)
        })

    return jsonify({"faces": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0",port=port,debug=False)