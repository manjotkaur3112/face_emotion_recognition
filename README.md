# Facial Emotion Recognition

A Flask web app that detects faces in uploaded images and predicts emotions using a trained deep learning model.

## Features

- Upload image prediction
- Live camera-based emotion detection
- Face detection with YuNet
- Emotion recognition using a CNN model
- Works with standard image formats and HEIC/HEIF input

## Project Structure

- `app.py` — Flask application and routes
- `utils.py` — emotion model loading and prediction logic
- `model/` — trained model files and YuNet detector
- `templates/` — HTML pages
- `static/` — CSS, JavaScript, and generated upload output

## Requirements

- Python 3.11+
- TensorFlow
- OpenCV
- Flask
- Pillow
- `pillow-heif`

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open the app in a browser:

```text
http://localhost:8080
```

## Deployment

This project is configured for Render using Docker. It can also be deployed with Gunicorn in a standard Python host environment.

## Notes

- The app stores uploaded/generated files in the system temp directory during runtime.
- The models are loaded from the project `model` directory.
- The app expects your trained model files to already exist in the project before running.
