from flask import Flask, render_template, jsonify
import os
from backend_emotion import start_detection, stop_detection, get_emotion

app = Flask(__name__)

IMAGE_DIR = "static/images"

images = []
image_emotions = {}

def get_image_emotion(filename):
    name = filename.lower()
    if name.startswith("su"):
        return "surprise"
    if name.startswith("a"):
        return "angry"
    if name.startswith("h"):
        return "happy"
    if name.startswith("s"):
        return "sad"
    if name.startswith("d"):
        return "disgust"
    if name.startswith("f"):
        return "fear"
    if name.startswith("n"):
        return "neutral"
    return "unknown"

for img in sorted(os.listdir(IMAGE_DIR)):
    if img.endswith((".jpg", ".png", ".jpeg")):
        images.append(img)
        image_emotions[img] = get_image_emotion(img)

@app.route("/")
def index():
    return render_template(
        "index.html",
        images=images,
        image_emotions=image_emotions
    )

@app.route("/start")
def start():
    start_detection()
    return jsonify({"status": "started"})

@app.route("/stop")
def stop():
    stop_detection()
    return jsonify({"status": "stopped"})

@app.route("/emotion")
def emotion():
    viewer_emotion, confidence = get_emotion()
    return jsonify({
        "emotion": viewer_emotion,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)