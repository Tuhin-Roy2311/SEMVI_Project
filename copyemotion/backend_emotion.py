from fer import FER
import cv2

detector = FER()
camera = None
running = False

def start_detection():
    global camera, running
    if not running:
        camera = cv2.VideoCapture(0)
        running = True

def stop_detection():
    global camera, running
    if camera:
        camera.release()
        camera = None
    running = False

def get_emotion():
    if not running or camera is None:
        return "waiting", 0

    ret, frame = camera.read()
    if not ret:
        return "waiting", 0

    result = detector.detect_emotions(frame)
    if not result:
        return "neutral", 50

    emotions = result[0]["emotions"]
    emotion = max(emotions, key=emotions.get)
    confidence = int(emotions[emotion] * 100)

    return emotion, confidence