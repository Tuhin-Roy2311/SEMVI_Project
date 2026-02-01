import os
import time
import threading
import math
import numpy as np
import cv2
import mediapipe as mp
from collections import deque

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '2'

from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- MEDIAPIPE SETUP ---
# We only need one FaceMesh for EVERYTHING (Gaze, Blink, Emotion)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- LANDMARK CONSTANTS ---
# Eyes (for Gaze/Blink)
LEFT_IRIS = 468
LEFT_KEY = [159, 145, 133, 33]
RIGHT_IRIS = 473
RIGHT_KEY = [386, 374, 362, 263]

# Mouth (for Emotion)
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Calibration
HOR_MIN, HOR_MAX = 0.40, 0.60
VER_MIN, VER_MAX = 0.30, 0.45
SMOOTHING = 0.2
BLINK_THRESHOLD = 0.45

# History for smoothing
x_history = deque(maxlen=5)
y_history = deque(maxlen=5)

# State
blink_counter = 0
blink_counter_frame = 0

# --- HELPER 1: EYE PROCESSING ---
def process_eye(landmarks, iris_id, key_points):
    def get_pt(idx): return landmarks[idx].x, landmarks[idx].y
    p_iris = get_pt(iris_id)
    p_top, p_bottom = get_pt(key_points[0]), get_pt(key_points[1])
    p_left, p_right = get_pt(key_points[2]), get_pt(key_points[3])

    v_len = math.hypot(p_top[0]-p_bottom[0], p_top[1]-p_bottom[1])
    h_len = math.hypot(p_left[0]-p_right[0], p_left[1]-p_right[1])
    blink_ratio = v_len / h_len if h_len != 0 else 1.0

    dist_h = p_iris[0] - p_left[0]
    total_w = p_right[0] - p_left[0]
    gaze_h = dist_h / total_w if total_w != 0 else 0.5

    dist_v = p_iris[1] - p_top[1]
    total_h = p_bottom[1] - p_top[1]
    gaze_v = dist_v / total_h if total_h != 0 else 0.5

    return blink_ratio, gaze_h, gaze_v

# --- HELPER 2: EMOTION LOGIC (GEOMETRIC) ---
def detect_emotion(landmarks):
    """
    Uses simple geometry to detect basic emotions.
    """
    def get_pt(idx): return np.array([landmarks[idx].x, landmarks[idx].y])

    # Get points
    top_lip = get_pt(UPPER_LIP)
    bottom_lip = get_pt(LOWER_LIP)
    left_corner = get_pt(LEFT_MOUTH_CORNER)
    right_corner = get_pt(RIGHT_MOUTH_CORNER)

    # 1. Calculate Mouth Openness (Vertical)
    mouth_open = np.linalg.norm(top_lip - bottom_lip)
    
    # 2. Calculate Mouth Width (Horizontal)
    mouth_width = np.linalg.norm(left_corner - right_corner)

    # 3. Calculate Smile Curve
    # Check if corners are higher than the center of the lip (inverted Y in computer vision)
    # Note: In screen coords, Y increases downwards. So "higher" means LOWER Y value.
    avg_corner_y = (left_corner[1] + right_corner[1]) / 2
    center_y = (top_lip[1] + bottom_lip[1]) / 2
    
    # Heuristics (You can tune these numbers!)
    # If mouth is very open -> Surprise
    if mouth_open > 0.05: 
        return "Surprise"
    
    # If corners are significantly above center -> Happy
    # (Checking if corners are "higher" on face, which is lower Y value)
    if (center_y - avg_corner_y) > 0.01:
        return "Happy"
    
    # If mouth is wide but not open -> Grin/Happy
    if mouth_width > 0.25: # Tunable threshold
        return "Happy"

    return "Neutral"

# --- MAIN TRACKER LOOP ---
def tracker_loop():
    global blink_counter, blink_counter_frame
    
    print(" [TRACKER] Starting Camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1) 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = face_mesh.process(rgb_frame)
            
            if output.multi_face_landmarks:
                lm = output.multi_face_landmarks[0].landmark
                
                # 1. Process Eyes (Gaze & Blink)
                l_blink, l_h, l_v = process_eye(lm, LEFT_IRIS, LEFT_KEY)
                r_blink, r_h, r_v = process_eye(lm, RIGHT_IRIS, RIGHT_KEY)

                if l_blink < BLINK_THRESHOLD and r_blink < BLINK_THRESHOLD:
                    if blink_counter_frame == 0:
                        blink_counter += 1
                        blink_counter_frame = 1
                        socketio.emit('blink_update', {'count': blink_counter})
                else:
                    if blink_counter_frame > 0:
                        blink_counter_frame += 1
                        if blink_counter_frame > 5: blink_counter_frame = 0

                # Gaze Smoothing
                avg_h = (l_h + r_h) / 2
                avg_v = (l_v + r_v) / 2
                raw_x = np.interp(avg_h, [HOR_MIN, HOR_MAX], [1.0, 0.0])
                raw_y = np.interp(avg_v, [VER_MIN, VER_MAX], [0.0, 1.0])
                x_history.append(raw_x)
                y_history.append(raw_y)
                smooth_x = sum(x_history) / len(x_history)
                smooth_y = sum(y_history) / len(y_history)

                socketio.emit('gaze_update', {'x': smooth_x, 'y': smooth_y})

                # 2. Process Emotion (Instant Geometry Check)
                emotion = detect_emotion(lm)
                socketio.emit('emotion_update', {'emotion': emotion})

            time.sleep(0.01)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('combined_index.html')

if __name__ == '__main__':
    # Only ONE thread needed now!
    t = threading.Thread(target=tracker_loop, daemon=True)
    t.start()
    
    print(" [SYSTEM] Fast MediaPipe Server running at http://127.0.0.1:5000")
    socketio.run(app, debug=True, use_reloader=False)