import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
import math
import tkinter as tk

# --- PART 1: TKINTER OVERLAY SETUP ---
class GazeOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        # Transparent full-screen window
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparentcolor", "grey")
        self.root.configure(bg='grey')

        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, 
                                bg='grey', highlightthickness=0)
        self.canvas.pack()
        
        # The Red Dot
        self.dot_size = 20
        self.dot = self.canvas.create_oval(-50, -50, -30, -30, fill='red', outline='red')

    def move_dot(self, x, y):
        r = self.dot_size // 2
        self.canvas.coords(self.dot, x-r, y-r, x+r, y+r)

    def update(self):
        self.root.update()

overlay = GazeOverlay()

# --- PART 2: CAMERA & MEDIAPIPE ---
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- CONFIGURATION ---
# 1. Direction Control (Change these if direction is still wrong)
FLIP_X = True  # If True, Looking Left moves dot Left. If False, it moves Right.
FLIP_Y = False # Usually False (Screen Y=0 is Top)

# 2. Calibration (Sensitivity)
# Adjust these numbers to match your eye range
HOR_MIN, HOR_MAX = 0.35, 0.65 
VER_MIN, VER_MAX = 0.30, 0.50  
SMOOTHING = 0.15 

# Smoothing variables
current_x, current_y = overlay.screen_w / 2, overlay.screen_h / 2

# Landmark IDs
LEFT_IRIS = 468
LEFT_KEY = [159, 145, 133, 33] # Top, Bottom, Left, Right
RIGHT_IRIS = 473
RIGHT_KEY = [386, 374, 362, 263]

def process_eye(landmarks, iris_id, key_points, frame_w, frame_h):
    def to_coords(id):
        return int(landmarks[id].x * frame_w), int(landmarks[id].y * frame_h)
    
    p_iris = to_coords(iris_id)
    p_top = to_coords(key_points[0])
    p_bottom = to_coords(key_points[1])
    p_left = to_coords(key_points[2])
    p_right = to_coords(key_points[3])

    # Blink Ratio
    v_len = math.hypot(p_top[0]-p_bottom[0], p_top[1]-p_bottom[1])
    h_len = math.hypot(p_left[0]-p_right[0], p_left[1]-p_right[1])
    blink_ratio = v_len / h_len if h_len != 0 else 1.0

    # Gaze Ratios
    # Horizontal: Distance from Left Corner
    dist_h = p_iris[0] - p_left[0]
    total_w = p_right[0] - p_left[0]
    gaze_h = dist_h / total_w if total_w != 0 else 0.5

    # Vertical: Distance from Top Lid
    dist_v = p_iris[1] - p_top[1]
    total_h = p_bottom[1] - p_top[1]
    gaze_v = dist_v / total_h if total_h != 0 else 0.5

    return blink_ratio, gaze_h, gaze_v

blink_counter = 0
blink_counter_frame = 0
BLINK_THRESHOLD = 0.35

while True:
    ret, frame = cam.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Mirror the camera
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    output = face_mesh.process(rgb_frame)
    if output.multi_face_landmarks:
        lm = output.multi_face_landmarks[0].landmark

        l_blink, l_h, l_v = process_eye(lm, LEFT_IRIS, LEFT_KEY, w, h)
        r_blink, r_h, r_v = process_eye(lm, RIGHT_IRIS, RIGHT_KEY, w, h)

        # 1. Blink Logic
        if l_blink < BLINK_THRESHOLD and r_blink < BLINK_THRESHOLD:
            if blink_counter_frame == 0:
                blink_counter += 1
                blink_counter_frame = 1
                print(f"Blink! {blink_counter}")
        else:
            if blink_counter_frame > 0:
                blink_counter_frame += 1
                if blink_counter_frame > 5: blink_counter_frame = 0

        # 2. Gaze Logic
        avg_h = (l_h + r_h) / 2
        avg_v = (l_v + r_v) / 2

        # 3. Mapping to Screen (WITH FLIP LOGIC)
        # X-Axis Flip Check
        if FLIP_X:
            # Map [Min, Max] -> [Width, 0] (Inverted)
            target_x = np.interp(avg_h, [HOR_MIN, HOR_MAX], [overlay.screen_w, 0])
        else:
            # Map [Min, Max] -> [0, Width] (Standard)
            target_x = np.interp(avg_h, [HOR_MIN, HOR_MAX], [0, overlay.screen_w])

        # Y-Axis Flip Check
        if FLIP_Y:
            target_y = np.interp(avg_v, [VER_MIN, VER_MAX], [overlay.screen_h, 0])
        else:
            target_y = np.interp(avg_v, [VER_MIN, VER_MAX], [0, overlay.screen_h])

        # Smoothing
        current_x += (target_x - current_x) * SMOOTHING
        current_y += (target_y - current_y) * SMOOTHING

        # Update Dot
        overlay.move_dot(current_x, current_y)

        # Draw Debug info
        cv2.putText(frame, f"Blinks: {blink_counter}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        cv2.putText(frame, f"H: {avg_h:.2f} V: {avg_v:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    overlay.update()
    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
overlay.root.destroy()