import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
import math

# 1. Setup Camera and MediaPipe
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Enables Iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Variables for Blinking
blink_counter = 0
blink_counter_frame = 0  
BLINK_THRESHOLD = 0.35   # Lower = harder to detect blink, Higher = easier

# 3. Landmark Indices (Constants)
# Left Eye
LEFT_IRIS = 468
LEFT_KEY_POINTS = [33, 133, 159, 145] # [Inner, Outer, Top, Bottom]

# Right Eye
RIGHT_IRIS = 473
RIGHT_KEY_POINTS = [362, 263, 386, 374] # [Inner, Outer, Top, Bottom]

# Helper function to calculate distance
def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Helper to process a single eye
def get_eye_ratios(landmarks, iris_id, key_points, frame_w, frame_h):
    # Extract coordinates
    def to_coords(id):
        return int(landmarks[id].x * frame_w), int(landmarks[id].y * frame_h)

    p_iris = to_coords(iris_id)
    p_inner = to_coords(key_points[0]) # Corner near nose
    p_outer = to_coords(key_points[1]) # Corner near ear
    p_top = to_coords(key_points[2])
    p_bottom = to_coords(key_points[3])

    # 1. BLINK RATIO (Vertical / Horizontal)
    v_len = calculate_distance(p_top, p_bottom)
    h_len = calculate_distance(p_inner, p_outer)
    blink_ratio = v_len / h_len if h_len != 0 else 1.0

    # 2. GAZE RATIO (Horizontal)
    # Distance from iris to corners
    dist_inner = abs(p_inner[0] - p_iris[0])
    dist_outer = abs(p_outer[0] - p_iris[0])
    total_w = dist_inner + dist_outer
    # Ratio: 0 = looking at inner corner, 1 = looking at outer corner
    gaze_ratio_h = dist_inner / total_w if total_w != 0 else 0.5

    # 3. GAZE RATIO (Vertical)
    dist_top = abs(p_top[1] - p_iris[1])
    dist_bottom = abs(p_bottom[1] - p_iris[1])
    total_h = dist_top + dist_bottom
    gaze_ratio_v = dist_top / total_h if total_h != 0 else 0.5

    # Return data and points for drawing
    return blink_ratio, gaze_ratio_h, gaze_ratio_v, (p_iris, p_inner, p_outer, p_top, p_bottom)

while True:
    ret, frame = cam.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape
    
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # --- PROCESS LEFT EYE ---
        l_blink, l_gaze_h, l_gaze_v, l_points = get_eye_ratios(
            landmarks, LEFT_IRIS, LEFT_KEY_POINTS, frame_w, frame_h
        )
        
        # --- PROCESS RIGHT EYE ---
        r_blink, r_gaze_h, r_gaze_v, r_points = get_eye_ratios(
            landmarks, RIGHT_IRIS, RIGHT_KEY_POINTS, frame_w, frame_h
        )

        # --- DRAW VISUALS ---
        # Draw Left Eye
        cv2.circle(frame, l_points[0], 3, (0, 255, 0), -1) # Iris
        cv2.line(frame, l_points[3], l_points[4], (0, 200, 0), 1) # Blink Line
        # Draw Right Eye
        cv2.circle(frame, r_points[0], 3, (0, 255, 0), -1) # Iris
        cv2.line(frame, r_points[3], r_points[4], (0, 200, 0), 1) # Blink Line


        # --- LOGIC 1: BLINK DETECTION (BOTH EYES) ---
        # We require BOTH eyes to be below threshold to count as a blink
        if l_blink < BLINK_THRESHOLD and r_blink < BLINK_THRESHOLD:
            if blink_counter_frame == 0:
                blink_counter += 1
                blink_counter_frame = 1 # Lock
                print(f"BLINK DETECTED! Total: {blink_counter}")
        else:
            if blink_counter_frame > 0:
                blink_counter_frame += 1
                if blink_counter_frame > 5: # Debounce for 5 frames
                    blink_counter_frame = 0


        # --- LOGIC 2: GAZE DETECTION (AVERAGE) ---
        # Note: Inner/Outer logic is mirrored for left/right eyes in context of "Left/Right" direction
        # To simplify: We just need to know if Iris is to the LEFT or RIGHT of the screen.
        
        # Average the ratios to get a stable "Face Gaze"
        avg_gaze_h = (l_gaze_h + r_gaze_h) / 2
        avg_gaze_v = (l_gaze_v + r_gaze_v) / 2
        
        # Determine Horizontal Direction
        hor_text = "CENTER"
        if avg_gaze_h < 0.45: hor_text = "RIGHT" # Mirrored: Looking "Right" moves iris closer to inner corner of Left eye
        elif avg_gaze_h > 0.55: hor_text = "LEFT"

        # Determine Vertical Direction
        ver_text = "CENTER"
        if avg_gaze_v < 0.35: ver_text = "UP"
        elif avg_gaze_v > 0.65: ver_text = "DOWN"

        # Display Info
        cv2.putText(frame, f"Gaze: {ver_text} - {hor_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_counter}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()