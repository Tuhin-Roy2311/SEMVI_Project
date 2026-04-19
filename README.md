# CopyEmotion: Quadrant Gaze and Emotion Matching

CopyEmotion is a Flask-based project that combines:

- Real-time eye gaze tracking (4 quadrants)
- Real-time facial emotion detection
- Emotion matching against target image labels
- Session analytics (blink count, attention plots)
- Optional AI-generated report text (Gemini via google-genai)

The application presents images in a 2x2 layout and tracks where the user is looking (UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT). It then compares the user emotion with the target emotion assigned to each image.

## Main Features

- 2x2 quadrant viewer with keyboard navigation
- Real-time gaze stream using Server-Sent Events (SSE)
- 5-step gaze calibration: CENTER, UP, DOWN, LEFT, RIGHT
- Blink detection and blink counter per session
- DeepFace-based emotion inference with smoothing
- Per-image session logs with target vs detected emotion
- Auto-generated gaze heatmap and scatter plot images
- RAG-style summary report using Gemini (with local fallback text if Gemini is unavailable)

## Tech Stack

- Backend: Flask
- Computer vision: OpenCV, MediaPipe Face Mesh
- Emotion detection: DeepFace
- Numerical and plotting: NumPy, SciPy, Matplotlib
- AI summary: google-genai (Gemini), python-dotenv
- Frontend: Single HTML template with CSS and vanilla JavaScript

## Project Structure

```text
perfect - Copy/
  app.py                      # Flask app, session flow, report generation, routes
  backend_emotion.py          # DeepFace emotion detector thread
  diagnostic.py               # Standalone gaze-signal diagnostic tool
  requirements.txt            # Python dependencies
  run.txt                     # Local run note
  eye_tracking/
    tracker.py                # QuadrantEyeTracker + calibration + blink logic
  RAG/
    Main.py                   # Gemini/fallback report generation from session JSON
  templates/
    gaze_viewer.html          # Full UI and frontend logic
  static/
    image_emotions.json       # Authoritative image -> target emotion map
    images/                   # Source images shown in the viewer
    results/                  # Generated session outputs (JSON + plots)
```

## How It Works

1. The app starts a background camera thread.
2. Frames are processed by `QuadrantEyeTracker` for gaze and blink detection.
3. Frames are also pushed to `EmotionDetector` (DeepFace) for facial emotion inference.
4. Frontend subscribes to `/gaze/stream` and polls `/emotion` and `/blinks`.
5. During the session, frontend tracks image dwell time and match state.
6. On report generation (`/api/stop_session`), backend creates:
   - Session interaction JSON
   - Heatmap PNG
   - Scatter PNG
   - RAG analysis text (Gemini or fallback)
## Setup

### 1) Prerequisites

- Python 3.12 recommended
- Webcam
- Windows PowerShell (for commands below)

### 2) Create and activate virtual environment

Using a short path is recommended on Windows to avoid DeepFace/TensorFlow path issues.

```powershell
py -3.12 -m venv C:\v312
& C:\v312\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install dependencies

From the project root:

```powershell
pip install -r requirements.txt
```

Optional but recommended for best compatibility on Windows (Python 3.12):

```powershell
pip install deepface==0.0.99 mediapipe==0.10.13 tensorflow==2.19.1 tf-keras==2.19.0 protobuf==4.25.9 numpy==2.1.3
```

### 4) Optional: Gemini API key for RAG answer generation

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

If this key is missing, the app still runs and uses a local fallback summary.

## Run the App

From project root:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## In-App Controls

- Start Session button: enter viewer and start camera session
- C: start calibration
- Left/Right arrows: navigate image pages (4 images per page)
- R: generate report
- ESC: exit viewer

## Configure Target Emotions

Target emotions are read from `static/image_emotions.json`.

Example:

```json
{
  "image1.png": "happy",
  "image2.png": "angry",
  "image3.png": "neutral",
  "image4.png": "fear"
}
```

Rules:

- Keys must match filenames in `static/images/`
- Values should be lowercase emotion labels
- Unknown/missing labels are treated as `unknown`

## API Endpoints

- `GET /` : Load main UI
- `GET /gaze/stream` : SSE stream of gaze labels
- `GET /images` : List images and emotion map
- `GET /emotion` : Current detected emotion and confidence
- `GET /blinks` : Total blink count
- `POST /camera/start` : Start session timing and camera state
- `POST /camera/stop` : Stop camera/session state
- `POST /calibrate` : Start calibration sequence
- `GET /calibrate/status` : Calibration progress and state
- `POST /api/stop_session` : Build final report, plots, JSON, and RAG analysis

## Output Files

Generated under `static/results/`:

- `session_<timestamp>_RAG_report.json`
- `session_<timestamp>_heatmap.png`
- `session_<timestamp>_scatter.png`

The report JSON contains:

- Session metadata (duration, images interacted, blinks)
- Per-image interactions (target emotion, user emotion, matched, gaze region, duration, views)
- `rag_analysis` section with generated answer, source, and optional error

## Standalone Diagnostic Tool

To inspect raw vertical gaze signals:

```powershell
python diagnostic.py
```

Press `q` to quit.

This is useful for validating signal range when tuning tracking behavior.

## Troubleshooting

### 1) Camera not found

- Ensure webcam is connected and not locked by another app.
- The backend tries camera indexes 0, 1, 2.

### 2) Tracker stuck on NO_FACE

- Improve lighting and camera angle.
- Keep your face centered and visible.

### 3) DeepFace import or runtime issues

- Reinstall with:

```powershell
pip install --upgrade deepface
```

### 4) MediaPipe compatibility issue (`mp.solutions` missing)

Known-good Windows/Python 3.12 setup for this project uses:

- `mediapipe==0.10.13`

Some newer builds can expose task-only APIs and break `mp.solutions.face_mesh` usage.

### 5) Gemini unavailable

- If `GEMINI_API_KEY` is missing/invalid or `google-genai` fails, app falls back to a local summary.

### 6) TensorFlow namespace/import problems

If TensorFlow appears broken, one known repair command is:

```powershell
pip install --force-reinstall --no-deps tensorflow==2.19.1
```

## Notes

- Matplotlib is configured with `Agg` backend, so plot rendering works without GUI windows.
- Session reports are written locally; there is no database dependency.
- This app is currently configured for local development (`debug=False`, threaded Flask server).
