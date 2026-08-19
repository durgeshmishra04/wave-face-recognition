import os
import time
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote
import sqlite3
import json
import hashlib
import io

import cv2
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from flask import Flask, Response, render_template_string
from insightface.app import FaceAnalysis

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

NVR_IP = "115.247.225.82"
NVR_USERNAME = "admin"

# IMPORTANT:
# Set your password in the environment instead of hard-coding it.
# Windows CMD:
#   set ALCON_PASSWORD=your_password
#
# Or PowerShell:
#   $env:ALCON_PASSWORD="your_password"
NVR_PASSWORD = os.getenv("ALCON_PASSWORD", "")

RTSP_PORT = 554
CHANNEL = 1                 # Camera 1 = MAIN GATE 01
SUBTYPE = 1                 # 0 = main stream, 1 = sub-stream
RTSP_PATH = "/cam/realmonitor"

# Face recognition threshold.
# Start around 0.45-0.55 and tune using your actual camera.
RECOGNITION_THRESHOLD = 0.50

# Process every Nth frame for face recognition.
# Higher = less CPU/GPU usage but less frequent recognition.
PROCESS_EVERY_N_FRAMES = 2

# Detection size. 640x640 is a good starting point.
DET_SIZE = (640, 640)

# GPU mode: auto uses CUDA when available and falls back to CPU.
# Set USE_GPU=true to require CUDA, or false to force CPU.
USE_GPU = os.getenv("USE_GPU", "auto").strip().lower()
CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))

# Folder containing enrolled/reference face images.
# Example:
#   known_faces/
#       Rahul.jpg
#       Amit.jpg
#       John.jpg
KNOWN_FACES_DIR = Path("known_faces")

# Local web server
HOST = "0.0.0.0"
PORT = 5000

# ============================================================
# GLOBAL STATE
# ============================================================

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()

camera_running = True
camera_status = "Starting..."

known_embeddings = {}
face_app = None
UNKNOWN_ALERT_COOLDOWN = 10.0  # seconds between unknown alerts
last_unknown_alert = 0.0
UNKNOWN_DISPLAY_DURATION = 3.0  # seconds to display red alert on video
UNKNOWN_GONE_CLEARANCE = 0.75  # seconds after last unknown seen to consider it gone
MIN_UNKNOWN_PRESENCE = 1.0     # minimum seconds unknown must be present before alert on leaving
unknown_present = False
unknown_first_seen = 0.0
unknown_last_seen = 0.0


# ============================================================
# RTSP URL
# ============================================================

def build_rtsp_url():
    if not NVR_PASSWORD:
        raise RuntimeError(
            "ALCON_PASSWORD environment variable is not set.\n"
            "Windows CMD: set ALCON_PASSWORD=your_password\n"
            "PowerShell: $env:ALCON_PASSWORD=\"your_password\""
        )

    # URL-encode username/password so characters such as @ are safe.
    user = quote(NVR_USERNAME, safe="")
    password = quote(NVR_PASSWORD, safe="")

    return (
        f"rtsp://{user}:{password}@{NVR_IP}:{RTSP_PORT}"
        f"{RTSP_PATH}?channel={CHANNEL}&subtype={SUBTYPE}"
    )


# ============================================================
# INSIGHTFACE / BUFFALO_L
# ============================================================

def get_onnx_providers():
    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is not None and USE_GPU not in {"false", "0", "no", "off"}:
        preload_dlls()

    available = ort.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in available

    if USE_GPU in {"true", "1", "yes", "on"} and not cuda_available:
        raise RuntimeError(
            "USE_GPU is enabled, but CUDAExecutionProvider is unavailable. "
            "Install onnxruntime-gpu and compatible NVIDIA CUDA/cuDNN libraries."
        )

    if USE_GPU not in {"auto", "true", "1", "yes", "on", "false", "0", "no", "off"}:
        raise ValueError("USE_GPU must be auto, true, or false")

    if cuda_available and USE_GPU not in {"false", "0", "no", "off"}:
        print(f"[INFO] Using CUDA GPU {CUDA_DEVICE_ID}.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("[INFO] Using CPU execution provider.")
    return ["CPUExecutionProvider"]


def initialize_face_model():
    global face_app
    providers = get_onnx_providers()

    print("[INFO] Loading InsightFace buffalo_l...")
    print("[INFO] Recognition model in buffalo_l is ArcFace-based.")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=providers
    )

    face_app.prepare(
        ctx_id=CUDA_DEVICE_ID if "CUDAExecutionProvider" in providers else -1,
        det_size=DET_SIZE,
        det_thresh=0.50
    )

    print("[OK] Face model loaded.")


# ============================================================
# KNOWN FACE DATABASE
# ============================================================

def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


# -----------------------------
# SQLite cache for embeddings
# -----------------------------

DB_FILENAME = KNOWN_FACES_DIR / "known_faces.db"


def _open_db():
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILENAME))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS known_embeddings (
            person TEXT PRIMARY KEY,
            embedding BLOB,
            image_paths TEXT,
            fingerprint TEXT,
            updated_at REAL
        )
        """
    )


def _compute_fingerprint(image_paths):
    entries = []
    for p in sorted(image_paths):
        try:
            st = p.stat()
            entries.append(f"{p.name}:{int(st.st_mtime)}")
        except Exception:
            entries.append(f"{p.name}:0")

    data = "|".join(entries).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _save_embedding_db(person, embedding, image_paths, fingerprint):
    buf = io.BytesIO()
    # use numpy's .npy format for portability
    np.save(buf, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
    blob = buf.getvalue()

    conn = _open_db()
    try:
        _ensure_table(conn)
        conn.execute(
            "REPLACE INTO known_embeddings (person, embedding, image_paths, fingerprint, updated_at) VALUES (?, ?, ?, ?, ?)",
            (person, sqlite3.Binary(blob), json.dumps([str(p) for p in image_paths]), fingerprint, time.time())
        )
        conn.commit()
    finally:
        conn.close()


def _load_embedding_db(person):
    conn = _open_db()
    try:
        _ensure_table(conn)
        cur = conn.execute("SELECT embedding, image_paths, fingerprint FROM known_embeddings WHERE person = ?", (person,))
        row = cur.fetchone()
        if not row:
            return None

        blob, image_paths_json, fingerprint = row
        buf = io.BytesIO(blob)
        buf.seek(0)
        arr = np.load(buf, allow_pickle=False)
        image_paths = json.loads(image_paths_json)
        return arr.astype(np.float32), image_paths, fingerprint
    finally:
        conn.close()


def load_known_faces():
    global known_embeddings

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Collect person entries either from subdirectories or single files.
    persons = [p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()]

    single_files = []
    if not persons:
        single_files = [
            p for p in KNOWN_FACES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        ]

    # helper to process a person given its image paths and name
    def _process_person(person_name, image_paths):
        # compute fingerprint
        fingerprint = _compute_fingerprint(image_paths)

        # try load from DB
        cached = _load_embedding_db(person_name)
        if cached is not None:
            emb, paths_json, cached_fp = cached
            if cached_fp == fingerprint:
                known_embeddings[person_name] = normalize_embedding(emb)
                print(f"[OK] Loaded cached: {person_name}")
                return

        # compute embeddings from images
        embeddings = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARNING] Could not read: {image_path}")
                continue

            faces = face_app.get(image)
            if not faces:
                print(f"[WARNING] No face found in: {image_path}")
                continue

            face = max(
                faces,
                key=lambda f: (
                    max(0, f.bbox[2] - f.bbox[0])
                    * max(0, f.bbox[3] - f.bbox[1])
                )
            )

            embeddings.append(normalize_embedding(face.embedding))

        if not embeddings:
            print(f"[WARNING] No valid face embeddings for: {person_name}")
            return

        avg = np.mean(np.stack(embeddings, axis=0), axis=0)
        avg = normalize_embedding(avg)

        known_embeddings[person_name] = avg
        _save_embedding_db(person_name, avg, image_paths, fingerprint)
        print(f"[OK] Enrolled: {person_name} ({len(embeddings)} image(s))")

    # Process single-file-per-person mode
    if single_files:
        if not single_files:
            print()
            print("[WARNING] No known faces found.")
            print(f"[INFO] Put reference photos inside: {KNOWN_FACES_DIR.resolve()}")
            print("[INFO] Example: known_faces/PersonName/image1.jpg")
            print()
            return

        for image_path in single_files:
            person_name = image_path.stem
            _process_person(person_name, [image_path])

    else:
        # directory-per-person mode
        for person_dir in persons:
            person_name = person_dir.name
            image_files = [
                p for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in supported
            ]

            if not image_files:
                print(f"[WARNING] No images found for: {person_name}")
                continue

            _process_person(person_name, image_files)

    print(f"[INFO] Loaded {len(known_embeddings)} known face(s).")


def recognize_face(face):
    if not known_embeddings:
        return "Unknown", 0.0

    query = normalize_embedding(face.embedding)

    best_name = "Unknown"
    best_score = -1.0

    for name, reference in known_embeddings.items():
        score = float(np.dot(query, reference))

        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= RECOGNITION_THRESHOLD:
        return best_name, best_score

    return "Unknown", best_score


# ============================================================
# DRAWING
# ============================================================

def draw_face(frame, face, name, score):
    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    # Keep coordinates inside the image.
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    # Green for recognized, red for unknown.
    if name != "Unknown":
        color = (0, 255, 0)
        label = f"{name}  {score:.2f}"
    else:
        color = (0, 0, 255)
        label = "Unknown"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # Label background
    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2
    )

    label_y1 = max(0, y1 - th - baseline - 8)
    label_y2 = max(th + baseline + 8, y1)

    cv2.rectangle(
        frame,
        (x1, label_y1),
        (x1 + tw + 10, label_y2),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_worker():
    global latest_frame, camera_status, last_unknown_alert
    global unknown_present, unknown_first_seen, unknown_last_seen

    frame_counter = 0
    last_faces = []

    while camera_running:

        try:
            rtsp_url = build_rtsp_url()

            print("[INFO] Connecting to RTSP...")
            print(
                f"[INFO] Camera: {CHANNEL}, "
                f"subtype: {SUBTYPE}"
            )

            # CAP_FFMPEG helps on many Windows installations.
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            # Reduce buffering where supported.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                camera_status = "RTSP connection failed"
                print("[ERROR] Could not open RTSP stream.")
                print("[INFO] Check RTSP port/path/credentials.")
                time.sleep(5)
                continue

            camera_status = "LIVE"
            print("[OK] RTSP stream connected.")

            while camera_running:
                ok, frame = cap.read()

                if not ok or frame is None:
                    camera_status = "Stream lost - reconnecting..."
                    print("[WARNING] Frame read failed.")
                    break

                frame_counter += 1

                # Run face analysis every Nth frame.
                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    try:
                        last_faces = face_app.get(frame)

                        # Track unknown presence and only alert after it leaves.
                        now = time.time()
                        seen_unknown_in_frame = False

                        for face in last_faces:
                            name, score = recognize_face(face)
                            face.recognized_name = name
                            face.recognition_score = score

                            if name == "Unknown":
                                seen_unknown_in_frame = True

                        if seen_unknown_in_frame:
                            # Start or continue tracking the unknown presence.
                            if not unknown_present:
                                unknown_present = True
                                unknown_first_seen = now
                            unknown_last_seen = now
                        else:
                            # No unknown in current frame. If we were tracking and the unknown
                            # has been gone for a short clearance interval and was present
                            # for at least MIN_UNKNOWN_PRESENCE, trigger the alert once.
                            if unknown_present:
                                if now - unknown_last_seen >= UNKNOWN_GONE_CLEARANCE:
                                    presence_duration = unknown_last_seen - unknown_first_seen
                                    if presence_duration >= MIN_UNKNOWN_PRESENCE:
                                        # Rate-limit alerts using last_unknown_alert.
                                        if now - last_unknown_alert > UNKNOWN_ALERT_COOLDOWN:
                                            print("[ALERT] Unknown person detected (left the frame)!")
                                            camera_status = "Unknown person detected"
                                            last_unknown_alert = now
                                    # Reset tracking state after handling.
                                    unknown_present = False
                                    unknown_first_seen = 0.0
                                    unknown_last_seen = 0.0

                    except Exception as e:
                        print(f"[ERROR] Face processing error: {e}")

                # Draw the most recently detected faces.
                output = frame.copy()

                for face in last_faces:
                    name = getattr(face, "recognized_name", "Unknown")
                    score = getattr(face, "recognition_score", 0.0)
                    draw_face(output, face, name, score)

                # If a recent unknown alert exists, draw a prominent red banner.
                try:
                    now = time.time()
                    if now - last_unknown_alert < UNKNOWN_DISPLAY_DURATION:
                        overlay = output.copy()
                        h, w = output.shape[:2]

                        # Red banner at top
                        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)

                        # Blend for semi-transparency
                        alpha = 0.6
                        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

                        # Large white alert text
                        alert_text = "UNKNOWN PERSON DETECTED"
                        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                        x = max(10, (w - tw) // 2)
                        y = 55

                        cv2.putText(
                            output,
                            alert_text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.1,
                            (255, 255, 255),
                            3,
                            cv2.LINE_AA
                        )
                except Exception:
                    pass

                # Add small status text box (bottom-left)
                cv2.rectangle(
                    output,
                    (10, 10),
                    (390, 48),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    output,
                    f"ALCON Camera {CHANNEL} | {camera_status}",
                    (20, 37),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Encode frame for browser.
                success, encoded = cv2.imencode(
                    ".jpg",
                    output,
                    [cv2.IMWRITE_JPEG_QUALITY, 80]
                )

                if success:
                    with frame_lock:
                        latest_frame = encoded.tobytes()

            cap.release()

        except Exception as e:
            camera_status = f"Error: {e}"
            print(f"[ERROR] Camera worker: {e}")
            time.sleep(5)


# ============================================================
# WEB STREAM
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ALCON Camera 1 - Face Recognition</title>

    <meta charset="UTF-8">

    <style>
        body {
            margin: 0;
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
        }

        h1 {
            margin: 18px 0 5px 0;
            font-size: 24px;
        }

        p {
            margin: 5px 0 15px 0;
            color: #bbb;
        }

        .video-container {
            width: 95%;
            max-width: 1600px;
            margin: auto;
        }

        img {
            width: 100%;
            height: auto;
            border: 2px solid #444;
            border-radius: 6px;
        }
    </style>
</head>

<body>

    <h1>ALCON — MAIN GATE 01</h1>
    <p>Camera 1 | Buffalo_L + ArcFace Face Recognition</p>

    <div class="video-container">
        <img src="/video_feed">
    </div>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


def generate_frames():
    while True:

        with frame_lock:
            frame = latest_frame

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(" ALCON CAMERA 1 - FACE RECOGNITION")
    print(" InsightFace buffalo_l + ArcFace")
    print("=" * 60)

    initialize_face_model()

    load_known_faces()

    worker = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    worker.start()

    url = f"http://{HOST}:{PORT}/"

    print()
    print(f"[INFO] Web interface: {url}")
    print("[INFO] Opening browser...")
    print("[INFO] Press CTRL+C in this terminal to stop.")
    print()

    # Give Flask a moment to start.
    threading.Timer(
        1.5,
        lambda: webbrowser.open_new_tab(url)
    ).start()

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":
    main()
