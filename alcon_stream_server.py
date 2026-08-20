import os
import time
import threading
import base64
from pathlib import Path
from urllib.parse import quote
import sqlite3
import json
import hashlib
import io
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from insightface.app import FaceAnalysis

load_dotenv()

# Force RTSP over TCP instead of UDP. This NVR is reachable at a public IP
# (not a LAN address), and OpenCV/FFmpeg defaults RTSP to UDP transport —
# over a WAN link that silently drops packets under any jitter, and the
# loss gets rendered as corrupted macroblocks / brightness pulses baked
# directly into the decoded frame. That shows up client-side as flicker,
# and no amount of client-side buffering can fix it since the source
# frame itself is already bad. Must be set before cv2.VideoCapture() is
# ever called.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ============================================================
# CONFIGURATION
# ============================================================

NVR_IP = "115.247.225.82"
NVR_USERNAME = "admin"
NVR_PASSWORD = os.getenv("ALCON_PASSWORD", "")

RTSP_PORT = 554
CHANNEL = 1
SUBTYPE = 1
RTSP_PATH = "/cam/realmonitor"

RECOGNITION_THRESHOLD = 0.50
PROCESS_EVERY_N_FRAMES = 2
DET_SIZE = (640, 640)

USE_GPU = os.getenv("USE_GPU", "auto").strip().lower()
CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))

KNOWN_FACES_DIR = Path("known_faces")

HOST = "0.0.0.0"
PORT = 5000

# Streaming tuning (important for mobile bandwidth)
STREAM_JPEG_QUALITY = 65        # lower = smaller payload, good for mobile data
STREAM_MAX_WIDTH = 800          # frames resized before sending over socket
STREAM_TARGET_FPS = 10          # frames pushed per second to clients (not the same as detection fps)

UNKNOWN_ALERT_COOLDOWN = 10.0
UNKNOWN_GONE_CLEARANCE = 0.75
MIN_UNKNOWN_PRESENCE = 1.0

MAX_ALERT_HISTORY = 100

# ============================================================
# APP / SOCKET SETUP
# ============================================================

app = Flask(__name__)
CORS(app)  # allow requests from the React Native app / dev machine
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

latest_frame_lock = threading.Lock()
latest_annotated_frame = None   # raw bytes (jpeg) for /api/snapshot

camera_running = True
camera_status = "Starting..."

known_embeddings = {}
face_app = None

unknown_present = False
unknown_first_seen = 0.0
unknown_last_seen = 0.0
last_unknown_alert = 0.0

alert_history = deque(maxlen=MAX_ALERT_HISTORY)  # each item: {"time":..., "message":...}
connected_clients = 0


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

    if cuda_available and USE_GPU not in {"false", "0", "no", "off"}:
        print(f"[INFO] Using CUDA GPU {CUDA_DEVICE_ID}.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("[INFO] Using CPU execution provider.")
    return ["CPUExecutionProvider"]


def initialize_face_model():
    global face_app
    providers = get_onnx_providers()

    print("[INFO] Loading InsightFace buffalo_l...")
    face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    face_app.prepare(
        ctx_id=CUDA_DEVICE_ID if "CUDAExecutionProvider" in providers else -1,
        det_size=DET_SIZE,
        det_thresh=0.50
    )
    print("[OK] Face model loaded.")


# ============================================================
# KNOWN FACE DATABASE (unchanged logic, SQLite-cached)
# ============================================================

def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm


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
    return hashlib.sha1("|".join(entries).encode("utf-8")).hexdigest()


def _save_embedding_db(person, embedding, image_paths, fingerprint):
    buf = io.BytesIO()
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
        cur = conn.execute(
            "SELECT embedding, image_paths, fingerprint FROM known_embeddings WHERE person = ?", (person,)
        )
        row = cur.fetchone()
        if not row:
            return None
        blob, image_paths_json, fingerprint = row
        buf = io.BytesIO(blob)
        buf.seek(0)
        arr = np.load(buf, allow_pickle=False)
        return arr.astype(np.float32), json.loads(image_paths_json), fingerprint
    finally:
        conn.close()


def load_known_faces():
    global known_embeddings
    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    persons = [p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()]
    single_files = []
    if not persons:
        single_files = [
            p for p in KNOWN_FACES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        ]

    def _process_person(person_name, image_paths):
        fingerprint = _compute_fingerprint(image_paths)
        cached = _load_embedding_db(person_name)
        if cached is not None:
            emb, _, cached_fp = cached
            if cached_fp == fingerprint:
                known_embeddings[person_name] = normalize_embedding(emb)
                print(f"[OK] Loaded cached: {person_name}")
                return

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
                key=lambda f: max(0, f.bbox[2] - f.bbox[0]) * max(0, f.bbox[3] - f.bbox[1])
            )
            embeddings.append(normalize_embedding(face.embedding))

        if not embeddings:
            print(f"[WARNING] No valid face embeddings for: {person_name}")
            return

        avg = normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0))
        known_embeddings[person_name] = avg
        _save_embedding_db(person_name, avg, image_paths, fingerprint)
        print(f"[OK] Enrolled: {person_name} ({len(embeddings)} image(s))")

    if single_files:
        for image_path in single_files:
            _process_person(image_path.stem, [image_path])
    else:
        for person_dir in persons:
            image_files = [
                p for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in supported
            ]
            if not image_files:
                print(f"[WARNING] No images found for: {person_dir.name}")
                continue
            _process_person(person_dir.name, image_files)

    print(f"[INFO] Loaded {len(known_embeddings)} known face(s).")


def recognize_face(face):
    if not known_embeddings:
        return "Unknown", 0.0

    query = normalize_embedding(face.embedding)
    best_name, best_score = "Unknown", -1.0

    for name, reference in known_embeddings.items():
        score = float(np.dot(query, reference))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= RECOGNITION_THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score


# ============================================================
# ALERT HANDLING
# ============================================================

def push_alert(message):
    entry = {"time": time.time(), "message": message}
    alert_history.append(entry)
    # notify any connected React Native clients immediately
    socketio.emit("face_alert", entry)


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker():
    global camera_status, last_unknown_alert
    global unknown_present, unknown_first_seen, unknown_last_seen
    global latest_annotated_frame

    frame_counter = 0
    last_faces = []
    last_emit_time = 0.0
    emit_interval = 1.0 / STREAM_TARGET_FPS

    while camera_running:
        try:
            rtsp_url = build_rtsp_url()
            print("[INFO] Connecting to RTSP...")
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                camera_status = "RTSP connection failed"
                print("[ERROR] Could not open RTSP stream.")
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
                faces_payload = []

                if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                    try:
                        last_faces = face_app.get(frame)
                        now = time.time()
                        seen_unknown_in_frame = False

                        for face in last_faces:
                            name, score = recognize_face(face)
                            face.recognized_name = name
                            face.recognition_score = score
                            if name == "Unknown":
                                seen_unknown_in_frame = True

                        if seen_unknown_in_frame:
                            if not unknown_present:
                                unknown_present = True
                                unknown_first_seen = now
                            unknown_last_seen = now
                        elif unknown_present and now - unknown_last_seen >= UNKNOWN_GONE_CLEARANCE:
                            presence_duration = unknown_last_seen - unknown_first_seen
                            if presence_duration >= MIN_UNKNOWN_PRESENCE:
                                if now - last_unknown_alert > UNKNOWN_ALERT_COOLDOWN:
                                    print("[ALERT] Unknown person detected (left the frame)!")
                                    camera_status = "Unknown person detected"
                                    last_unknown_alert = now
                                    push_alert("Unknown person detected at MAIN GATE 01")
                            unknown_present = False
                            unknown_first_seen = 0.0
                            unknown_last_seen = 0.0

                    except Exception as e:
                        print(f"[ERROR] Face processing error: {e}")

                height, width = frame.shape[:2]
                for face in last_faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    faces_payload.append({
                        "name": getattr(face, "recognized_name", "Unknown"),
                        "score": round(float(getattr(face, "recognition_score", 0.0)), 3),
                        "box": {
                            "x1": int(max(0, min(x1, width - 1))),
                            "y1": int(max(0, min(y1, height - 1))),
                            "x2": int(max(0, min(x2, width - 1))),
                            "y2": int(max(0, min(y2, height - 1))),
                        }
                    })

                now = time.time()
                if now - last_emit_time >= emit_interval:
                    last_emit_time = now
                    send_frame = frame.copy()

                    for face in last_faces:
                        x1, y1, x2, y2 = face.bbox.astype(int)
                        name = getattr(face, "recognized_name", "Unknown")
                        score = float(getattr(face, "recognition_score", 0.0))

                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))

                        if name == "Unknown":
                            color = (0, 0, 255)
                            label = "Unknown"
                        else:
                            color = (0, 255, 0)
                            label = f"{name}  {score:.2f}"

                        cv2.rectangle(send_frame, (x1, y1), (x2, y2), color, 2)

                        (text_width, text_height), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            2,
                        )
                        label_y1 = max(0, y1 - text_height - baseline - 8)
                        label_y2 = max(text_height + baseline + 8, y1)
                        cv2.rectangle(
                            send_frame,
                            (x1, label_y1),
                            (x1 + text_width + 10, label_y2),
                            color,
                            -1,
                        )
                        cv2.putText(
                            send_frame,
                            label,
                            (x1 + 5, label_y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    if width > STREAM_MAX_WIDTH:
                        scale = STREAM_MAX_WIDTH / float(width)
                        send_frame = cv2.resize(send_frame, (STREAM_MAX_WIDTH, int(height * scale)))

                    success, encoded = cv2.imencode(
                        ".jpg", send_frame, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
                    )

                    if success:
                        encoded_bytes = encoded.tobytes()
                        with latest_frame_lock:
                            latest_annotated_frame = encoded_bytes

                        socketio.emit("face_frame", {
                            "image": base64.b64encode(encoded_bytes).decode("utf-8"),
                            "width": send_frame.shape[1],
                            "height": send_frame.shape[0],
                            "faces": faces_payload,
                            "status": camera_status,
                            "timestamp": now,
                        })

            cap.release()

        except Exception as e:
            camera_status = f"Error: {e}"
            print(f"[ERROR] Camera worker: {e}")
            time.sleep(5)


# ============================================================
# REST API (for React Native)
# ============================================================

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": camera_status,
        "connected_clients": connected_clients,
        "known_faces_count": len(known_embeddings),
    })


@app.route("/api/known-faces")
def api_known_faces():
    return jsonify({"people": sorted(known_embeddings.keys())})


@app.route("/api/alerts")
def api_alerts():
    limit = request.args.get("limit", default=20, type=int)
    items = list(alert_history)[-limit:]
    items.reverse()  # most recent first
    return jsonify({"alerts": items})


@app.route("/api/snapshot")
def api_snapshot():
    with latest_frame_lock:
        frame = latest_annotated_frame

    if frame is None:
        return jsonify({"error": "No frame available yet"}), 503

    b64_frame = base64.b64encode(frame).decode("utf-8")
    return jsonify({"image": b64_frame, "status": camera_status})


# ============================================================
# SOCKET.IO EVENTS
# ============================================================

@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    print(f"[INFO] Client connected. Total: {connected_clients}")


@socketio.on("disconnect")
def on_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    print(f"[INFO] Client disconnected. Total: {connected_clients}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" ALCON CAMERA 1 - FACE RECOGNITION (Socket.IO API mode)")
    print("=" * 60)

    initialize_face_model()
    load_known_faces()

    worker = threading.Thread(target=camera_worker, daemon=True)
    worker.start()

    print()
    print(f"[INFO] REST API base:      http://<server-ip>:{PORT}/api/")
    print(f"[INFO] Socket.IO endpoint: http://<server-ip>:{PORT}  (event: 'face_frame', 'face_alert')")
    print("[INFO] Press CTRL+C in this terminal to stop.")
    print()

    socketio.run(app, host=HOST, port=PORT, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()