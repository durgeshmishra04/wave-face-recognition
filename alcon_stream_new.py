
import os
import time
import threading
import base64
from pathlib import Path
from urllib.parse import quote
from urllib.parse import urlencode
from urllib.request import HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, build_opener
import sqlite3
import json
import hashlib
import io
import uuid
import shutil
from collections import deque
from types import SimpleNamespace

import firebase_admin
from firebase_admin import credentials, messaging

import cv2
import numpy as np
import onnxruntime as ort

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from insightface.app import FaceAnalysis
from detection_database import DetectionDatabase
from detection_events import DetectionEventManager


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

NVR_IP = "115.247.225.82"
NVR_USERNAME = "admin"
NVR_PASSWORD = os.getenv("ALCON_PASSWORD", "")

RTSP_PORT = 554
NVR_HTTP_PORT = int(os.getenv("NVR_HTTP_PORT", "80"))
CHANNEL = 1
SUBTYPE = 1
RTSP_PATH = "/cam/realmonitor"

RECOGNITION_THRESHOLD = 0.50
PROCESS_EVERY_N_FRAMES = 2
DET_SIZE_VALUE = int(os.getenv("DET_SIZE", "800"))
DET_SIZE = (DET_SIZE_VALUE, DET_SIZE_VALUE)
DET_THRESH = float(os.getenv("DET_THRESH", "0.40"))
UNKNOWN_FACE_MIN_SCORE = float(
    os.getenv("UNKNOWN_FACE_MIN_SCORE", "0.60")
)
PERSON_MODEL = os.getenv("PERSON_MODEL", "yolo11n.pt")
PERSON_CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.45"))
PERSON_IOU = float(os.getenv("PERSON_IOU", "0.45"))

USE_GPU = os.getenv("USE_GPU", "auto").strip().lower()
CUDA_DEVICE_ID = int(os.getenv("CUDA_DEVICE_ID", "0"))
VEHICLE_MODEL_PATH = "yolo26l.pt"
VEHICLE_CONFIDENCE = 0.35
VEHICLE_DEVICE = f"cuda:{CUDA_DEVICE_ID}"
TWO_WHEELER_CLASSES = {
    "motorcycle"
}
FOUR_WHEELER_CLASSES = {
    "car",
    "bus",
    "truck"
}
VEHICLE_PROCESS_EVERY_N_FRAMES = 2

# Resolve this from the source file, not the process working directory. This
# keeps known faces and finalized detection history on the same server DB even
# when the service is started by Android deployment tooling or a scheduler.
PROJECT_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = PROJECT_DIR / "known_faces"

HOST = "0.0.0.0"
PORT = 5000


# ============================================================
# PUBLIC IMAGE CONFIGURATION
# ============================================================

# Images will be saved here
# Keep event images beside the application so image_url records remain valid
# for every mobile login, independent of the process working directory.
ALERT_IMAGE_DIR = PROJECT_DIR / "alert_images"
ALERT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# Change this to your actual public HTTPS domain.
#
# Example:
# PUBLIC_BASE_URL = "https://bhcs.biconnect.in"
#
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "http://103.234.71.180:5000/"
).rstrip("/")


# FCM topic
FCM_TOPIC = "all_devices"


# ============================================================
# STREAMING
# ============================================================

STREAM_JPEG_QUALITY = 65
STREAM_MAX_WIDTH = 800
STREAM_TARGET_FPS = 10
MAX_CONSECUTIVE_READ_FAILURES = 5

UNKNOWN_GONE_CLEARANCE = 5.0
MIN_UNKNOWN_PRESENCE = 1.5

# Alert deduplication: prevent duplicate alerts within this time window (seconds)
ALERT_DEDUP_TIME = 10.0
# Consecutive missing ROI frames must span this duration before an exit closes
# an event. It is deliberately independent of the former alert throttle.
EXIT_CONFIRM_SECONDS = float(os.getenv("EXIT_CONFIRM_SECONDS", "1.0"))
# Final API/Firebase image is selected from confirmed exit frame - 6.
EXIT_FRAME_OFFSET = int(os.getenv("EXIT_FRAME_OFFSET", "6"))

# Region of interest for Person and Face Detection: defined by the custom polygon
ROI_LEFT = float(os.getenv("ROI_LEFT", "0.10"))
ROI_TOP = 0.54
ROI_RIGHT = 0.995
ROI_BOTTOM = 0.99
PERSON_ROI_POLYGON = np.array(
    [
        [0.098, 1.000],
        [0.149, 0.796],
        [0.195, 0.574],
        [0.265, 0.530],
        [0.450, 0.556],
        [0.700, 0.556],
        [0.850, 0.613],
        [0.912, 0.617],
        [0.951, 1.000],
    ],
    dtype=np.float32,
)
VEHICLE_ROI_LEFT = float(os.getenv("VEHICLE_ROI_LEFT", "0.00"))
VEHICLE_ROI_TOP = float(os.getenv("VEHICLE_ROI_TOP", "0.35"))
VEHICLE_ROI_RIGHT = float(os.getenv("VEHICLE_ROI_RIGHT", "1.00"))
VEHICLE_ROI_BOTTOM = float(os.getenv("VEHICLE_ROI_BOTTOM", "0.70"))

MAX_ALERT_HISTORY = 100


# ============================================================
# RTSP OVER TCP
# ============================================================

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "rtsp_flags;prefer_tcp|"
    "max_delay;1000000|"
    "reorder_queue_size;512|"
    "buffer_size;4194304|"
    "stimeout;5000000|"
    "rw_timeout;5000000|"
    "fflags;discardcorrupt|"
    "err_detect;ignore_err|"
    "flush_packets;1"
)


# ============================================================
# APP / SOCKET SETUP
# ============================================================

app = Flask(__name__)

CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

latest_frame_lock = threading.Lock()

latest_annotated_frame = None

camera_running = True
camera_status = "Starting..."

known_embeddings = {}
# Metadata for registration folders. Legacy folders retain their folder name
# as the display name, while Android registrations use a unique folder key.
known_person_metadata = {}
face_app = None
person_detector = None
vehicle_detector = None
detection_events = None

unknown_present = False
unknown_first_seen = 0.0
unknown_last_seen = 0.0
last_alert_time = 0.0

alert_history = deque(maxlen=MAX_ALERT_HISTORY)

connected_clients = 0


# ============================================================
# FIREBASE INITIALIZATION
# ============================================================

def initialize_firebase():

    try:

        # Prevent duplicate initialization
        if firebase_admin._apps:
            print("[INFO] Firebase already initialized.")
            return

        firebase_json = os.getenv(
            "FIREBASE_SERVICE_ACCOUNT",
            "./pythonai.json"
        )

        if not os.path.exists(firebase_json):
            raise FileNotFoundError(
                f"Firebase service account not found: {firebase_json}"
            )

        cred = credentials.Certificate(firebase_json)

        firebase_admin.initialize_app(cred)

        print("[OK] Firebase initialized.")

    except Exception as e:

        print(
            f"[ERROR] Firebase initialization failed: {e}"
        )

        raise


# ============================================================
# RTSP URL
# ============================================================

def configure_h264_stream():

    config_url = (
        f"http://{NVR_IP}:{NVR_HTTP_PORT}/cgi-bin/configManager.cgi?"
        + urlencode({
            "action": "setConfig",
            "Encode[0].MainFormat[0].Video.Codec": "H.264",
            "Encode[0].ExtraFormat[0].Video.Codec": "H.264",
        })
    )

    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(
        None,
        config_url,
        NVR_USERNAME,
        NVR_PASSWORD,
    )
    opener = build_opener(
        HTTPDigestAuthHandler(password_manager)
    )

    try:
        with opener.open(config_url, timeout=10) as response:
            result = response.read().decode("utf-8", errors="replace")
        if "OK" not in result.upper():
            raise RuntimeError(result.strip() or "Camera rejected H.264 configuration")
        print("[OK] Camera streams configured to H.264.")
    except Exception as e:
        print(f"[WARNING] Could not configure camera to H.264: {e}")

def build_rtsp_url():

    if not NVR_PASSWORD:

        raise RuntimeError(
            "ALCON_PASSWORD environment variable is not set."
        )

    user = quote(
        NVR_USERNAME,
        safe=""
    )

    password = quote(
        NVR_PASSWORD,
        safe=""
    )

    return (
        f"rtsp://{user}:{password}@"
        f"{NVR_IP}:{RTSP_PORT}"
        f"{RTSP_PATH}"
        f"?channel={CHANNEL}&subtype={SUBTYPE}"
    )


# ============================================================
# INSIGHTFACE
# ============================================================

def get_onnx_providers():

    preload_dlls = getattr(
        ort,
        "preload_dlls",
        None
    )

    if (
        preload_dlls is not None
        and USE_GPU not in {
            "false",
            "0",
            "no",
            "off"
        }
    ):

        preload_dlls()

    available = ort.get_available_providers()

    cuda_available = (
        "CUDAExecutionProvider"
        in available
    )

    if (
        USE_GPU in {
            "true",
            "1",
            "yes",
            "on"
        }
        and not cuda_available
    ):

        raise RuntimeError(
            "USE_GPU is enabled, but "
            "CUDAExecutionProvider is unavailable."
        )

    if (
        cuda_available
        and USE_GPU not in {
            "false",
            "0",
            "no",
            "off"
        }
    ):

        print(
            f"[INFO] Using CUDA GPU "
            f"{CUDA_DEVICE_ID}."
        )

        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

    print("[INFO] Using CPU execution provider.")

    return [
        "CPUExecutionProvider"
    ]


def initialize_face_model():

    global face_app
    global person_detector
    global vehicle_detector

    if YOLO is None:
        raise RuntimeError(
            "Person detection requires ultralytics. "
            "Install dependencies from requirements.txt."
        )

    print(
        f"[INFO] Loading person detector: {PERSON_MODEL}..."
    )
    person_detector = YOLO(PERSON_MODEL)
    print("[OK] Person detector loaded.")

    providers = get_onnx_providers()

    print(
        "[INFO] Loading InsightFace buffalo_l..."
    )

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=providers
    )

    face_app.prepare(
        ctx_id=(
            CUDA_DEVICE_ID
            if "CUDAExecutionProvider"
            in providers
            else -1
        ),
        det_size=DET_SIZE,
        det_thresh=DET_THRESH
    )

    print(
        f"[INFO] Loading vehicle detector: {VEHICLE_MODEL_PATH}..."
    )
    vehicle_detector = YOLO(VEHICLE_MODEL_PATH)
    print("[OK] Vehicle detector loaded.")

    print("[OK] Face model loaded.")


# ============================================================
# KNOWN FACE DATABASE
# ============================================================

DB_FILENAME = (
    KNOWN_FACES_DIR /
    "known_faces.db"
)

detection_database = DetectionDatabase(DB_FILENAME)


def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,

        
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return embedding

    return embedding / norm


def _open_db():

    KNOWN_FACES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        str(DB_FILENAME)
    )

    conn.execute(
        "PRAGMA journal_mode=WAL;"
    )

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

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at REAL NOT NULL,
            known_count INTEGER NOT NULL DEFAULT 0,
            unknown_count INTEGER NOT NULL DEFAULT 0,
            two_wheeler_count INTEGER NOT NULL DEFAULT 0,
            four_wheeler_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS registered_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            registration_id TEXT NOT NULL UNIQUE,
            storage_key TEXT NOT NULL UNIQUE,
            gate_no TEXT NOT NULL,
            employee_id TEXT UNIQUE,
            employee_name TEXT NOT NULL,
            designation TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS registered_face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            registration_id TEXT NOT NULL,
            image_number INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE (registration_id, image_number)
        )
        """
    )


def _embedding_blob(embedding):

    buffer = io.BytesIO()
    np.save(
        buffer,
        np.asarray(embedding, dtype=np.float32),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _registered_person_metadata():
    """Return metadata keyed by the existing known_embeddings storage key."""

    conn = _open_db()
    try:
        _ensure_table(conn)
        rows = conn.execute(
            """
            SELECT storage_key, employee_name, employee_id, designation, gate_no
            FROM registered_persons
            """
        ).fetchall()
        return {
            row[0]: {
                "employee_name": row[1],
                "employee_id": row[2],
                "designation": row[3],
                "gate_no": row[4],
            }
            for row in rows
        }
    finally:
        conn.close()


def _save_detection_event(
    known_count,
    unknown_count,
    two_wheeler_count,
    four_wheeler_count,
    detected_at
):

    conn = _open_db()

    try:

        _ensure_table(conn)

        conn.execute(
            """
            INSERT INTO detection_events
            (
                detected_at,
                known_count,
                unknown_count,
                two_wheeler_count,
                four_wheeler_count
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                detected_at,
                known_count,
                unknown_count,
                two_wheeler_count,
                four_wheeler_count
            )
        )

        conn.commit()

    except Exception as e:

        print(
            f"[ERROR] Detection event database save failed: {e}"
        )

    finally:

        conn.close()


def _compute_fingerprint(image_paths):

    entries = []

    for p in sorted(image_paths):

        try:

            st = p.stat()

            entries.append(
                f"{p.name}:{int(st.st_mtime)}"
            )

        except Exception:

            entries.append(
                f"{p.name}:0"
            )

    return hashlib.sha1(
        "|".join(entries).encode("utf-8")
    ).hexdigest()


def _save_embedding_db(
    person,
    embedding,
    image_paths,
    fingerprint
):

    buf = io.BytesIO()

    np.save(
        buf,
        np.asarray(
            embedding,
            dtype=np.float32
        ),
        allow_pickle=False
    )

    blob = buf.getvalue()

    conn = _open_db()

    try:

        _ensure_table(conn)

        conn.execute(
            """
            REPLACE INTO known_embeddings
            (
                person,
                embedding,
                image_paths,
                fingerprint,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person,
                sqlite3.Binary(blob),
                json.dumps(
                    [
                        str(p)
                        for p in image_paths
                    ]
                ),
                fingerprint,
                time.time()
            )
        )

        conn.commit()

    finally:

        conn.close()


def _load_embedding_db(person):

    conn = _open_db()

    try:

        _ensure_table(conn)

        cur = conn.execute(
            """
            SELECT
                embedding,
                image_paths,
                fingerprint
            FROM known_embeddings
            WHERE person = ?
            """,
            (person,)
        )

        row = cur.fetchone()

        if not row:

            return None

        (
            blob,
            image_paths_json,
            fingerprint
        ) = row

        buf = io.BytesIO(blob)

        buf.seek(0)

        arr = np.load(
            buf,
            allow_pickle=False
        )

        return (
            arr.astype(np.float32),
            json.loads(image_paths_json),
            fingerprint
        )

    finally:

        conn.close()


def load_known_faces():

    global known_embeddings
    global known_person_metadata

    KNOWN_FACES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    known_embeddings = {}
    registered_metadata = _registered_person_metadata()
    known_person_metadata = {}

    supported = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    persons = [
        p
        for p in KNOWN_FACES_DIR.iterdir()
        if p.is_dir()
    ]

    single_files = []

    if not persons:

        single_files = [
            p
            for p in KNOWN_FACES_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in supported
            )
        ]


    def _process_person(
        person_name,
        image_paths
    ):

        known_person_metadata[person_name] = registered_metadata.get(
            person_name,
            {
                "employee_name": person_name,
                "employee_id": None,
            },
        )

        fingerprint = _compute_fingerprint(
            image_paths
        )

        cached = _load_embedding_db(
            person_name
        )

        if cached is not None:

            emb, _, cached_fp = cached

            if cached_fp == fingerprint:

                known_embeddings[
                    person_name
                ] = normalize_embedding(emb)

                print(
                    f"[OK] Loaded cached: "
                    f"{person_name}"
                )

                return

        embeddings = []

        for image_path in image_paths:

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                print(
                    f"[WARNING] Could not read: "
                    f"{image_path}"
                )

                continue

            faces = face_app.get(image)

            if not faces:

                print(
                    f"[WARNING] No face found in: "
                    f"{image_path}"
                )

                continue

            face = max(
                faces,
                key=lambda f:
                max(
                    0,
                    f.bbox[2] - f.bbox[0]
                )
                *
                max(
                    0,
                    f.bbox[3] - f.bbox[1]
                )
            )

            embeddings.append(
                normalize_embedding(
                    face.embedding
                )
            )

        if not embeddings:

            print(
                f"[WARNING] No valid face "
                f"embeddings for: "
                f"{person_name}"
            )

            return

        avg = normalize_embedding(
            np.mean(
                np.stack(
                    embeddings,
                    axis=0
                ),
                axis=0
            )
        )

        known_embeddings[
            person_name
        ] = avg

        _save_embedding_db(
            person_name,
            avg,
            image_paths,
            fingerprint
        )

        print(
            f"[OK] Enrolled: "
            f"{person_name} "
            f"({len(embeddings)} image(s))"
        )


    if single_files:

        for image_path in single_files:

            _process_person(
                image_path.stem,
                [image_path]
            )

    else:

        for person_dir in persons:

            image_files = [
                p
                for p in person_dir.iterdir()
                if (
                    p.is_file()
                    and p.suffix.lower()
                    in supported
                )
            ]

            if not image_files:

                print(
                    f"[WARNING] No images found: "
                    f"{person_dir.name}"
                )

                continue

            _process_person(
                person_dir.name,
                image_files
            )

    print(
        f"[INFO] Loaded "
        f"{len(known_embeddings)} "
        f"known face(s)."
    )


def recognize_face(face):

    if not known_embeddings:

        return "Unknown", 0.0

    query = normalize_embedding(
        face.embedding
    )

    best_key = None
    best_score = -1.0

    for (
        name,
        reference
    ) in known_embeddings.items():

        score = float(
            np.dot(
                query,
                reference
            )
        )

        if score > best_score:

            best_score = score
            best_key = name

    if best_score >= RECOGNITION_THRESHOLD:

        metadata = known_person_metadata.get(
            best_key,
            {"employee_name": best_key, "employee_id": None},
        )
        # The event manager reads this existing optional attribute when it
        # records a known-person event.
        face.person_id = metadata.get("employee_id")

        return (
            metadata["employee_name"],
            best_score
        )

    return (
        "Unknown",
        best_score
    )


def boxes_overlap(
    first_box,
    second_box
):

    first_x1, first_y1, first_x2, first_y2 = first_box

    second_x1, second_y1, second_x2, second_y2 = second_box

    intersection_width = max(
        0,
        min(first_x2, second_x2)
        -
        max(first_x1, second_x1)
    )

    intersection_height = max(
        0,
        min(first_y2, second_y2)
        -
        max(first_y1, second_y1)
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    first_area = (
        max(
            0,
            first_x2 - first_x1
        )
        *
        max(
            0,
            first_y2 - first_y1
        )
    )

    second_area = (
        max(
            0,
            second_x2 - second_x1
        )
        *
        max(
            0,
            second_y2 - second_y1
        )
    )

    union_area = (
        first_area
        +
        second_area
        -
        intersection_area
    )

    return (
        union_area > 0
        and
        intersection_area / union_area >= 0.10
    )


def face_inside_person(face_box, person_box):

    face_x1, face_y1, face_x2, face_y2 = face_box
    person_x1, person_y1, person_x2, person_y2 = person_box

    face_center_x = (face_x1 + face_x2) / 2
    face_center_y = (face_y1 + face_y2) / 2

    return (
        person_x1 <= face_center_x <= person_x2
        and person_y1 <= face_center_y <= person_y2
    )


def face_in_roi(face_box, frame_width, frame_height):

    x1, y1, x2, y2 = face_box

    face_center_x = (x1 + x2) / 2.0
    face_center_y = (y1 + y2) / 2.0
    face_bottom_y = float(y2)

    poly_px = (
        PERSON_ROI_POLYGON
        * np.array([frame_width, frame_height], dtype=np.float32)
    ).astype(np.int32)

    in_center = (
        cv2.pointPolygonTest(
            poly_px,
            (float(face_center_x), float(face_center_y)),
            False,
        )
        >= 0
    )
    in_bottom = (
        cv2.pointPolygonTest(
            poly_px,
            (float(face_center_x), face_bottom_y),
            False,
        )
        >= 0
    )

    return in_center or in_bottom


def vehicle_in_roi(vehicle_box, frame_width, frame_height):

    x1, y1, x2, y2 = vehicle_box

    vehicle_center_x = (x1 + x2) / 2.0
    vehicle_center_y = (y1 + y2) / 2.0
    vehicle_bottom_y = float(y2)

    poly_px = (
        PERSON_ROI_POLYGON
        * np.array([frame_width, frame_height], dtype=np.float32)
    ).astype(np.int32)

    in_center = (
        cv2.pointPolygonTest(
            poly_px,
            (float(vehicle_center_x), float(vehicle_center_y)),
            False,
        )
        >= 0
    )
    in_bottom = (
        cv2.pointPolygonTest(
            poly_px,
            (float(vehicle_center_x), vehicle_bottom_y),
            False,
        )
        >= 0
    )

    return in_center or in_bottom


def detect_person_boxes(frame):

    if person_detector is None:
        raise RuntimeError("Person detector is not initialized.")

    results = person_detector.predict(
        frame,
        classes=[0],
        conf=PERSON_CONFIDENCE,
        iou=PERSON_IOU,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    return [
        box.astype(int)
        for box in results[0].boxes.xyxy.cpu().numpy()
    ]


def detect_vehicle_boxes(frame):

    if vehicle_detector is None:
        raise RuntimeError("Vehicle detector is not initialized.")

    results = vehicle_detector.predict(
        frame,
        conf=VEHICLE_CONFIDENCE,
        device=VEHICLE_DEVICE,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    names = results[0].names
    vehicles = []

    for box, class_id, confidence in zip(
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
    ):
        class_name = str(names[int(class_id)]).strip().lower()
        class_name = class_name.replace("-", " ").replace("_", " ")

        if class_name in {"motorcycle", "motorbike", "two wheeler"}:
            class_name = "motorcycle"
        elif class_name in {"car", "bus", "truck", "four wheeler"}:
            class_name = (
                "car"
                if class_name == "four wheeler"
                else class_name
            )

        if (
            class_name not in TWO_WHEELER_CLASSES
            and class_name not in FOUR_WHEELER_CLASSES
        ):
            continue

        vehicle_type = (
            "two_wheeler"
            if class_name in TWO_WHEELER_CLASSES
            else "four_wheeler"
        )

        vehicles.append({
            "box": box.astype(int),
            "class_name": class_name,
            "vehicle_type": vehicle_type,
            "confidence": float(confidence),
        })

    return vehicles


def annotate_alert_frame(frame, faces):

    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        name = getattr(face, "recognized_name", "Unknown")
        score = float(getattr(face, "recognition_score", 0.0))

        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        label = "Unknown" if name == "Unknown" else f"{name}  {score:.2f}"

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated_frame


# ============================================================
# SAVE ALERT IMAGE
# ============================================================

def save_alert_image(frame):

    notification_id = str(
        uuid.uuid4()
    )

    filename = (
        f"{notification_id}.jpg"
    )

    image_path = (
        ALERT_IMAGE_DIR /
        filename
    )

    success = cv2.imwrite(
        str(image_path),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )

    if not success:

        raise RuntimeError(
            "Failed to save alert image"
        )

    image_url = (
        f"{PUBLIC_BASE_URL}"
        f"/alerts/{filename}"
    )

    return (
        notification_id,
        image_url
    )


# ============================================================
# ALERT HANDLING
# ============================================================

def push_alert(
    message,
    frame,
    gate_name="Main Gate 01"
):

    # --------------------------------------------------------
    # Generate ID and save image
    # --------------------------------------------------------

    try:

        (
            notification_id,
            image_url
        ) = save_alert_image(frame)

    except Exception as e:

        print(
            f"[ERROR] Alert image save failed: {e}"
        )

        notification_id = str(
            uuid.uuid4()
        )

        image_url = ""


    # --------------------------------------------------------
    # Local alert history
    # --------------------------------------------------------

    entry = {
        "id": notification_id,
        "time": time.time(),
        "message": message,
        "gate": gate_name,
        "image_url": image_url,
        "type": "person_detected"
    }

    alert_history.append(entry)


    # --------------------------------------------------------
    # React Native foreground Socket.IO clients
    # --------------------------------------------------------

    socketio.emit(
        "face_alert",
        entry
    )


    # --------------------------------------------------------
    # Firebase Cloud Messaging
    # --------------------------------------------------------

    try:

        firebase_message = messaging.Message(

            notification=messaging.Notification(
                title="New Person Detected",
                body=message,

                # Image URL
                image=(
                    image_url
                    if image_url
                    else None
                )
            ),

            data={
                "type": "person_detected",
                "notification_id": notification_id,
                "title": "New Person Detected",
                "message": message,
                "gate": gate_name,
                "image_url": image_url
            },

            topic=FCM_TOPIC
        )

        response = messaging.send(
            firebase_message
        )

        print(
            "[INFO] Firebase notification sent:"
        )

        print(response)

        print(
            f"[INFO] Alert image URL: "
            f"{image_url}"
        )

    except Exception as e:

        print(
            f"[ERROR] Firebase notification "
            f"failed: {e}"
        )


def initialize_detection_events():

    global detection_events

    detection_events = DetectionEventManager(
        database=detection_database,
        image_dir=ALERT_IMAGE_DIR,
        public_base_url=PUBLIC_BASE_URL,
        socketio=socketio,
        firebase_topic=FCM_TOPIC,
        dedup_seconds=EXIT_CONFIRM_SECONDS,
        exit_frame_offset=EXIT_FRAME_OFFSET,
    )


# ============================================================
# CAMERA WORKER
# ============================================================

def camera_worker():

    global camera_status

    global last_alert_time

    global unknown_present
    global unknown_first_seen
    global unknown_last_seen

    global latest_annotated_frame


    frame_counter = 0

    last_faces = []
    # The person detector only gates face recognition. Final person events
    # require an InsightFace human-face detection, preventing animals or other
    # YOLO person false positives from becoming unknown-person alerts.
    last_event_people = []
    last_vehicles = []

    known_face_memory = []

    unknown_frame_history = deque(maxlen=5)
    max_unknown_count = 0
    unknown_alert_sent = False
    last_detection_log_time = 0.0

    last_emit_time = 0.0

    emit_interval = (
        1.0 /
        STREAM_TARGET_FPS
    )


    while camera_running:

        try:

            rtsp_url = build_rtsp_url()

            print(
                "[INFO] Connecting to RTSP..."
            )

            cap = cv2.VideoCapture(
                rtsp_url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    10000,
                ]
            )


            if not cap.isOpened():

                camera_status = (
                    "RTSP connection failed"
                )

                print(
                    "[ERROR] Could not open "
                    "RTSP stream."
                )

                time.sleep(5)

                continue


            camera_status = "LIVE"

            print(
                "[OK] RTSP stream connected."
            )

            consecutive_read_failures = 0


            while camera_running:

                ok, frame = cap.read()

                if (
                    not ok
                    or frame is None
                ):

                    consecutive_read_failures += 1

                    if (
                        consecutive_read_failures
                        <
                        MAX_CONSECUTIVE_READ_FAILURES
                    ):

                        time.sleep(0.05)

                        continue

                    camera_status = (
                        "Stream lost - "
                        "reconnecting..."
                    )

                    print(
                        "[WARNING] Frame read "
                        "failed repeatedly."
                    )

                    break


                consecutive_read_failures = 0

                frame_counter += 1

                faces_payload = []
                vehicles_payload = []


                # ------------------------------------------------
                # FACE DETECTION
                # ------------------------------------------------

                if (
                    frame_counter
                    %
                    PROCESS_EVERY_N_FRAMES
                    == 0
                ):

                    try:

                        previous_faces = last_faces

                        person_boxes = detect_person_boxes(frame)

                        if (
                            frame_counter
                            %
                            VEHICLE_PROCESS_EVERY_N_FRAMES
                            == 0
                        ):
                            detected_vehicles = detect_vehicle_boxes(frame)
                            last_vehicles = [
                                vehicle
                                for vehicle in detected_vehicles
                                if vehicle_in_roi(
                                    vehicle["box"],
                                    frame.shape[1],
                                    frame.shape[0],
                                )
                            ]

                        vehicle_person_boxes = [
                            person_box
                            for person_box in person_boxes
                            if any(
                                boxes_overlap(
                                    person_box,
                                    vehicle["box"],
                                )
                                for vehicle in last_vehicles
                            )
                        ]

                        detected_faces = (
                            face_app.get(frame)
                            if person_boxes
                            else []
                        )

                        last_faces = [
                            face
                            for face in detected_faces
                            if any(
                                face_inside_person(
                                    face.bbox.astype(int),
                                    person_box,
                                )
                                for person_box in person_boxes
                            )
                            and (
                                face_in_roi(
                                    face.bbox.astype(int),
                                    frame.shape[1],
                                    frame.shape[0],
                                )
                                or any(
                                    face_inside_person(
                                        face.bbox.astype(int),
                                        person_box,
                                    )
                                    for person_box in vehicle_person_boxes
                                )
                            )
                        ]

                        now = time.time()

                        seen_unknown_in_frame = False
                        unknown_count_in_frame = 0


                        # ----------------------------------------
                        # Remove old remembered faces
                        # ----------------------------------------

                        known_face_memory[:] = [

                            remembered_face

                            for remembered_face
                            in known_face_memory

                            if (
                                now
                                -
                                remembered_face[
                                    "last_seen"
                                ]
                                <
                                UNKNOWN_GONE_CLEARANCE
                            )
                        ]


                        # ----------------------------------------
                        # Process faces
                        # ----------------------------------------

                        for face in last_faces:

                            (
                                name,
                                score
                            ) = recognize_face(
                                face
                            )

                            current_box = (
                                face.bbox.astype(int)
                            )
                            face_is_in_roi = face_in_roi(
                                current_box,
                                frame.shape[1],
                                frame.shape[0],
                            )


                            # ------------------------------------
                            # Previous frame matching
                            # ------------------------------------

                            if name == "Unknown":

                                for previous_face in previous_faces:

                                    previous_name = getattr(
                                        previous_face,
                                        "recognized_name",
                                        "Unknown"
                                    )

                                    if (
                                        previous_name
                                        !=
                                        "Unknown"
                                        and
                                        boxes_overlap(
                                            current_box,
                                            previous_face.bbox.astype(
                                                int
                                            )
                                        )
                                    ):

                                        name = previous_name

                                        score = getattr(
                                            previous_face,
                                            "recognition_score",
                                            score
                                        )

                                        break


                            # ------------------------------------
                            # Memory matching
                            # ------------------------------------

                            if name == "Unknown":

                                for remembered_face in known_face_memory:

                                    if boxes_overlap(
                                        current_box,
                                        remembered_face["box"]
                                    ):

                                        name = remembered_face[
                                            "name"
                                        ]

                                        score = remembered_face[
                                            "score"
                                        ]

                                        remembered_face[
                                            "box"
                                        ] = current_box

                                        remembered_face[
                                            "last_seen"
                                        ] = now

                                        break


                            # ------------------------------------
                            # Known face memory
                            # ------------------------------------

                            if name != "Unknown":

                                matching_memory = next(
                                    (
                                        remembered_face

                                        for remembered_face
                                        in known_face_memory

                                        if (
                                            remembered_face[
                                                "name"
                                            ]
                                            ==
                                            name
                                            and
                                            boxes_overlap(
                                                current_box,
                                                remembered_face[
                                                    "box"
                                                ]
                                            )
                                        )
                                    ),
                                    None
                                )


                                if matching_memory is None:

                                    known_face_memory.append(
                                        {
                                            "name": name,
                                            "score": score,
                                            "box": current_box,
                                            "last_seen": now
                                        }
                                    )

                                else:

                                    matching_memory[
                                        "box"
                                    ] = current_box

                                    matching_memory[
                                        "score"
                                    ] = score

                                    matching_memory[
                                        "last_seen"
                                    ] = now


                            face.recognized_name = name

                            face.recognition_score = score
                            face.in_roi = face_is_in_roi

                            face_detection_score = float(
                                getattr(face, "det_score", 0.0)
                            )


                            if (
                                name == "Unknown"
                                and face_is_in_roi
                                and face_detection_score
                                >= UNKNOWN_FACE_MIN_SCORE
                            ):

                                seen_unknown_in_frame = True
                                unknown_count_in_frame += 1

                        roi_person_boxes = [
                            person_box
                            for person_box in person_boxes
                            if (
                                face_in_roi(
                                    person_box,
                                    frame.shape[1],
                                    frame.shape[0],
                                )
                                or person_box in vehicle_person_boxes
                            )
                        ]
                        known_person_boxes = [
                            person_box
                            for person_box in roi_person_boxes
                            if any(
                                getattr(
                                    face,
                                    "recognized_name",
                                    "Unknown"
                                ) != "Unknown"
                                and face_inside_person(
                                    face.bbox.astype(int),
                                    person_box,
                                )
                                for face in last_faces
                            )
                        ]
                        last_event_people = []
                        for person_box in roi_person_boxes:
                            matched_face = next(
                                (
                                    face
                                    for face in last_faces
                                    if face_inside_person(
                                        face.bbox.astype(int),
                                        person_box,
                                    )
                                ),
                                None,
                            )
                            if matched_face is not None:
                                last_event_people.append(
                                    SimpleNamespace(
                                        # Keep the body box only for stable
                                        # tracking; API/Firebase annotations
                                        # use annotation_box (the face) only.
                                        bbox=np.array(person_box),
                                        annotation_box=tuple(
                                            int(value)
                                            for value in matched_face.bbox
                                        ),
                                        recognized_name=getattr(
                                            matched_face,
                                            "recognized_name",
                                            "Unknown",
                                        ),
                                        recognition_score=float(getattr(
                                            matched_face,
                                            "recognition_score",
                                            0.0,
                                        )),
                                        person_id=getattr(
                                            matched_face,
                                            "person_id",
                                            None,
                                        ),
                                    )
                                )
                        unmatched_person_count = max(
                            0,
                            len(roi_person_boxes)
                            -
                            len(known_person_boxes)
                        )
                        unknown_vehicle_person_detected = any(
                            person_box in vehicle_person_boxes
                            and person_box not in known_person_boxes
                            for person_box in roi_person_boxes
                        )

                        if unmatched_person_count:
                            seen_unknown_in_frame = True
                            unknown_count_in_frame = max(
                                unknown_count_in_frame,
                                unmatched_person_count,
                            )

                        if now - last_detection_log_time >= 5.0:

                            known_count_in_frame = sum(
                                1
                                for face in last_faces
                                if getattr(
                                    face,
                                    "recognized_name",
                                    "Unknown"
                                ) != "Unknown"
                            )
                            two_wheeler_count = sum(
                                1
                                for vehicle in last_vehicles
                                if vehicle["vehicle_type"] == "two_wheeler"
                            )
                            four_wheeler_count = sum(
                                1
                                for vehicle in last_vehicles
                                if vehicle["vehicle_type"] == "four_wheeler"
                            )

                            if (
                                known_count_in_frame
                                or unknown_count_in_frame
                                or two_wheeler_count
                                or four_wheeler_count
                            ):
                                _save_detection_event(
                                    known_count_in_frame,
                                    unknown_count_in_frame,
                                    two_wheeler_count,
                                    four_wheeler_count,
                                    now
                                )

                            vehicle_labels = ", ".join(
                                f"{vehicle['vehicle_type']}"
                                f" ({vehicle['class_name']}, "
                                f"{vehicle['confidence']:.2f})"
                                for vehicle in last_vehicles
                            ) or "none"

                            print(
                                f"[INFO] Faces detected: "
                                f"{len(detected_faces)}, "
                                f"persons: {len(person_boxes)}, "
                                f"inside ROI: {len(last_faces)}, "
                                f"known: {known_count_in_frame}, "
                                f"unknown: {unknown_count_in_frame}, "
                                f"two_wheeler: {two_wheeler_count}, "
                                f"four_wheeler: {four_wheeler_count}, "
                                f"vehicles: {vehicle_labels}"
                            )

                            for vehicle in last_vehicles:
                                print(
                                    f"[VEHICLE] Vehicle detected: "
                                    f"{vehicle['vehicle_type']} "
                                    f"({vehicle['class_name']}), "
                                    f"confidence: "
                                    f"{vehicle['confidence']:.2f}"
                                )

                            last_detection_log_time = now


                        # ----------------------------------------
                        # Unknown person presence
                        # ----------------------------------------

                        if seen_unknown_in_frame:

                            if not unknown_present:

                                unknown_present = True

                                unknown_first_seen = now

                            unknown_frame_history.append(
                                annotate_alert_frame(
                                    frame,
                                    last_faces,
                                )
                            )
                            max_unknown_count = max(
                                max_unknown_count,
                                unknown_count_in_frame,
                            )

                            if unknown_present:

                                unknown_last_seen = now

                                if (
                                    not unknown_alert_sent
                                    and (
                                        unknown_vehicle_person_detected
                                        or now - unknown_first_seen
                                        >= MIN_UNKNOWN_PRESENCE
                                    )
                                ):

                                    if (
                                        now - last_alert_time
                                        >= ALERT_DEDUP_TIME
                                    ):

                                        print(
                                            "[ALERT] Unknown person "
                                            "detected in ROI."
                                        )

                                        camera_status = (
                                            "Unknown person detected"
                                        )

                                        print(
                                            "[INFO] Unknown event will be "
                                            "saved by the detection manager."
                                        )

                                        last_alert_time = now

                                    else:

                                        print(
                                            "[INFO] Alert suppressed "
                                            "(duplicate within "
                                            f"{ALERT_DEDUP_TIME}s window)"
                                        )

                                    unknown_alert_sent = True


                        # ----------------------------------------
                        # Person left frame
                        # ----------------------------------------

                        elif (
                            unknown_present
                            and
                            now
                            -
                            unknown_last_seen
                            >=
                            UNKNOWN_GONE_CLEARANCE
                        ):

                            unknown_present = False

                            unknown_first_seen = 0.0

                            unknown_last_seen = 0.0

                            unknown_frame_history.clear()

                            max_unknown_count = 0

                            unknown_alert_sent = False


                    except Exception as e:

                        print(
                            "[ERROR] Face processing "
                            f"error: {e}"
                        )


                # ------------------------------------------------
                # FACE PAYLOAD
                # ------------------------------------------------

                height, width = frame.shape[:2]

                for vehicle in last_vehicles:
                    x1, y1, x2, y2 = vehicle["box"]
                    vehicles_payload.append(
                        {
                            "type": vehicle["vehicle_type"],
                            "class_name": vehicle["class_name"],
                            "confidence": round(vehicle["confidence"], 3),
                            "box": {
                                "x1": int(max(0, min(x1, width - 1))),
                                "y1": int(max(0, min(y1, height - 1))),
                                "x2": int(max(0, min(x2, width - 1))),
                                "y2": int(max(0, min(y2, height - 1))),
                            },
                        }
                    )

                for face in last_faces:

                    x1, y1, x2, y2 = (
                        face.bbox.astype(int)
                    )

                    faces_payload.append(
                        {
                            "name": getattr(
                                face,
                                "recognized_name",
                                "Unknown"
                            ),

                            "score": round(
                                float(
                                    getattr(
                                        face,
                                        "recognition_score",
                                        0.0
                                    )
                                ),
                                3
                            ),

                            "box": {
                                "x1": int(
                                    max(
                                        0,
                                        min(
                                            x1,
                                            width - 1
                                        )
                                    )
                                ),

                                "y1": int(
                                    max(
                                        0,
                                        min(
                                            y1,
                                            height - 1
                                        )
                                    )
                                ),

                                "x2": int(
                                    max(
                                        0,
                                        min(
                                            x2,
                                            width - 1
                                        )
                                    )
                                ),

                                "y2": int(
                                    max(
                                        0,
                                        min(
                                            y2,
                                            height - 1
                                        )
                                    )
                                )
                            }
                        }
                    )


                # ------------------------------------------------
                # ANNOTATED FRAME
                # ------------------------------------------------

                now = time.time()


                if (
                    now
                    -
                    last_emit_time
                    >=
                    emit_interval
                ):

                    last_emit_time = now

                    send_frame = frame.copy()


                    for face in last_faces:

                        x1, y1, x2, y2 = (
                            face.bbox.astype(int)
                        )

                        name = getattr(
                            face,
                            "recognized_name",
                            "Unknown"
                        )

                        score = float(
                            getattr(
                                face,
                                "recognition_score",
                                0.0
                            )
                        )


                        x1 = max(
                            0,
                            min(
                                x1,
                                width - 1
                            )
                        )

                        y1 = max(
                            0,
                            min(
                                y1,
                                height - 1
                            )
                        )

                        x2 = max(
                            0,
                            min(
                                x2,
                                width - 1
                            )
                        )

                        y2 = max(
                            0,
                            min(
                                y2,
                                height - 1
                            )
                        )


                        if name == "Unknown":

                            color = (
                                0,
                                0,
                                255
                            )

                            label = "Unknown"

                        else:

                            color = (
                                0,
                                255,
                                0
                            )

                            label = (
                                f"{name} "
                                f"{score:.2f}"
                            )


                        cv2.rectangle(
                            send_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )


                        (
                            text_width,
                            text_height
                        ), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            2
                        )


                        label_y1 = max(
                            0,
                            y1
                            -
                            text_height
                            -
                            baseline
                            -
                            8
                        )


                        label_y2 = max(
                            text_height
                            +
                            baseline
                            +
                            8,
                            y1
                        )


                        cv2.rectangle(
                            send_frame,
                            (x1, label_y1),
                            (
                                x1
                                +
                                text_width
                                +
                                10,
                                label_y2
                            ),
                            color,
                            -1
                        )


                        cv2.putText(
                            send_frame,
                            label,
                            (
                                x1 + 5,
                                label_y2 - 6
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (
                                255,
                                255,
                                255
                            ),
                            2,
                            cv2.LINE_AA
                        )


                    for vehicle in last_vehicles:
                        x1, y1, x2, y2 = vehicle["box"]
                        x1 = max(0, min(int(x1), width - 1))
                        y1 = max(0, min(int(y1), height - 1))
                        x2 = max(0, min(int(x2), width - 1))
                        y2 = max(0, min(int(y2), height - 1))
                        label = (
                            f"Vehicle detected: "
                            f"{vehicle['vehicle_type']} "
                            f"{vehicle['confidence']:.2f}"
                        )
                        color = (
                            (255, 165, 0)
                            if vehicle["vehicle_type"] == "two_wheeler"
                            else (255, 0, 0)
                        )

                        cv2.rectangle(
                            send_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )
                        cv2.putText(
                            send_frame,
                            label,
                            (x1 + 5, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                            cv2.LINE_AA
                        )


                    if detection_events is not None:
                        detection_events.process_frame(
                            frame=send_frame,
                            faces=last_event_people,
                            vehicles=last_vehicles,
                            gate_name="Main Gate 01",
                            detected_at=now,
                            alert_frame=send_frame,
                        )


                    # --------------------------------------------
                    # Resize
                    # --------------------------------------------

                    if width > STREAM_MAX_WIDTH:

                        scale = (
                            STREAM_MAX_WIDTH
                            /
                            float(width)
                        )

                        send_frame = cv2.resize(
                            send_frame,
                            (
                                STREAM_MAX_WIDTH,
                                int(
                                    height * scale
                                )
                            )
                        )


                    # --------------------------------------------
                    # JPEG encode
                    # --------------------------------------------

                    success, encoded = cv2.imencode(
                        ".jpg",
                        send_frame,
                        [
                            cv2.IMWRITE_JPEG_QUALITY,
                            STREAM_JPEG_QUALITY
                        ]
                    )


                    if success:

                        encoded_bytes = (
                            encoded.tobytes()
                        )


                        with latest_frame_lock:

                            latest_annotated_frame = (
                                encoded_bytes
                            )


                        # ----------------------------------------
                        # Socket.IO live frame
                        # ----------------------------------------

                        socketio.emit(
                            "face_frame",
                            {
                                "image":
                                    base64.b64encode(
                                        encoded_bytes
                                    ).decode("utf-8"),

                                "width":
                                    send_frame.shape[1],

                                "height":
                                    send_frame.shape[0],

                                "faces":
                                    faces_payload,

                                "vehicles":
                                    vehicles_payload,

                                "status":
                                    camera_status,

                                "timestamp":
                                    now
                            }
                        )


            cap.release()


        except Exception as e:

            camera_status = (
                f"Error: {e}"
            )

            print(
                f"[ERROR] Camera worker: {e}"
            )

            time.sleep(5)


# ============================================================
# ANDROID PERSON REGISTRATION
# ============================================================

def _validate_registration_images():
    """Decode five uploads and create compatible InsightFace embeddings."""

    expected = [f"image{number}" for number in range(1, 6)]
    supplied = [
        field
        for field in request.files
        if field.lower().startswith("image")
    ]
    if set(supplied) != set(expected) or any(
        len(request.files.getlist(field)) != 1
        for field in expected
    ):
        raise ValueError("Exactly 5 face images are required")
    if face_app is None:
        raise RuntimeError("Face recognition model is not ready")

    images, embeddings = [], []
    for number, field in enumerate(expected, start=1):
        raw = request.files[field].read()
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        ) if raw else None
        if image is None:
            raise ValueError(f"Image {number} is not a valid image")
        faces = face_app.get(image)
        if not faces:
            raise ValueError(f"No face detected in image {number}")
        if len(faces) != 1:
            raise ValueError(f"Image {number} must contain exactly one face")
        embedding = normalize_embedding(faces[0].embedding)
        if embedding.size == 0:
            raise ValueError(
                f"Unable to generate face embedding for image {number}"
            )
        images.append(image)
        embeddings.append(embedding)
    return images, embeddings


def _save_registered_person(gate_no, employee_name, designation, employee_id,
                            images, embeddings):
    """Persist a completely validated registration and update the live cache."""

    registration_id = f"REG_{uuid.uuid4().hex[:12].upper()}"
    storage_key = employee_id or registration_id
    final_dir = KNOWN_FACES_DIR / storage_key
    staging_dir = KNOWN_FACES_DIR / (
        f".{storage_key}.registration-{uuid.uuid4().hex}"
    )
    moved_to_final = False
    conn = None
    try:
        conn = _open_db()
        _ensure_table(conn)
        if employee_id and conn.execute(
            "SELECT 1 FROM registered_persons WHERE employee_id = ?",
            (employee_id,),
        ).fetchone():
            raise ValueError("Employee ID already registered")
        if final_dir.exists():
            raise ValueError("Employee registration folder already exists")

        staging_dir.mkdir(parents=True, exist_ok=False)
        image_paths = []
        for number, image in enumerate(images, start=1):
            image_path = staging_dir / f"image_{number}.jpg"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Unable to save image {number}")
            image_paths.append(image_path)

        # Move only after every image was safely written. The final directory
        # uses a server-generated key, never an Android filename.
        staging_dir.replace(final_dir)
        moved_to_final = True
        final_paths = [final_dir / path.name for path in image_paths]
        fingerprint = _compute_fingerprint(final_paths)
        average_embedding = normalize_embedding(np.mean(
            np.stack(embeddings, axis=0), axis=0,
        ))
        now = time.time()

        conn.execute(
            """
            INSERT INTO registered_persons (
                registration_id, storage_key, gate_no, employee_id,
                employee_name, designation, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                registration_id, storage_key, gate_no, employee_id,
                employee_name, designation, now, now,
            ),
        )
        for number, (path, embedding) in enumerate(
            zip(final_paths, embeddings), start=1,
        ):
            conn.execute(
                """
                INSERT INTO registered_face_embeddings (
                    registration_id, image_number, image_path, embedding,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    registration_id, number, str(path),
                    sqlite3.Binary(_embedding_blob(embedding)), now,
                ),
            )
        # This is the exact existing known_embeddings storage format used by
        # load_known_faces()/recognize_face(), not a second recognition store.
        conn.execute(
            """
            REPLACE INTO known_embeddings (
                person, embedding, image_paths, fingerprint, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                storage_key, sqlite3.Binary(_embedding_blob(average_embedding)),
                json.dumps([str(path) for path in final_paths]), fingerprint, now,
            ),
        )
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        if moved_to_final:
            shutil.rmtree(final_dir, ignore_errors=True)
        else:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        if conn is not None:
            conn.close()

    # Hot-add the normalized average to the live existing recognition cache;
    # no model reload or RTSP interruption is needed.
    known_embeddings[storage_key] = average_embedding
    known_person_metadata[storage_key] = {
        "employee_name": employee_name,
        "employee_id": employee_id,
        "designation": designation,
        "gate_no": gate_no,
    }
    return registration_id


@app.route("/api/register-person", methods=["POST"])
def api_register_person():
    required_fields = ("gate_no", "employee_name", "designation")
    values = {
        field: request.form.get(field, "").strip()
        for field in required_fields
    }
    for field in required_fields:
        if not values[field]:
            return jsonify({
                "success": False,
                "message": f"{field} is required",
            }), 400
    employee_id = request.form.get("employee_id", "").strip() or None
    try:
        images, embeddings = _validate_registration_images()
        registration_id = _save_registered_person(
            values["gate_no"], values["employee_name"],
            values["designation"], employee_id, images, embeddings,
        )
    except ValueError as error:
        return jsonify({"success": False, "message": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"success": False, "message": str(error)}), 503
    except Exception as error:
        print(f"[ERROR] Person registration failed: {error}")
        return jsonify({
            "success": False,
            "message": "Unable to register employee",
        }), 500
    return jsonify({
        "success": True,
        "message": "Employee registered successfully",
        "registration_id": registration_id,
        "gate_no": values["gate_no"],
        "employee_id": employee_id,
        "employee_name": values["employee_name"],
        "designation": values["designation"],
        "images_registered": 5,
    }), 201


# ============================================================
# SERVE ALERT IMAGES
# ============================================================

@app.route(
    "/alerts/<filename>",
    methods=["GET"]
)
def alert_image(filename):

    return send_from_directory(
        ALERT_IMAGE_DIR,
        filename
    )


# ============================================================
# REST API
# ============================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def api_status():

    return jsonify(
        {
            "status": camera_status,
            "connected_clients":
                connected_clients,
            "known_faces_count":
                len(known_embeddings)
        }
    )


@app.route(
    "/api/mobile",
    methods=["GET"]
)
def api_mobile():
    """Return the stable, combined response used by the Android app."""
    limit = request.args.get(
        "limit",
        # No limit means the Android login receives the complete persisted
        # detection-event history, not only the current phone's cache.
        default=0,
        type=int
    )
    limit = max(0, limit)
    detection_type = request.args.get("type", default="").strip().lower()
    valid_detection_types = {
        "known_person",
        "unknown_person",
        "vehicle",
    }
    if detection_type not in valid_detection_types:
        detection_type = ""
    include_snapshot = request.args.get(
        "include_snapshot",
        default="true"
    ).lower() in {"1", "true", "yes"}

    # Read the persisted master history first so summary totals remain correct
    # for every mobile installation. Filtering changes only the returned list.
    all_detections = detection_database.latest(0)
    detections = [
        item
        for item in all_detections
        if not detection_type or item["type"] == detection_type
    ]
    if limit:
        detections = detections[:limit]
    detection_summary = {
        "total_events": len(all_detections),
        "known_events": sum(
            item["type"] == "known_person"
            for item in all_detections
        ),
        "unknown_events": sum(
            item["type"] == "unknown_person"
            for item in all_detections
        ),
        "unknown_people": sum(
            int(item.get("unknown_count") or 1)
            for item in all_detections
            if item["type"] == "unknown_person"
        ),
        "vehicle_events": sum(
            item["type"] == "vehicle"
            for item in all_detections
        ),
    }
    alert_limit = len(alert_history) if limit == 0 else min(limit, len(alert_history))
    alerts = list(alert_history)[-alert_limit:]
    alerts.reverse()

    gates = sorted({
        item["gate"]
        for item in detections
        if item.get("gate")
    } | {"Main Gate 01"})

    snapshot = None
    if include_snapshot:
        with latest_frame_lock:
            frame = latest_annotated_frame
        if frame is not None:
            snapshot = {
                "mime_type": "image/jpeg",
                "base64": base64.b64encode(frame).decode("utf-8"),
            }

    return jsonify(
        {
            "success": True,
            "version": 1,
            "data": {
                "status": camera_status,
                "connected_clients": connected_clients,
                "known_faces_count": len(known_embeddings),
                "people": sorted(known_embeddings.keys()),
                "alerts": alerts,
                "detections": detections,
                "detection_summary": detection_summary,
                "snapshot": snapshot,
            },
            "options": {
                "gates": gates,
                "people": sorted(known_embeddings.keys()),
                "detection_types": [
                    "known_person",
                    "unknown_person",
                    "vehicle",
                ],
                "vehicle_types": [
                    "two_wheeler",
                    "four_wheeler",
                ],
            },
        }
    )


@app.route(
    "/api/known-faces",
    methods=["GET"]
)
def api_known_faces():

    return jsonify(
        {
            "people":
                sorted(
                    known_embeddings.keys()
                )
        }
    )


@app.route(
    "/api/alerts",
    methods=["GET"]
)
def api_alerts():

    limit = request.args.get(
        "limit",
        default=20,
        type=int
    )

    items = list(
        alert_history
    )[-limit:]

    items.reverse()

    return jsonify(
        {
            "alerts": items
        }
    )


@app.route(
    "/api/detections",
    methods=["GET"]
)
def api_detections():

    limit = request.args.get(
        "limit",
        default=0,
        type=int
    )

    return jsonify(
        {
            "detections": detection_database.latest(limit)
        }
    )


@app.route(
    "/api/snapshot",
    methods=["GET"]
)
def api_snapshot():

    with latest_frame_lock:

        frame = latest_annotated_frame


    if frame is None:

        return jsonify(
            {
                "error":
                    "No frame available yet"
            }
        ), 503


    b64_frame = base64.b64encode(
        frame
    ).decode("utf-8")


    return jsonify(
        {
            "image": b64_frame,
            "status": camera_status
        }
    )


# ============================================================
# MANUAL TEST NOTIFICATION
# ============================================================

@app.route(
    "/api/test-notification",
    methods=["POST"]
)
def test_notification():

    with latest_frame_lock:

        frame = latest_annotated_frame


    if frame is None:

        return jsonify(
            {
                "success": False,
                "error":
                    "No camera frame available"
            }
        ), 503


    push_alert(
        "New person detected "
        "on Main Gate 01",
        frame,
        "Main Gate 01"
    )


    return jsonify(
        {
            "success": True,
            "message":
                "Test notification sent"
        }
    )


# ============================================================
# SOCKET.IO EVENTS
# ============================================================

@socketio.on("connect")
def on_connect():

    global connected_clients

    connected_clients += 1

    print(
        "[INFO] Client connected. "
        f"Total: {connected_clients}"
    )


@socketio.on("disconnect")
def on_disconnect():

    global connected_clients

    connected_clients = max(
        0,
        connected_clients - 1
    )

    print(
        "[INFO] Client disconnected. "
        f"Total: {connected_clients}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        " ALCON CAMERA 1 - "
        "FACE RECOGNITION"
    )

    print(
        " FCM IMAGE NOTIFICATION ENABLED"
    )

    print("=" * 60)


    # ----------------------------------------
    # Firebase
    # ----------------------------------------

    initialize_firebase()


    # ----------------------------------------
    # Face model
    # ----------------------------------------

    configure_h264_stream()

    initialize_face_model()
    initialize_detection_events()


    # ----------------------------------------
    # Known faces
    # ----------------------------------------

    load_known_faces()


    # ----------------------------------------
    # Camera worker
    # ----------------------------------------

    worker = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    worker.start()


    print()

    print(
        f"[INFO] REST API base: "
        f"http://<server-ip>:{PORT}/api/"
    )

    print(
        f"[INFO] Alert images: "
        f"{PUBLIC_BASE_URL}/alerts/"
    )

    print(
        "[INFO] FCM topic: "
        f"{FCM_TOPIC}"
    )

    print(
        "[INFO] Socket.IO events: "
        "'face_frame', 'face_alert'"
    )

    print(
        "[INFO] Press CTRL+C "
        "to stop."
    )

    print()


    socketio.run(
        app,
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
