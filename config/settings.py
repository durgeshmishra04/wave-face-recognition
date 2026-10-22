"""Project-root paths and environment-backed settings for new modules."""

import os

from dotenv import load_dotenv

from runtime_paths import resolve_runtime_path, runtime_root

PROJECT_ROOT = runtime_root()
load_dotenv(PROJECT_ROOT / ".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "vms_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "vms_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
KNOWN_FACES_DIR = PROJECT_ROOT / "known_faces"
ALERT_IMAGE_DIR = PROJECT_ROOT / "alert_images"
FIREBASE_SERVICE_ACCOUNT = resolve_runtime_path(
    os.getenv("FIREBASE_SERVICE_ACCOUNT", "pythonai.json")
)
