"""Project-root paths and environment-backed settings for new modules."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT", "./pythonai.json")
