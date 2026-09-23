"""Environment-backed configuration for the isolated object theft KPI."""

import json
import os


def _env_bool(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_cameras(raw_value):
    return {
        str(item).strip().upper()
        for item in str(raw_value).split(",")
        if str(item).strip()
    }


def _parse_class_names(raw_value):
    return {
        str(item).strip().lower()
        for item in str(raw_value).split(",")
        if str(item).strip()
    }


OBJECT_THEFT_ENABLED = _env_bool("OBJECT_THEFT_ENABLED", "true")
OBJECT_THEFT_CAMERAS = _parse_cameras(os.getenv("OBJECT_THEFT_CAMERAS", "CAM008"))
OBJECT_THEFT_MODEL_PATH = os.getenv("OBJECT_THEFT_MODEL_PATH", "barrel.pt").strip()
OBJECT_THEFT_CONFIDENCE_THRESHOLD = float(
    os.getenv("OBJECT_THEFT_CONFIDENCE_THRESHOLD", "0.55")
)
OBJECT_THEFT_IOU_THRESHOLD = float(
    os.getenv("OBJECT_THEFT_IOU_THRESHOLD", "0.55")
)
OBJECT_THEFT_TRACK_IOU = float(os.getenv("OBJECT_THEFT_TRACK_IOU", "0.30"))
OBJECT_THEFT_MOVEMENT_THRESHOLD = float(
    os.getenv("OBJECT_THEFT_MOVEMENT_THRESHOLD", "50.0")
)
OBJECT_THEFT_CONFIRM_FRAMES = max(
    1, int(os.getenv("OBJECT_THEFT_CONFIRM_FRAMES", "5"))
)
OBJECT_THEFT_MAX_MISSED_FRAMES = max(
    1, int(os.getenv("OBJECT_THEFT_MAX_MISSED_FRAMES", "4"))
)
OBJECT_THEFT_EVIDENCE_OFFSET = int(os.getenv("OBJECT_THEFT_EVIDENCE_OFFSET", "-6"))
OBJECT_THEFT_INFERENCE_INTERVAL = max(
    1, int(os.getenv("OBJECT_THEFT_INFERENCE_INTERVAL", "1"))
)
OBJECT_THEFT_ROI_ENABLED = _env_bool("OBJECT_THEFT_ROI_ENABLED", "true")
OBJECT_THEFT_ROI_POINTS = []
try:
    raw_roi = os.getenv("OBJECT_THEFT_ROI_POINTS", "")
    if raw_roi:
        OBJECT_THEFT_ROI_POINTS = json.loads(raw_roi)
except (TypeError, ValueError):
    OBJECT_THEFT_ROI_POINTS = []
