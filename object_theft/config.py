"""Environment-backed configuration for the isolated object theft KPI."""

import json
import os


def _env_bool(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_targets(raw_value):
    if raw_value is None:
        return ["drum", "drums", "barrel", "barrels", "container", "containers", "metal drum", "metal barrel", "industrial drum", "industrial barrel", "storage drum", "storage barrel"]
    if isinstance(raw_value, (list, tuple, set)):
        result = [str(item).strip() for item in raw_value if str(item).strip()]
        return result or ["drum"]
    candidates = [item.strip() for item in str(raw_value).split(",")]
    return [item for item in candidates if item] or ["drum"]


def _parse_cameras(raw_value):
    if raw_value is None:
        return {"CAM008"}
    if isinstance(raw_value, (list, tuple, set)):
        items = raw_value
    else:
        items = str(raw_value).split(",")
    return {str(item).strip().upper() for item in items if str(item).strip()}


OBJECT_THEFT_ENABLED = _env_bool("OBJECT_THEFT_ENABLED", "true")
OBJECT_THEFT_CAMERAS = _parse_cameras(os.getenv("OBJECT_THEFT_CAMERAS", "CAM008"))
OBJECT_THEFT_TARGETS = _parse_targets(os.getenv("OBJECT_THEFT_TARGETS", "drum,drums,barrel,barrels,container,containers,metal drum,metal barrel,industrial drum,industrial barrel,storage drum,storage barrel"))
OBJECT_THEFT_MODEL_ID = os.getenv("OBJECT_THEFT_MODEL_ID", "IDEA-Research/grounding-dino-base")
OBJECT_THEFT_MODEL_PATH = os.getenv("OBJECT_THEFT_MODEL_PATH", "")
OBJECT_THEFT_LOCAL_FILES_ONLY = _env_bool("OBJECT_THEFT_LOCAL_FILES_ONLY", "false")
OBJECT_THEFT_CONFIDENCE_THRESHOLD = float(os.getenv("OBJECT_THEFT_CONFIDENCE_THRESHOLD", "0.35"))
OBJECT_THEFT_DETECTION_CONFIDENCE = float(os.getenv("OBJECT_THEFT_DETECTION_CONFIDENCE", os.getenv("OBJECT_THEFT_CONFIDENCE_THRESHOLD", "0.35")))
OBJECT_THEFT_IOU_THRESHOLD = float(os.getenv("OBJECT_THEFT_IOU_THRESHOLD", "0.30"))
OBJECT_THEFT_TRACK_IOU = float(os.getenv("OBJECT_THEFT_TRACK_IOU", "0.30"))
OBJECT_THEFT_MOVEMENT_THRESHOLD = float(os.getenv("OBJECT_THEFT_MOVEMENT_THRESHOLD", "50.0"))
OBJECT_THEFT_CONFIRM_FRAMES = max(1, int(os.getenv("OBJECT_THEFT_CONFIRM_FRAMES", "5")))
OBJECT_THEFT_DISAPPEARANCE_GRACE_FRAMES = max(1, int(os.getenv("OBJECT_THEFT_DISAPPEARANCE_GRACE_FRAMES", "15")))
OBJECT_THEFT_MAX_MISSED_FRAMES = max(1, int(os.getenv("OBJECT_THEFT_MAX_MISSED_FRAMES", os.getenv("OBJECT_THEFT_DISAPPEARANCE_GRACE_FRAMES", "15"))))
OBJECT_THEFT_EVIDENCE_OFFSET = int(os.getenv("OBJECT_THEFT_EVIDENCE_OFFSET", "-6"))
OBJECT_THEFT_INFERENCE_INTERVAL = max(1, int(os.getenv("OBJECT_THEFT_INFERENCE_INTERVAL", "2")))
OBJECT_THEFT_ROI_ENABLED = _env_bool("OBJECT_THEFT_ROI_ENABLED", "true")
OBJECT_THEFT_ROI_POINTS = []
try:
    raw_roi = os.getenv("OBJECT_THEFT_ROI_POINTS", "")
    if raw_roi:
        OBJECT_THEFT_ROI_POINTS = json.loads(raw_roi)
except (TypeError, ValueError):
    OBJECT_THEFT_ROI_POINTS = []
