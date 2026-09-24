"""Environment-backed configuration for the Smoke/Fire KPI."""

import os

# FIRE_DETECTION_ENABLED is the deployment-facing global switch. Keep the
# existing SMOKE_FIRE_ENABLED name as a backwards-compatible alias.
FIRE_DETECTION_ENABLED = os.getenv(
    "FIRE_DETECTION_ENABLED",
    os.getenv("SMOKE_FIRE_ENABLED", "true"),
).strip().lower() in {"1", "true", "yes", "on"}
SMOKE_FIRE_ENABLED = FIRE_DETECTION_ENABLED
SMOKE_FIRE_TEST_MODE = os.getenv("SMOKE_FIRE_TEST_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
SMOKE_FIRE_MODEL_PATH = os.getenv("SMOKE_FIRE_MODEL_PATH", "salahyolo26.pt")
SMOKE_CONFIDENCE_THRESHOLD = float(os.getenv("SMOKE_CONFIDENCE_THRESHOLD", "0.95"))
FIRE_CONFIDENCE_THRESHOLD = float(os.getenv("FIRE_CONFIDENCE_THRESHOLD", "0.95"))
SMOKE_FIRE_IOU = float(os.getenv("SMOKE_FIRE_IOU", "0.45"))
SMOKE_FIRE_CONFIRM_FRAMES = max(2, int(os.getenv("SMOKE_FIRE_CONFIRM_FRAMES", "3")))
SMOKE_FIRE_CONFIRM_SECONDS = max(0.0, float(os.getenv("SMOKE_FIRE_CONFIRM_SECONDS", "0")))
SMOKE_FIRE_CLEAR_FRAMES = max(2, int(os.getenv("SMOKE_FIRE_CLEAR_FRAMES", "10")))
SMOKE_FIRE_CLEAR_SECONDS = max(0.0, float(os.getenv("SMOKE_FIRE_CLEAR_SECONDS", "3.0")))
SMOKE_FIRE_ASSOCIATION_IOU = float(os.getenv("SMOKE_FIRE_ASSOCIATION_IOU", "0.15"))
SMOKE_FIRE_ASSOCIATION_DISTANCE_RATIO = float(os.getenv("SMOKE_FIRE_ASSOCIATION_DISTANCE_RATIO", "1.5"))
