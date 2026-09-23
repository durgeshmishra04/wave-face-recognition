"""Environment-backed configuration for the independent fall KPI."""

import os

# FALL_DETECTION_ENABLED is the deployment-facing global switch. FALL_ENABLED
# remains the existing package-level alias used by the pipeline.
FALL_DETECTION_ENABLED = os.getenv(
    "FALL_DETECTION_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}
FALL_ENABLED = FALL_DETECTION_ENABLED
FALL_MODEL_PATH = os.getenv("FALL_MODEL_PATH", "yolo11m-pose.pt")
FALL_CONFIDENCE = float(os.getenv("FALL_CONFIDENCE", "0.65"))
FALL_POSE_CONFIDENCE = float(os.getenv("FALL_POSE_CONFIDENCE", "0.65"))
FALL_CONFIRMATION_FRAMES = max(2, int(os.getenv("FALL_CONFIRMATION_FRAMES", "2")))
FALL_RECOVERY_FRAMES = max(2, int(os.getenv("FALL_RECOVERY_FRAMES", "5")))
FALL_MIN_NORMAL_FRAMES = max(1, int(os.getenv("FALL_MIN_NORMAL_FRAMES", "2")))
# A camera can first observe a person after they have already fallen. Keep
# temporal confirmation, but do not require an unseen upright transition.
# Set this true only to restore the legacy transition-only behaviour.
FALL_REQUIRE_UPRIGHT_TRANSITION = os.getenv(
    "FALL_REQUIRE_UPRIGHT_TRANSITION", "false"
).strip().lower() in {"1", "true", "yes", "on"}
FALL_HISTORY_SIZE = max(4, int(os.getenv("FALL_HISTORY_SIZE", "12")))
FALL_TRACK_TIMEOUT_SECONDS = max(1.0, float(os.getenv("FALL_TRACK_TIMEOUT_SECONDS", "10")))
FALL_MIN_IOU = float(os.getenv("FALL_MIN_IOU", "0.25"))
FALL_HORIZONTAL_ASPECT = float(os.getenv("FALL_HORIZONTAL_ASPECT", "0.90"))
FALL_TORSO_ANGLE_DEGREES = float(os.getenv("FALL_TORSO_ANGLE_DEGREES", "55"))
FALL_UPRIGHT_ASPECT = float(os.getenv("FALL_UPRIGHT_ASPECT", "0.75"))
FALL_UPRIGHT_ANGLE_DEGREES = float(os.getenv("FALL_UPRIGHT_ANGLE_DEGREES", "40"))
