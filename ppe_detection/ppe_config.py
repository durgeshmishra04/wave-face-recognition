"""Environment-backed configuration for the PPE annotation KPI."""

import os

PPE_ENABLED = os.getenv("PPE_ENABLED", os.getenv("PPE_DETECTION_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
PPE_MODEL_PATH = os.getenv("PPE_MODEL_PATH", "yolo26n_ppe.pt")
PPE_CONFIDENCE = float(os.getenv("PPE_CONFIDENCE", "0.35"))
PPE_IOU = float(os.getenv("PPE_IOU", "0.45"))
HELMET_ENABLED = os.getenv("HELMET_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
GLOVES_ENABLED = os.getenv("GLOVES_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
BOOTS_ENABLED = os.getenv("BOOTS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
HELMET_CONFIDENCE = float(os.getenv("HELMET_CONFIDENCE", str(PPE_CONFIDENCE)))
GLOVES_CONFIDENCE = float(os.getenv("GLOVES_CONFIDENCE", str(PPE_CONFIDENCE)))
BOOTS_CONFIDENCE = float(os.getenv("BOOTS_CONFIDENCE", str(PPE_CONFIDENCE)))
PPE_HELMET_CONFIRM_FRAMES = max(1, int(os.getenv("PPE_HELMET_CONFIRM_FRAMES", "3")))
PPE_HELMET_CLEAR_FRAMES = max(1, int(os.getenv("PPE_HELMET_CLEAR_FRAMES", "10")))
PPE_ANNOTATION_ENABLED = os.getenv("PPE_ANNOTATION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
PPE_TRACK_TIMEOUT_SECONDS = max(1.0, float(os.getenv("PPE_TRACK_TIMEOUT_SECONDS", "10")))
PPE_PERSON_MARGIN = max(0.0, float(os.getenv("PPE_PERSON_MARGIN", "0.05")))

# Model labels are normalized before this lookup. Unknown classes are ignored;
# this KPI never claims support for categories absent from the loaded model.
PPE_CLASS_LABELS = {
    "helmet": "Helmet", "hardhat": "Helmet", "hard_hat": "Helmet",
    "vest": "Safety Vest", "safety_vest": "Safety Vest",
    "glove": "Gloves", "gloves": "Gloves",
    "mask": "Mask", "face_mask": "Mask",
    "boot": "Safety Boots", "boots": "Safety Boots", "safety_boots": "Safety Boots",
    "safety_shoe": "Safety Boots", "safety_shoes": "Safety Boots",
}

# Only these positive PPE classes are processed by this KPI.  In particular,
# `no_helmet` (if present in a model) is deliberately not a helmet detection.
PPE_POSITIVE_CLASSES = {
    "helmet": "Helmet", "hardhat": "Helmet", "hard_hat": "Helmet",
    "glove": "Gloves", "gloves": "Gloves",
    "boot": "Safety Boots", "boots": "Safety Boots", "safety_boots": "Safety Boots",
}
