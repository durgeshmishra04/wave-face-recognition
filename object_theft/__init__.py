"""Independent object theft KPI package."""

from .config import (
    OBJECT_THEFT_CAMERAS,
    OBJECT_THEFT_CONFIRM_FRAMES,
    OBJECT_THEFT_ENABLED,
    OBJECT_THEFT_MODEL_ID,
    OBJECT_THEFT_TARGETS,
)
from .detector import ObjectTheftDetection, ObjectTheftDetector
from .integration import should_process_object_theft

__all__ = [
    "ObjectTheftDetector",
    "ObjectTheftDetection",
    "OBJECT_THEFT_ENABLED",
    "OBJECT_THEFT_CAMERAS",
    "OBJECT_THEFT_CONFIRM_FRAMES",
    "OBJECT_THEFT_MODEL_ID",
    "OBJECT_THEFT_TARGETS",
    "should_process_object_theft",
]
