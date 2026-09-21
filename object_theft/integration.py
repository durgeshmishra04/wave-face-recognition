"""Small integration helpers for the camera worker and startup/bootstrap hooks."""

from .config import OBJECT_THEFT_CAMERAS, OBJECT_THEFT_ENABLED


def should_process_object_theft(camera_id):
    return OBJECT_THEFT_ENABLED and str(camera_id or "").upper() in OBJECT_THEFT_CAMERAS
