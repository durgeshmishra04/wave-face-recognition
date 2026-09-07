"""Compatibility exports for face detection and recognition."""

from alcon_stream_new import (
    face_app,
    initialize_face_model,
    load_known_faces,
    recognize_face,
)

__all__ = [
    "face_app",
    "initialize_face_model",
    "load_known_faces",
    "recognize_face",
]
