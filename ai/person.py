"""Compatibility exports for person detection."""

from alcon_stream_new import detect_person_boxes

detect_persons = detect_person_boxes

__all__ = ["detect_person_boxes", "detect_persons"]
