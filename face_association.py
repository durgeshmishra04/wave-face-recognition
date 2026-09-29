"""Strict face-to-person association gate for InsightFace results."""

from __future__ import annotations

import os

import numpy as np

FACE_PERSON_ASSOCIATION_ENABLED = (
    os.getenv("FACE_PERSON_ASSOCIATION_ENABLED", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
FACE_PERSON_MIN_CENTER_INSIDE = (
    os.getenv("FACE_PERSON_MIN_CENTER_INSIDE", "true")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
FACE_PERSON_MIN_OVERLAP = float(os.getenv("FACE_PERSON_MIN_OVERLAP", "0.10"))


def _intersection_area(first_box, second_box):
    first_x1, first_y1, first_x2, first_y2 = map(float, first_box)
    second_x1, second_y1, second_x2, second_y2 = map(float, second_box)

    inter_left = max(first_x1, second_x1)
    inter_top = max(first_y1, second_y1)
    inter_right = min(first_x2, second_x2)
    inter_bottom = min(first_y2, second_y2)

    inter_width = max(inter_right - inter_left, 0.0)
    inter_height = max(inter_bottom - inter_top, 0.0)
    return inter_width * inter_height


def _face_overlap_fraction(face_box, person_box):
    face_area = max((float(face_box[2]) - float(face_box[0])) * (float(face_box[3]) - float(face_box[1])), 1e-6)
    return _intersection_area(face_box, person_box) / face_area


def face_associated_with_person(face_box, person_boxes, frame_shape):
    """Require a valid spatial match to a tracked YOLO person before UX events.

    A face is accepted only when its center sits inside the person box (or it has a
    meaningful overlap), and it remains in the upper/body region rather than the lower
    torso/legs. This rejects poles, cables, lights, reflections, and other false
    night-time objects while keeping genuine face detections for tracked people.
    """
    if not FACE_PERSON_ASSOCIATION_ENABLED:
        return None

    if not person_boxes:
        return None

    face_box = np.asarray(face_box, dtype=np.float32).reshape(4)
    face_width = max(float(face_box[2]) - float(face_box[0]), 0.0)
    face_height = max(float(face_box[3]) - float(face_box[1]), 0.0)
    if face_width <= 0 or face_height <= 0:
        return None

    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None

    min_face_width = max(frame_width * 0.008, 12.0)
    min_face_height = max(frame_height * 0.012, 16.0)
    if face_width < min_face_width or face_height < min_face_height:
        return None

    face_center_x = (float(face_box[0]) + float(face_box[2])) / 2.0
    face_center_y = (float(face_box[1]) + float(face_box[3])) / 2.0

    for person_box in person_boxes:
        person_box = np.asarray(person_box, dtype=np.float32).reshape(4)
        person_width = max(float(person_box[2]) - float(person_box[0]), 1.0)
        person_height = max(float(person_box[3]) - float(person_box[1]), 1.0)
        if person_width <= 0 or person_height <= 0:
            continue

        person_x1, person_y1, person_x2, person_y2 = [float(v) for v in person_box]

        center_inside = (
            person_x1 - person_width * 0.08 <= face_center_x <= person_x2 + person_width * 0.08
            and person_y1 - person_height * 0.08 <= face_center_y <= person_y2 + person_height * 0.08
        )
        overlap = _face_overlap_fraction(face_box, person_box)
        upper_body_limit = person_y1 + person_height * 0.70
        face_in_upper_body = face_center_y <= upper_body_limit
        meaningful_overlap = overlap >= FACE_PERSON_MIN_OVERLAP

        if FACE_PERSON_MIN_CENTER_INSIDE:
            if center_inside and face_in_upper_body:
                return person_box.astype(np.int32)
            if meaningful_overlap and face_in_upper_body:
                return person_box.astype(np.int32)
        elif meaningful_overlap and face_in_upper_body:
            return person_box.astype(np.int32)

    return None
