"""Lightweight object tracking and state machine for object theft detection."""

from __future__ import annotations

import math
import time
import uuid

import cv2
import numpy as np

from .config import (
    OBJECT_THEFT_CONFIRM_FRAMES,
    OBJECT_THEFT_MAX_MISSED_FRAMES,
    OBJECT_THEFT_MOVEMENT_THRESHOLD,
    OBJECT_THEFT_TRACK_IOU,
)


class ObjectTheftTracker:
    """Tracks target objects across successive detections while suppressing duplicate theft alerts."""

    def __init__(
        self,
        iou_threshold=OBJECT_THEFT_TRACK_IOU,
        confirm_frames=OBJECT_THEFT_CONFIRM_FRAMES,
        max_missed=OBJECT_THEFT_MAX_MISSED_FRAMES,
        movement_threshold=OBJECT_THEFT_MOVEMENT_THRESHOLD,
    ):
        self.iou_threshold = float(iou_threshold)
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_missed = max(1, int(max_missed))
        self.movement_threshold = float(movement_threshold)
        self.tracks = []
        self._next_track_id = 1

    @staticmethod
    def _center(box):
        return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)

    @staticmethod
    def _iou(a, b):
        left = max(float(a[0]), float(b[0]))
        top = max(float(a[1]), float(b[1]))
        right = min(float(a[2]), float(b[2]))
        bottom = min(float(a[3]), float(b[3]))
        inter = max(0.0, right - left) * max(0.0, bottom - top)
        a_area = max(float(a[2]) - float(a[0]), 0.0) * max(float(a[3]) - float(a[1]), 0.0)
        b_area = max(float(b[2]) - float(b[0]), 0.0) * max(float(b[3]) - float(b[1]), 0.0)
        union = max(a_area + b_area - inter, 0.0)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _inside_roi(box, polygon):
        if polygon is None or len(polygon) < 3:
            return True
        center = ObjectTheftTracker._center(box)
        return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), center, False) >= 0

    def _match_track(self, box, active, roi_polygon=None):
        best = None
        best_score = -1.0
        center = self._center(box)
        for track in active:
            track_box = track["bbox"]
            iou = self._iou(box, track_box)
            dist = math.hypot(center[0] - self._center(track_box)[0], center[1] - self._center(track_box)[1])
            score = iou + max(0.0, 1.0 - dist / 100.0)
            roi_exit_match = (
                roi_polygon is not None
                and track.get("inside_roi", True)
                and not self._inside_roi(box, roi_polygon)
                and dist <= max(80.0, self.movement_threshold * 2.0)
            )
            if iou >= self.iou_threshold or (iou <= 0.0 and dist <= 35.0) or roi_exit_match:
                if score > best_score:
                    best, best_score = track, score
        return best

    def update(self, camera_id, detections, roi_polygon=None, now=None):
        now = time.time() if now is None else now
        for track in self.tracks:
            track["matched_this_frame"] = False

        seen = set()
        confirmed = []

        for detection in detections:
            matched = self._match_track(detection["bbox"], self.tracks, roi_polygon)
            if matched is None:
                matched = {
                    "track_id": self._next_track_id,
                    "camera_id": camera_id,
                    "class_name": detection["class_name"],
                    "canonical_class": detection.get("canonical_class", "DRUM_CONTAINER"),
                    "bbox": tuple(int(v) for v in detection["bbox"]),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "first_seen": now,
                    "last_seen": now,
                    "baseline_bbox": tuple(int(v) for v in detection["bbox"]),
                    "baseline_center": self._center(detection["bbox"]),
                    "baseline_area": max((detection["bbox"][2] - detection["bbox"][0]) * (detection["bbox"][3] - detection["bbox"][1]), 1.0),
                    "inside_roi": self._inside_roi(detection["bbox"], roi_polygon),
                    "center": self._center(detection["bbox"]),
                    "state": "PRESENT",
                    "alert_generated": False,
                    "event_finalized": False,
                    "moved_frames": 0,
                    "missed_frames": 0,
                    "session_id": uuid.uuid4().hex[:12],
                    "matched_this_frame": True,
                    "active": True,
                }
                self.tracks.append(matched)
                self._next_track_id += 1
            else:
                matched["matched_this_frame"] = True
                matched["bbox"] = tuple(int(v) for v in detection["bbox"])
                matched["confidence"] = float(detection.get("confidence", matched.get("confidence", 0.0)))
                matched["center"] = self._center(matched["bbox"])
                previous_roi_state = bool(matched.get("inside_roi", True))
                matched["inside_roi"] = self._inside_roi(matched["bbox"], roi_polygon)
                matched["last_seen"] = now
                matched["camera_id"] = camera_id
                if previous_roi_state and not matched["inside_roi"] and not matched.get("alert_generated", False):
                    matched["state"] = "REMOVAL_CONFIRMED"
                    matched["event_finalized"] = True
                    matched["alert_generated"] = True
                    confirmed.append({
                        "camera_id": camera_id,
                        "track_id": matched["track_id"],
                        "class_name": matched.get("canonical_class", matched.get("class_name", "DRUM_CONTAINER")),
                        "canonical_class": matched.get("canonical_class", "DRUM_CONTAINER"),
                        "bbox": tuple(int(v) for v in matched["bbox"]),
                        "confidence": float(matched.get("confidence", 0.0)),
                        "session_id": matched.get("session_id"),
                        "state": "REMOVAL_CONFIRMED",
                    })

            matched["canonical_class"] = detection.get("canonical_class", matched.get("canonical_class", "DRUM_CONTAINER"))
            seen.add(matched["track_id"])

            center = matched.get("center")
            baseline = matched.get("baseline_center")
            if center and baseline:
                dx = abs(center[0] - baseline[0])
                dy = abs(center[1] - baseline[1])
                if max(dx, dy) > self.movement_threshold:
                    matched["state"] = "MOVED"
                    matched["moved_frames"] = int(matched.get("moved_frames", 0)) + 1
                    if matched["moved_frames"] >= self.confirm_frames:
                        matched["state"] = "REMOVAL_CANDIDATE"
                else:
                    matched["moved_frames"] = 0
                    matched["state"] = "PRESENT"
            else:
                matched["state"] = "PRESENT"

        for track in list(self.tracks):
            if track["track_id"] in seen:
                continue
            track["missed_frames"] = int(track.get("missed_frames", 0)) + 1
            if track.get("alert_generated"):
                track["active"] = False
                continue
            if track.get("missed_frames", 0) >= self.max_missed:
                track["state"] = "REMOVAL_CONFIRMED"
                track["event_finalized"] = True
                track["alert_generated"] = True
                confirmed.append({
                    "camera_id": camera_id,
                    "track_id": track["track_id"],
                    "class_name": track.get("canonical_class", track.get("class_name", "DRUM_CONTAINER")),
                    "canonical_class": track.get("canonical_class", "DRUM_CONTAINER"),
                    "bbox": tuple(int(v) for v in track["bbox"]),
                    "confidence": float(track.get("confidence", 0.0)),
                    "session_id": track.get("session_id"),
                    "state": "REMOVAL_CONFIRMED",
                })
            elif track.get("missed_frames", 0) >= max(1, self.max_missed // 2):
                track["state"] = "REMOVAL_CANDIDATE"
            else:
                track["state"] = "ABSENT"

        self.tracks = [track for track in self.tracks if track.get("active", True)]
        return confirmed
