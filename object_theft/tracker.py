"""Persistent Object Theft tracks and conservative removal confirmation."""

from __future__ import annotations

import math
import time
import uuid

import cv2
import numpy as np

from .config import (
    OBJECT_THEFT_CONFIRM_FRAMES,
    OBJECT_THEFT_MAX_MISSED_FRAMES,
    OBJECT_THEFT_TRACK_IOU,
)


class ObjectTheftTracker:
    """Confirm presence before treating a later disappearance as theft."""

    def __init__(
        self,
        iou_threshold=OBJECT_THEFT_TRACK_IOU,
        confirm_frames=OBJECT_THEFT_CONFIRM_FRAMES,
        max_missed=OBJECT_THEFT_MAX_MISSED_FRAMES,
        movement_threshold=50.0,
    ):
        self.iou_threshold = float(iou_threshold)
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_missed = max(1, int(max_missed))
        self.movement_threshold = float(movement_threshold)
        self.tracks = []
        self._next_track_id = 1

    @staticmethod
    def _center(box):
        return (
            (float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0,
        )

    @staticmethod
    def _iou(a, b):
        left = max(float(a[0]), float(b[0]))
        top = max(float(a[1]), float(b[1]))
        right = min(float(a[2]), float(b[2]))
        bottom = min(float(a[3]), float(b[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area_a = max(float(a[2]) - float(a[0]), 0.0) * max(
            float(a[3]) - float(a[1]), 0.0
        )
        area_b = max(float(b[2]) - float(b[0]), 0.0) * max(
            float(b[3]) - float(b[1]), 0.0
        )
        return intersection / max(area_a + area_b - intersection, 1e-6)

    @classmethod
    def _inside_roi(cls, box, polygon):
        if polygon is None or len(polygon) < 3:
            return True
        return (
            cv2.pointPolygonTest(
                np.asarray(polygon, dtype=np.float32),
                cls._center(box),
                False,
            )
            >= 0
        )

    def _match_track(self, box, active, roi_polygon=None):
        best = None
        best_score = -1.0
        center = self._center(box)
        for track in active:
            if track.get("matched_this_frame") or track.get("event_finalized"):
                continue
            track_box = track["bbox"]
            iou = self._iou(box, track_box)
            distance = math.hypot(
                center[0] - self._center(track_box)[0],
                center[1] - self._center(track_box)[1],
            )
            max_distance = 35.0 + (track.get("missed_frames", 0) * 20.0)
            if track.get("state") == "REMOVAL_CANDIDATE":
                max_distance = max(max_distance, 80.0)
            roi_exit_match = (
                roi_polygon is not None
                and track.get("last_seen_inside_roi", False)
                and not self._inside_roi(box, roi_polygon)
                and distance <= 80.0
            )
            if (
                iou < self.iou_threshold
                and distance > max_distance
                and not roi_exit_match
            ):
                continue
            score = iou + max(0.0, 1.0 - distance / 100.0)
            if score > best_score:
                best, best_score = track, score
        return best

    @staticmethod
    def _event(track, camera_id):
        return {
            "camera_id": camera_id,
            "track_id": track["track_id"],
            "class_name": track.get("class_name", "DRUM_CONTAINER"),
            "canonical_class": track.get("canonical_class", "DRUM_CONTAINER"),
            "bbox": tuple(track["bbox"]),
            "confidence": float(track.get("confidence", 0.0)),
            "session_id": track["session_id"],
            "state": "REMOVAL_CONFIRMED",
            "mask": track.get("mask"),
        }

    def _confirm_removal(self, track, camera_id, confirmed):
        if not track.get("presence_confirmed") or track.get("alert_generated"):
            return
        track["state"] = "REMOVAL_CONFIRMED"
        track["alert_generated"] = True
        track["event_finalized"] = True
        confirmed.append(self._event(track, camera_id))
        track["state"] = "FINALIZED"
        print(
            "[OBJECT-THEFT] REMOVAL CONFIRMED "
            f"camera={camera_id} track={track['track_id']} "
            f"session={track['session_id']}"
        )

    def update(self, camera_id, detections, roi_polygon=None, now=None):
        now = time.time() if now is None else now
        for track in self.tracks:
            track["matched_this_frame"] = False

        seen = set()
        confirmed = []

        for detection in detections:
            matched = self._match_track(
                detection["bbox"], self.tracks, roi_polygon
            )
            if matched is None:
                bbox = tuple(int(value) for value in detection["bbox"])
                inside = self._inside_roi(bbox, roi_polygon)
                matched = {
                    "track_id": self._next_track_id,
                    "camera_id": camera_id,
                    "class_name": detection.get("class_name", "DRUM_CONTAINER"),
                    "canonical_class": detection.get(
                        "canonical_class", "DRUM_CONTAINER"
                    ),
                    "bbox": bbox,
                    "confidence": float(detection.get("confidence", 0.0)),
                    "first_seen": now,
                    "last_seen": now,
                    "last_seen_inside_roi": inside,
                    "inside_roi": inside,
                    "center": self._center(bbox),
                    "baseline_bbox": None,
                    "baseline_center": None,
                    "baseline_area": 0.0,
                    "presence_confirmations": 0,
                    "presence_confirmed": False,
                    "removal_confirmations": 0,
                    "missed_frames": 0,
                    "state": "PRESENCE_CANDIDATE",
                    "alert_generated": False,
                    "event_finalized": False,
                    "session_id": uuid.uuid4().hex[:12],
                    "matched_this_frame": True,
                    "active": True,
                    "mask": detection.get("mask"),
                }
                self.tracks.append(matched)
                self._next_track_id += 1
                if inside:
                    print(
                        "[OBJECT-THEFT] Presence candidate "
                        f"camera={camera_id} track={matched['track_id']}"
                    )
            else:
                matched["matched_this_frame"] = True
                matched["bbox"] = tuple(int(value) for value in detection["bbox"])
                matched["center"] = self._center(matched["bbox"])
                matched["confidence"] = float(
                    detection.get("confidence", matched["confidence"])
                )
                matched["mask"] = detection.get("mask", matched.get("mask"))
                matched["last_seen"] = now
                matched["missed_frames"] = 0
                matched["inside_roi"] = self._inside_roi(
                    matched["bbox"], roi_polygon
                )
                matched["last_seen_inside_roi"] = matched["inside_roi"]

            matched["canonical_class"] = detection.get(
                "canonical_class", matched.get("canonical_class", "DRUM_CONTAINER")
            )
            inside = bool(matched["inside_roi"])
            seen.add(matched["track_id"])

            if not matched.get("presence_confirmed"):
                if inside:
                    matched["presence_confirmations"] += 1
                    if matched["presence_confirmations"] >= self.confirm_frames:
                        matched["presence_confirmed"] = True
                        matched["state"] = "PRESENT"
                        matched["baseline_bbox"] = matched["bbox"]
                        matched["baseline_center"] = matched["center"]
                        matched["baseline_area"] = max(
                            (matched["bbox"][2] - matched["bbox"][0])
                            * (matched["bbox"][3] - matched["bbox"][1]),
                            1.0,
                        )
                        print(
                            "[OBJECT-THEFT] PRESENT / BASELINE CONFIRMED "
                            f"camera={camera_id} track={matched['track_id']} "
                            f"session={matched['session_id']}"
                        )
                else:
                    matched["state"] = "PRESENCE_CANDIDATE"
                continue

            if inside:
                if matched.get("state") == "REMOVAL_CANDIDATE":
                    print(
                        "[OBJECT-THEFT] Object returned to ROI "
                        f"camera={camera_id} track={matched['track_id']} "
                        f"session={matched['session_id']} action=CANCEL_REMOVAL"
                    )
                matched["state"] = "PRESENT"
                matched["removal_confirmations"] = 0
                matched["missed_frames"] = 0
            else:
                matched["state"] = "REMOVAL_CANDIDATE"
                matched["removal_confirmations"] += 1
                print(
                    "[OBJECT-THEFT] Removal candidate "
                    f"camera={camera_id} track={matched['track_id']} "
                    f"session={matched['session_id']} "
                    f"missed={matched['removal_confirmations']}/{self.max_missed}"
                )
                if matched["removal_confirmations"] >= self.max_missed:
                    self._confirm_removal(matched, camera_id, confirmed)

        for track in self.tracks:
            if track["track_id"] in seen or track.get("event_finalized"):
                continue
            track["missed_frames"] += 1
            if not track.get("presence_confirmed"):
                track["state"] = "PRESENCE_CANDIDATE"
                continue
            track["state"] = "REMOVAL_CANDIDATE"
            track["removal_confirmations"] += 1
            print(
                "[OBJECT-THEFT] Removal candidate "
                f"camera={camera_id} track={track['track_id']} "
                f"session={track['session_id']} "
                f"missed={track['removal_confirmations']}/{self.max_missed}"
            )
            if track["removal_confirmations"] >= self.max_missed:
                self._confirm_removal(track, camera_id, confirmed)

        self.tracks = [
            track for track in self.tracks if track.get("active", True)
        ]
        return confirmed
