"""YOLO pose adapter; it never creates people or person tracks."""

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .fall_config import (
    FALL_CONFIDENCE, FALL_HORIZONTAL_ASPECT, FALL_MIN_IOU,
    FALL_POSE_CONFIDENCE, FALL_TORSO_ANGLE_DEGREES, FALL_UPRIGHT_ANGLE_DEGREES,
    FALL_UPRIGHT_ASPECT,
)
from .fall_tracker import FallTracker


@dataclass(frozen=True)
class FallEvent:
    camera_id: str
    track_id: int
    box: tuple
    confidence: float
    posture: str


class FallDetector:
    """Associate pose results conservatively to authoritative existing tracks."""

    def __init__(self, model_path, model_loader, confidence=FALL_CONFIDENCE,
                 pose_confidence=FALL_POSE_CONFIDENCE, min_iou=FALL_MIN_IOU):
        self.model = model_loader(model_path)  # Loaded exactly once at startup.
        self.confidence = confidence
        self.pose_confidence = pose_confidence
        self.min_iou = min_iou
        self.tracker = FallTracker()

    @staticmethod
    def _iou(a, b):
        left, top = max(a[0], b[0]), max(a[1], b[1])
        right, bottom = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _inside_roi(box, polygon):
        if polygon is None or len(polygon) < 3:
            return False
        point = ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
        return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), point, False) >= 0

    def _associate(self, pose_box, tracks):
        best, score = None, self.min_iou
        for track in tracks:
            if not track.get("matched_this_frame"):
                continue
            box = np.asarray(track["box"], dtype=np.float32)
            overlap = self._iou(pose_box, box)
            if overlap >= score:
                best, score = track, overlap
        return best

    def _posture(self, pose_box, keypoints, confidences):
        # COCO indices: shoulders 5/6, hips 11/12. Both pairs are required.
        required = (5, 6, 11, 12)
        if keypoints is None or confidences is None or len(keypoints) < 13 or len(confidences) < 13:
            return "invalid"
        if any(float(confidences[index]) < self.pose_confidence for index in required):
            return "invalid"
        shoulders = np.mean(np.asarray(keypoints)[[5, 6]], axis=0)
        hips = np.mean(np.asarray(keypoints)[[11, 12]], axis=0)
        torso = hips - shoulders
        torso_angle = math.degrees(math.atan2(abs(float(torso[0])), max(abs(float(torso[1])), 1e-6)))
        width, height = max(float(pose_box[2] - pose_box[0]), 1.0), max(float(pose_box[3] - pose_box[1]), 1.0)
        aspect = width / height
        if aspect >= FALL_HORIZONTAL_ASPECT or torso_angle >= FALL_TORSO_ANGLE_DEGREES:
            return "fallen"
        if aspect <= FALL_UPRIGHT_ASPECT and torso_angle <= FALL_UPRIGHT_ANGLE_DEGREES:
            return "upright"
        return "ambiguous"

    def process(self, frame, camera_id, person_tracks, roi_polygon, inference, now=None):
        """Return newly confirmed falls for authoritative tracks inside the given ROI."""
        now = time.time() if now is None else now
        eligible_tracks = [
            track for track in person_tracks
            if track.get("matched_this_frame")
            and self._inside_roi(track["box"], roi_polygon)
        ]
        # Avoid pose inference altogether unless the primary detector has an
        # active person track inside the configured master ROI.
        if not eligible_tracks:
            self.tracker.cleanup(now)
            return []
        results = inference(self.model.predict, frame, conf=self.confidence, verbose=False)
        events = []
        if not results:
            self.tracker.cleanup(now)
            return events
        result = results[0]
        boxes = getattr(getattr(result, "boxes", None), "xyxy", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            self.tracker.cleanup(now)
            return events
        pose_boxes = boxes.cpu().numpy()
        points = keypoints.xy.cpu().numpy()
        confs = keypoints.conf.cpu().numpy() if getattr(keypoints, "conf", None) is not None else None
        scores = result.boxes.conf.cpu().numpy() if getattr(result.boxes, "conf", None) is not None else np.ones(len(pose_boxes))
        for pose_box, points_for_pose, conf_for_pose, score in zip(pose_boxes, points, confs if confs is not None else [None] * len(pose_boxes), scores):
            track = self._associate(pose_box, eligible_tracks)
            if track is None:
                continue
            posture = self._posture(pose_box, points_for_pose, conf_for_pose)
            if self.tracker.update(camera_id, track["track_id"], posture, now):
                events.append(FallEvent(str(camera_id), int(track["track_id"]), tuple(int(v) for v in track["box"]), float(score), posture))
        self.tracker.cleanup(now)
        return events
