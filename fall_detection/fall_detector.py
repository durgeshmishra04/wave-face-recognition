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
        """Classify posture conservatively. A person must show a real fall transition
        before a sustained fallen posture can become a fall candidate.
        """
        if keypoints is None or confidences is None or len(keypoints) < 17 or len(confidences) < 17:
            return "invalid"

        keypoints = np.asarray(keypoints, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)

        width = max(float(pose_box[2] - pose_box[0]), 1.0)
        height = max(float(pose_box[3] - pose_box[1]), 1.0)
        aspect = width / height

        points = {}
        for index in (5, 6, 11, 12, 13, 14, 15, 16):
            if index < len(keypoints) and float(confidences[index]) >= self.pose_confidence:
                points[index] = keypoints[index]

        shoulder_points = [points[i] for i in (5, 6) if i in points]
        hip_points = [points[i] for i in (11, 12) if i in points]
        knee_points = [points[i] for i in (13, 14) if i in points]
        ankle_points = [points[i] for i in (15, 16) if i in points]

        if not shoulder_points or not hip_points:
            return "ambiguous"

        shoulder_center = np.mean(np.asarray(shoulder_points), axis=0)
        hip_center = np.mean(np.asarray(hip_points), axis=0)
        knee_center = np.mean(np.asarray(knee_points), axis=0) if knee_points else None
        ankle_center = np.mean(np.asarray(ankle_points), axis=0) if ankle_points else None

        torso = hip_center - shoulder_center
        torso_angle = math.degrees(
            math.atan2(abs(float(torso[0])), max(abs(float(torso[1])), 1e-6))
        )
        torso_vertical = abs(float(torso[1]))
        shoulder_hip_gap = abs(float(hip_center[1] - shoulder_center[1]))

        # Normal upright posture is mostly vertical and compact.
        if torso_angle <= FALL_UPRIGHT_ANGLE_DEGREES and aspect <= FALL_UPRIGHT_ASPECT:
            return "upright"

        # Stable, low, bent postures are common during sitting, crouching, and shoe tying.
        # These are not fall evidence by themselves.
        if knee_center is not None and ankle_center is not None:
            knee_drop = float(knee_center[1] - hip_center[1])
            ankle_drop = float(ankle_center[1] - hip_center[1])
            if knee_drop > 12 and ankle_drop > 18 and torso_angle >= 25 and torso_angle < 75:
                return "crouching"

        if torso_angle >= 55 and torso_angle < 90:
            if knee_center is not None and ankle_center is not None:
                if float(hip_center[1]) <= float(knee_center[1]) and float(knee_center[1]) <= float(ankle_center[1]):
                    return "sitting"
            if abs(float(hip_center[1] - shoulder_center[1])) < max(height * 0.18, 20.0):
                return "bending"

        # A truly fallen person must show a strong body-lie signal with low torso and
        # a wide/short bounding box. Use these as supporting evidence, not the sole trigger.
        body_low = abs(float(hip_center[1] - shoulder_center[1])) < max(height * 0.22, 24.0)
        lower_body_low = ankle_center is not None and float(ankle_center[1]) > float(hip_center[1]) + 10.0
        strong_lie = (
            aspect >= FALL_HORIZONTAL_ASPECT
            and torso_angle >= max(FALL_TORSO_ANGLE_DEGREES, 60.0)
            and body_low
            and (lower_body_low or shoulder_hip_gap < max(height * 0.18, 20.0))
            and torso_vertical <= max(height * 0.45, 18.0)
        )
        if strong_lie:
            return "fallen"

        if aspect >= 1.0 and torso_angle >= 35 and torso_angle < 75:
            return "bending"
        if aspect > 0.9 and torso_angle >= 25 and shoulder_hip_gap > max(height * 0.22, 24.0):
            return "sitting"

        return "ambiguous"

    @staticmethod
    def _pose_metrics(pose_box, keypoints, confidences, pose_confidence):
        if keypoints is None or confidences is None or len(keypoints) < 17 or len(confidences) < 17:
            return {
                "center_y": None,
                "bottom_y": None,
                "torso_angle": None,
                "aspect_ratio": None,
            }
        keypoints = np.asarray(keypoints, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)
        points = {}
        for index in (5, 6, 11, 12, 13, 14, 15, 16):
            if index < len(keypoints) and float(confidences[index]) >= pose_confidence:
                points[index] = keypoints[index]
        if not points:
            return {
                "center_y": None,
                "bottom_y": None,
                "torso_angle": None,
                "aspect_ratio": None,
            }
        shoulder_points = [points[i] for i in (5, 6) if i in points]
        hip_points = [points[i] for i in (11, 12) if i in points]
        if not shoulder_points or not hip_points:
            return {
                "center_y": None,
                "bottom_y": None,
                "torso_angle": None,
                "aspect_ratio": None,
            }
        shoulder_center = np.mean(np.asarray(shoulder_points), axis=0)
        hip_center = np.mean(np.asarray(hip_points), axis=0)
        width = max(float(pose_box[2] - pose_box[0]), 1.0)
        height = max(float(pose_box[3] - pose_box[1]), 1.0)
        torso = hip_center - shoulder_center
        torso_angle = math.degrees(
            math.atan2(abs(float(torso[0])), max(abs(float(torso[1])), 1e-6))
        )
        center_y = float((pose_box[1] + pose_box[3]) / 2.0)
        bottom_y = float(pose_box[3])
        return {
            "center_y": center_y,
            "bottom_y": bottom_y,
            "torso_angle": torso_angle,
            "aspect_ratio": width / height,
        }

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
            metrics = self._pose_metrics(pose_box, points_for_pose, conf_for_pose, self.pose_confidence)
            print(
                f"[FALL POSE] camera={camera_id} track_id={track['track_id']} "
                f"posture={posture} aspect={metrics['aspect_ratio']} torso_angle={metrics['torso_angle']} "
                f"center_y={metrics['center_y']} bottom_y={metrics['bottom_y']}"
            )
            if self.tracker.update(
                camera_id,
                track["track_id"],
                posture,
                now=now,
                center_y=metrics["center_y"],
                bottom_y=metrics["bottom_y"],
                torso_angle=metrics["torso_angle"],
                aspect_ratio=metrics["aspect_ratio"],
                bbox=tuple(float(v) for v in track["box"]),
            ):
                events.append(FallEvent(str(camera_id), int(track["track_id"]), tuple(int(v) for v in track["box"]), float(score), posture))
        self.tracker.cleanup(now)
        return events
