"""YOLO PPE adapter that annotates only PPE associated with existing tracks."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .ppe_config import (
    BOOTS_CONFIDENCE, BOOTS_ENABLED, GLOVES_CONFIDENCE, GLOVES_ENABLED,
    HELMET_CONFIDENCE, HELMET_ENABLED, PPE_CONFIDENCE, PPE_IOU,
    PPE_PERSON_MARGIN, PPE_POSITIVE_CLASSES, PPE_TRACK_TIMEOUT_SECONDS,
)
from .ppe_tracker import HelmetAlertTracker, PPETracker


@dataclass(frozen=True)
class PPEDetection:
    class_name: str
    label: str
    confidence: float
    box: tuple
    track_id: int


class PPEDetector:
    """Runs PPE inference without becoming a source of people or events."""

    def __init__(self, model_path, model_loader, confidence=PPE_CONFIDENCE, iou=PPE_IOU):
        self.model = model_loader(model_path)  # Exactly one model instance.
        self.confidence = confidence
        self.iou = iou
        self.tracker = PPETracker(PPE_TRACK_TIMEOUT_SECONDS)
        self.class_names = self._class_names(getattr(self.model, "names", {}))
        self.supported_classes = {
            name: PPE_POSITIVE_CLASSES[self._normalise(name)]
            for name in self.class_names.values()
            if self._normalise(name) in PPE_POSITIVE_CLASSES
        }
        self.helmet_tracker = HelmetAlertTracker()

    @staticmethod
    def _class_names(names):
        return names if isinstance(names, dict) else dict(enumerate(names or []))

    @staticmethod
    def _normalise(name):
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _inside_roi(box, polygon):
        point = ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
        return polygon is not None and len(polygon) >= 3 and cv2.pointPolygonTest(
            np.asarray(polygon, dtype=np.float32), point, False
        ) >= 0

    @staticmethod
    def _associate(box, tracks):
        center_x, center_y = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        best, best_score = None, float("inf")
        for track in tracks:
            if not track.get("matched_this_frame"):
                continue
            x1, y1, x2, y2 = (float(value) for value in track["box"])
            margin_x, margin_y = (x2 - x1) * PPE_PERSON_MARGIN, (y2 - y1) * PPE_PERSON_MARGIN
            if not (x1 - margin_x <= center_x <= x2 + margin_x and y1 - margin_y <= center_y <= y2 + margin_y):
                continue
            # Prefer the closest valid body center to avoid mixing PPE among workers.
            score = ((center_x - (x1 + x2) / 2.0) / max(x2 - x1, 1.0)) ** 2 + ((center_y - (y1 + y2) / 2.0) / max(y2 - y1, 1.0)) ** 2
            if score < best_score:
                best, best_score = track, score
        return best

    @staticmethod
    def _is_enabled(label, confidence):
        return ((label == "Helmet" and HELMET_ENABLED and confidence >= HELMET_CONFIDENCE)
                or (label == "Gloves" and GLOVES_ENABLED and confidence >= GLOVES_CONFIDENCE)
                or (label == "Safety Boots" and BOOTS_ENABLED and confidence >= BOOTS_CONFIDENCE))

    def process(self, frame, camera_id, person_tracks, roi_polygon, inference, now=None):
        """Return detections and newly confirmed helmet sessions for existing tracks."""
        now = time.time() if now is None else now
        eligible = [track for track in person_tracks if track.get("matched_this_frame") and self._inside_roi(track["box"], roi_polygon)]
        if not eligible:
            self.tracker.cleanup(now)
            self.helmet_tracker.cleanup(now)
            return [], []
        try:
            results = inference(self.model.predict, frame, conf=self.confidence, iou=self.iou, verbose=False)
        except Exception as error:
            print(f"[{camera_id}] [PPE] inference failed; continuing existing pipeline: {error}")
            self.tracker.cleanup(now)
            return [], []
        if not results or getattr(results[0], "boxes", None) is None:
            self.tracker.cleanup(now)
            return [], []
        boxes = results[0].boxes
        raw_boxes, scores, classes = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        detections = []
        for box, score, class_id in zip(raw_boxes, scores, classes):
            class_name = self.class_names.get(int(class_id))
            label = PPE_POSITIVE_CLASSES.get(self._normalise(class_name)) if class_name is not None else None
            track = self._associate(box, eligible)
            if label is None or track is None or not self._is_enabled(label, float(score)):
                continue
            detection = PPEDetection(self._normalise(class_name), label, float(score), tuple(int(value) for value in box), int(track["track_id"]))
            detections.append(detection)
            self.tracker.update(camera_id, detection.track_id, label, detection.confidence, now)
        self.tracker.cleanup(now)
        eligible_ids = [track["track_id"] for track in eligible]
        helmets = [item for item in detections if item.label == "Helmet"]
        confirmed = self.helmet_tracker.update(camera_id, eligible_ids, helmets, now)
        self.helmet_tracker.cleanup(now)
        return detections, confirmed

    @staticmethod
    def annotate(frame, detections):
        """Draw only on the existing outgoing frame; no IO or notification side effects."""
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color = (0, 215, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{detection.label} {detection.confidence:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return frame
