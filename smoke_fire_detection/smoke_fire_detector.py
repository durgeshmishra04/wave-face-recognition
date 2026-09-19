"""YOLO smoke/fire inference, ROI filtering, temporal incident hand-off, annotation."""

from dataclasses import dataclass
import time

import cv2
import numpy as np

from .smoke_fire_config import FIRE_CONFIDENCE_THRESHOLD, SMOKE_CONFIDENCE_THRESHOLD, SMOKE_FIRE_IOU
from .smoke_fire_tracker import SmokeFireTracker


@dataclass(frozen=True)
class SmokeFireDetection:
    type: str
    confidence: float
    box: tuple
    camera_id: str


@dataclass(frozen=True)
class SmokeFireIncident:
    incident_id: int
    type: str
    confidence: float
    box: tuple
    camera_id: str
    smoke_present: bool
    fire_present: bool


class SmokeFireDetector:
    """Detects only genuine model fire/smoke classes; it has no alert transport."""

    def __init__(self, model_path, model_loader, smoke_confidence=SMOKE_CONFIDENCE_THRESHOLD,
                 fire_confidence=FIRE_CONFIDENCE_THRESHOLD, iou=SMOKE_FIRE_IOU):
        self.model = model_loader(model_path)
        self.smoke_confidence = smoke_confidence
        self.fire_confidence = fire_confidence
        self.iou = iou
        names = getattr(self.model, "names", {})
        self.class_names = names if isinstance(names, dict) else dict(enumerate(names or []))
        self.class_types = {
            class_id: str(name).strip().lower()
            for class_id, name in self.class_names.items()
            if str(name).strip().lower() in {"fire", "smoke"}
        }
        missing = {"fire", "smoke"} - set(self.class_types.values())
        if missing:
            raise ValueError(f"Model is missing required Smoke/Fire classes: {sorted(missing)}")
        self.tracker = SmokeFireTracker()

    @staticmethod
    def _inside_roi(box, polygon):
        center = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
        return polygon is not None and len(polygon) >= 3 and cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), center, False) >= 0

    def process(self, frame, camera_id, roi_polygon, inference, now=None):
        """Return visible detections and newly confirmed incidents, without IO side effects."""
        now = time.time() if now is None else now
        try:
            results = inference(self.model.predict, frame, conf=min(self.smoke_confidence, self.fire_confidence), iou=self.iou, verbose=False)
        except Exception as error:
            print(f"[{camera_id}] [SMOKE/FIRE] inference failed; existing pipeline continues: {error}")
            return [], self.tracker.update(camera_id, [], now)
        detections = []
        if results and getattr(results[0], "boxes", None) is not None:
            boxes = results[0].boxes
            for box, confidence, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                detection_type = self.class_types.get(int(class_id))
                threshold = self.fire_confidence if detection_type == "fire" else self.smoke_confidence
                if detection_type is None or confidence < threshold or not self._inside_roi(box, roi_polygon):
                    continue
                detections.append(SmokeFireDetection(detection_type, float(confidence), tuple(int(value) for value in box), str(camera_id)))
        confirmed = self.tracker.update(camera_id, detections, now)
        incidents = [
            SmokeFireIncident(item.incident_id, "fire" if "fire" in item.types else "smoke", item.confidence, item.box, str(camera_id), "smoke" in item.types, "fire" in item.types)
            for item in confirmed
        ]
        return detections, incidents

    @staticmethod
    def annotate(frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color = (0, 0, 255) if detection.type == "fire" else (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{detection.type.upper()} {detection.confidence:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)
        return frame
