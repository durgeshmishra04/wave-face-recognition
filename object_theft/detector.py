"""Ultralytics YOLO adapter for the isolated object theft KPI."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .config import (
    OBJECT_THEFT_CLASS_NAMES,
    OBJECT_THEFT_CONFIDENCE_THRESHOLD,
    OBJECT_THEFT_ENABLED,
    OBJECT_THEFT_INFERENCE_INTERVAL,
    OBJECT_THEFT_IOU_THRESHOLD,
    OBJECT_THEFT_MODEL_PATH,
)
from .tracker import ObjectTheftTracker


CANONICAL_CLASS = "DRUM_CONTAINER"


@dataclass(frozen=True)
class ObjectTheftDetection:
    camera_id: str
    track_id: int
    class_name: str
    canonical_class: str
    confidence: float
    bbox: tuple
    session_id: str


class ObjectTheftDetector:
    """Load one custom YOLO model and reuse it for CAM008 frames."""

    def __init__(
        self,
        model_path=None,
        model_loader=None,
        confidence=OBJECT_THEFT_CONFIDENCE_THRESHOLD,
        iou=OBJECT_THEFT_IOU_THRESHOLD,
    ):
        self.enabled = OBJECT_THEFT_ENABLED
        self.model_path = model_path or OBJECT_THEFT_MODEL_PATH
        self.model_loader = model_loader
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_names = {}
        self.allowed_class_ids = set()
        self.inference_interval = OBJECT_THEFT_INFERENCE_INTERVAL
        self.frame_count = 0
        self.tracker = ObjectTheftTracker(iou_threshold=self.iou)

        if self.enabled:
            self.model = self._load_model()

    def _load_model(self):
        try:
            if not self.model_path:
                raise FileNotFoundError(
                    "OBJECT_THEFT_MODEL_PATH is not configured"
                )
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(self.model_path)

            if self.model_loader is not None:
                model = self.model_loader(self.model_path)
            else:
                from ultralytics import YOLO

                model = YOLO(self.model_path)

            raw_names = getattr(model, "names", None)
            if raw_names is None:
                raise ValueError("custom YOLO model has no model.names")
            self.model_names = {
                int(class_id): str(name) for class_id, name in dict(raw_names).items()
            }
            configured_names = OBJECT_THEFT_CLASS_NAMES
            if configured_names:
                self.allowed_class_ids = {
                    class_id
                    for class_id, name in self.model_names.items()
                    if name.strip().lower() in configured_names
                }
            else:
                self.allowed_class_ids = {
                    class_id
                    for class_id, name in self.model_names.items()
                    if any(
                        token in name.strip().lower()
                        for token in ("drum", "barrel", "container")
                    )
                }

            print(f"[OBJECT-THEFT] Model source: {self.model_path}")
            print(f"[OBJECT-THEFT] Model classes: {self.model_names}")
            print(f"[OBJECT-THEFT] Device: {self.device}")
            if not self.allowed_class_ids:
                raise ValueError(
                    "model.names contains no configured drum/barrel/container class"
                )
            return model
        except Exception as error:
            print(f"[OBJECT-THEFT] Model unavailable - safely disabled: {error}")
            return None

    @staticmethod
    def _value(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _merge_detections(detections):
        merged = []
        for detection in detections:
            for item in merged:
                if item["canonical_class"] != detection["canonical_class"]:
                    continue
                a = np.asarray(item["bbox"], dtype=np.float32)
                b = np.asarray(detection["bbox"], dtype=np.float32)
                left, top = np.maximum(a[:2], b[:2])
                right, bottom = np.minimum(a[2:], b[2:])
                intersection = max(0.0, right - left) * max(0.0, bottom - top)
                area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                overlap = intersection / max(area_a + area_b - intersection, 1e-6)
                if overlap >= 0.35:
                    if detection["confidence"] > item["confidence"]:
                        item.update(detection)
                    break
            else:
                merged.append(detection)
        return merged

    def _parse_results(self, results, frame_shape):
        height, width = frame_shape[:2]
        detections = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", [])
            scores = getattr(boxes, "conf", [])
            class_ids = getattr(boxes, "cls", [])
            for box, score, class_id in zip(xyxy, scores, class_ids):
                class_id = int(self._value(class_id))
                score = float(self._value(score))
                if class_id not in self.allowed_class_ids or score < self.confidence:
                    continue
                coords = [float(self._value(value)) for value in box]
                x1 = max(0, min(width, int(coords[0])))
                y1 = max(0, min(height, int(coords[1])))
                x2 = max(0, min(width, int(coords[2])))
                y2 = max(0, min(height, int(coords[3])))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    {
                        "class_name": self.model_names[class_id],
                        "canonical_class": CANONICAL_CLASS,
                        "confidence": score,
                        "bbox": (x1, y1, x2, y2),
                    }
                )
        return self._merge_detections(detections)

    def _detect(self, frame):
        return self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )

    def process(self, frame, camera_id, roi_polygon=None, inference=None, now=None):
        if (
            not self.enabled
            or self.model is None
            or str(camera_id or "").upper() != "CAM008"
            or inference is None
        ):
            return []
        self.frame_count += 1
        if self.frame_count % self.inference_interval != 0:
            return []
        try:
            results = inference(self._detect, frame)
            detections = self._parse_results(results, frame.shape)
        except Exception as error:
            print(
                f"[{camera_id}] [OBJECT-THEFT] inference failed; "
                f"continuing existing pipeline: {error}"
            )
            return []
        confirmed = self.tracker.update(
            camera_id,
            detections,
            roi_polygon=roi_polygon,
            now=time.time() if now is None else now,
        )
        return [
            ObjectTheftDetection(
                camera_id=str(camera_id),
                track_id=int(item["track_id"]),
                class_name=str(item.get("class_name", CANONICAL_CLASS)),
                canonical_class=str(item.get("canonical_class", CANONICAL_CLASS)),
                confidence=float(item.get("confidence", 0.0)),
                bbox=tuple(item["bbox"]),
                session_id=str(item.get("session_id") or "session-unknown"),
            )
            for item in confirmed
        ]
