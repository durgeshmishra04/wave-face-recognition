"""Grounding DINO adapter for object theft detection on CAM008."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .config import (
    OBJECT_THEFT_CONFIDENCE_THRESHOLD,
    OBJECT_THEFT_ENABLED,
    OBJECT_THEFT_IOU_THRESHOLD,
    OBJECT_THEFT_LOCAL_FILES_ONLY,
    OBJECT_THEFT_MODEL_ID,
    OBJECT_THEFT_MODEL_PATH,
    OBJECT_THEFT_TARGETS,
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
    """Load and run Grounding DINO once, then reuse it for CAM008 frames."""

    def __init__(
        self,
        model_path=None,
        model_id=None,
        model_loader=None,
        confidence=OBJECT_THEFT_CONFIDENCE_THRESHOLD,
        iou=OBJECT_THEFT_IOU_THRESHOLD,
        targets=None,
    ):
        self.enabled = OBJECT_THEFT_ENABLED
        self.model_path = model_path or OBJECT_THEFT_MODEL_PATH
        self.model_id = model_id or OBJECT_THEFT_MODEL_ID
        self.model_loader = model_loader
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.targets = list(targets or OBJECT_THEFT_TARGETS)
        self.local_files_only = OBJECT_THEFT_LOCAL_FILES_ONLY
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.tracker = ObjectTheftTracker(iou_threshold=self.iou)

        if self.enabled:
            self.model, self.processor = self._load_model()

    def _load_model(self):
        """Load a compatible Grounding DINO backend from a local path or Hugging Face repository."""
        try:
            if self.model_loader is not None:
                model = self.model_loader(self.model_path or self.model_id)
                print(f"[OBJECT-THEFT] Model source: {self.model_path or self.model_id}")
                print(f"[OBJECT-THEFT] Device: {self.device}")
                return model, None

            from transformers import AutoProcessor, GroundingDinoForObjectDetection

            model_source = self.model_path or self.model_id
            print("[OBJECT-THEFT] Initializing...")
            print(f"[OBJECT-THEFT] Model source: {model_source}")
            if self.model_path and os.path.exists(self.model_path):
                processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self.local_files_only)
                model = GroundingDinoForObjectDetection.from_pretrained(self.model_path, local_files_only=self.local_files_only)
            else:
                processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=self.local_files_only)
                model = GroundingDinoForObjectDetection.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            model.to(self.device)
            model.eval()
            print("[OBJECT-THEFT] Model loaded successfully")
            print(f"[OBJECT-THEFT] Device: {self.device}")
            print("[OBJECT-THEFT] Cameras: CAM008")
            print(f"[OBJECT-THEFT] Targets: {self.targets}")
            return model, processor
        except Exception as error:
            print(f"[OBJECT-THEFT] Model unavailable - safely disabled: {error}")
            return None, None

    @staticmethod
    def _canon_label(value):
        text = str(value or "").strip().lower()
        if any(token in text for token in ("drum", "barrel", "container")):
            return CANONICAL_CLASS
        return text.upper() if text else "UNKNOWN"

    @staticmethod
    def _normalise(text):
        text = str(text or "").strip().lower()
        text = text.replace("-", " ").replace("_", " ")
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _merge_detections(detections):
        merged = []
        for detection in detections:
            matched = False
            for item in merged:
                box_a = np.asarray(item["bbox"], dtype=np.float32)
                box_b = np.asarray(detection["bbox"], dtype=np.float32)
                if item["canonical_class"] != detection["canonical_class"]:
                    continue
                left = max(float(box_a[0]), float(box_b[0]))
                top = max(float(box_a[1]), float(box_b[1]))
                right = min(float(box_a[2]), float(box_b[2]))
                bottom = min(float(box_a[3]), float(box_b[3]))
                inter = max(0.0, right - left) * max(0.0, bottom - top)
                area_a = max(float(box_a[2] - box_a[0]), 0.0) * max(float(box_a[3] - box_a[1]), 0.0)
                area_b = max(float(box_b[2] - box_b[0]), 0.0) * max(float(box_b[3] - box_b[1]), 0.0)
                iou = inter / max(area_a + area_b - inter, 1e-6)
                if iou >= 0.35:
                    item["bbox"] = tuple(int(v) for v in np.asarray([box_a, box_b]).min(axis=0).tolist()) + tuple(int(v) for v in np.asarray([box_a, box_b]).max(axis=0).tolist())
                    item["confidence"] = max(item["confidence"], detection["confidence"])
                    matched = True
                    break
            if not matched:
                merged.append(detection)
        return merged

    def _parse_grounding_dino_results(self, results):
        detections = []
        if not results:
            return detections
        for result in results:
            annotations = result.get("annotations", []) if isinstance(result, dict) else []
            for annotation in annotations:
                boxes = annotation.get("boxes") or []
                scores = annotation.get("scores") or []
                labels = annotation.get("labels") or []
                for box, score, label in zip(boxes, scores, labels):
                    label_text = self._normalise(label)
                    if score < self.confidence:
                        continue
                    if any(target in label_text for target in [self._normalise(t) for t in self.targets]):
                        canonical = self._canon_label(label_text)
                        detections.append({
                            "class_name": label_text,
                            "canonical_class": canonical,
                            "confidence": float(score),
                            "bbox": tuple(int(v) for v in box),
                        })
        return self._merge_detections(detections)

    def _detect(self, frame, prompts=None, threshold=None):
        if self.model is None or self.processor is None:
            return []
        prompts = prompts or self.targets
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        text = list(dict.fromkeys([str(p).strip() for p in prompts if str(p).strip()]))
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold or self.confidence,
            text_threshold=threshold or self.confidence,
            target_sizes=target_sizes,
        )
        return results

    def process(self, frame, camera_id, roi_polygon=None, inference=None, now=None):
        now = time.time() if now is None else now
        if not self.enabled or self.model is None or self.processor is None:
            return []
        if inference is None:
            return []
        try:
            results = inference(self._detect, frame, prompts=self.targets, threshold=self.confidence)
        except Exception as error:
            print(f"[{camera_id}] [OBJECT-THEFT] inference failed; continuing existing pipeline: {error}")
            return []
        detections = self._parse_grounding_dino_results(results)
        if not detections:
            return []
        confirmed = self.tracker.update(camera_id, detections, roi_polygon=roi_polygon, now=now)
        alerts = []
        for item in confirmed:
            alerts.append(ObjectTheftDetection(
                camera_id=str(camera_id),
                track_id=int(item["track_id"]),
                class_name=str(item.get("class_name", item.get("canonical_class", "DRUM_CONTAINER"))),
                canonical_class=str(item.get("canonical_class", "DRUM_CONTAINER")),
                confidence=float(item.get("confidence", 0.0)),
                bbox=tuple(item["bbox"]),
                session_id=str(item.get("session_id") or "session-unknown"),
            ))
        return alerts
