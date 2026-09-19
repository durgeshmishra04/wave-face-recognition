"""Bounded camera-local incident state for smoke/fire, never person tracking."""

from dataclasses import dataclass, field
import math
import time

from .smoke_fire_config import (
    SMOKE_FIRE_ASSOCIATION_DISTANCE_RATIO, SMOKE_FIRE_ASSOCIATION_IOU,
    SMOKE_FIRE_CLEAR_FRAMES, SMOKE_FIRE_CLEAR_SECONDS,
    SMOKE_FIRE_CONFIRM_FRAMES, SMOKE_FIRE_CONFIRM_SECONDS,
)


@dataclass
class IncidentState:
    incident_id: int
    box: tuple
    types: set = field(default_factory=set)
    candidate_frames: int = 0
    missed_frames: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    alerted: bool = False
    confidence: float = 0.0


class SmokeFireTracker:
    """Associates nearby smoke/fire boxes into one bounded incident session."""

    def __init__(self):
        self.incidents = {}
        self._next_id = 1

    @staticmethod
    def _iou(first, second):
        left, top = max(first[0], second[0]), max(first[1], second[1])
        right, bottom = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        union = max(0, first[2] - first[0]) * max(0, first[3] - first[1]) + max(0, second[2] - second[0]) * max(0, second[3] - second[1]) - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _near(first, second):
        first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
        distance = math.dist(first_center, second_center)
        scale = max(first[2] - first[0], first[3] - first[1], second[2] - second[0], second[3] - second[1], 1)
        return distance <= scale * SMOKE_FIRE_ASSOCIATION_DISTANCE_RATIO

    def _match(self, active, box):
        matches = [incident for incident in active if self._iou(incident.box, box) >= SMOKE_FIRE_ASSOCIATION_IOU or self._near(incident.box, box)]
        return max(matches, key=lambda incident: self._iou(incident.box, box), default=None)

    def update(self, camera_id, detections, now=None):
        """Return newly confirmed incidents; preserves smoke→fire as one session."""
        now = time.time() if now is None else now
        camera_id = str(camera_id)
        active = self.incidents.setdefault(camera_id, [])
        seen_ids = set()
        emitted = []
        for detection in detections:
            incident = self._match(active, detection.box)
            if incident is None:
                incident = IncidentState(self._next_id, detection.box, first_seen=now, last_seen=now)
                self._next_id += 1
                active.append(incident)
            if incident.incident_id not in seen_ids:
                incident.candidate_frames += 1
                seen_ids.add(incident.incident_id)
            incident.box = detection.box
            incident.types.add(detection.type)
            incident.confidence = max(incident.confidence, detection.confidence)
            incident.last_seen = now
            incident.missed_frames = 0
            confirmed_by_frames = incident.candidate_frames >= SMOKE_FIRE_CONFIRM_FRAMES
            confirmed_by_time = SMOKE_FIRE_CONFIRM_SECONDS > 0 and now - incident.first_seen >= SMOKE_FIRE_CONFIRM_SECONDS
            if not incident.alerted and (confirmed_by_frames or confirmed_by_time):
                incident.alerted = True
                emitted.append(incident)
        remaining = []
        for incident in active:
            if incident.incident_id not in seen_ids:
                incident.missed_frames += 1
            cleared_by_frames = incident.missed_frames >= SMOKE_FIRE_CLEAR_FRAMES
            cleared_by_time = SMOKE_FIRE_CLEAR_SECONDS > 0 and now - incident.last_seen >= SMOKE_FIRE_CLEAR_SECONDS
            if cleared_by_frames or cleared_by_time:
                print(f"[SMOKE/FIRE] EVENT_CLEARED | camera={camera_id} | incident={incident.incident_id}")
            else:
                remaining.append(incident)
        self.incidents[camera_id] = remaining
        return emitted
