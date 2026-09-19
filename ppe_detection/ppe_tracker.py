"""Small bounded association cache keyed by existing camera/person tracks."""

import time
from dataclasses import dataclass

from .ppe_config import PPE_HELMET_CLEAR_FRAMES, PPE_HELMET_CONFIRM_FRAMES, PPE_TRACK_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ConfirmedHelmet:
    camera_id: str
    track_id: int
    confidence: float
    box: tuple


class HelmetAlertTracker:
    """Bounded confirmation/clearance state for existing camera person tracks."""

    def __init__(self, confirm_frames=PPE_HELMET_CONFIRM_FRAMES,
                 clear_frames=PPE_HELMET_CLEAR_FRAMES):
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames
        self.states = {}

    def update(self, camera_id, eligible_track_ids, helmets, now=None):
        """Return each newly confirmed helmet session exactly once."""
        now = time.time() if now is None else now
        camera_id = str(camera_id)
        by_track = {item.track_id: item for item in helmets}
        confirmed = []
        for track_id in eligible_track_ids:
            key = (camera_id, int(track_id))
            state = self.states.setdefault(key, {
                "seen": 0, "missing": 0, "alerted": False, "last_seen": now,
            })
            item = by_track.get(int(track_id))
            if item is not None:
                state["seen"] += 1
                state["missing"] = 0
                state["last_seen"] = now
                if not state["alerted"] and state["seen"] >= self.confirm_frames:
                    state["alerted"] = True
                    confirmed.append(ConfirmedHelmet(camera_id, int(track_id), item.confidence, item.box))
            else:
                state["missing"] += 1
                if state["missing"] >= self.clear_frames:
                    state.update({"seen": 0, "missing": 0, "alerted": False})
        return confirmed

    def cleanup(self, now=None):
        """Bound state without allowing one camera worker to clear another's."""
        now = time.time() if now is None else now
        self.states = {
            key: value for key, value in self.states.items()
            if now - value["last_seen"] <= PPE_TRACK_TIMEOUT_SECONDS
        }


class PPETracker:
    """Caches only the latest PPE attributes; it is not a person tracker."""

    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.states = {}

    def update(self, camera_id, track_id, label, confidence, now=None):
        now = time.time() if now is None else now
        state = self.states.setdefault((str(camera_id), int(track_id)), {"items": {}, "last_seen": now})
        state["last_seen"] = now
        previous = state["items"].get(label, 0.0)
        state["items"][label] = max(previous, float(confidence))

    def cleanup(self, now=None):
        now = time.time() if now is None else now
        self.states = {
            key: state for key, state in self.states.items()
            if now - state["last_seen"] <= self.timeout_seconds
        }
