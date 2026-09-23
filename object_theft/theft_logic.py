"""State logic for object theft detection. Keeps the theft policy separate from the loader/tracker."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ObjectTheftState:
    state: str = "ABSENT"
    baseline_bbox: tuple | None = None
    baseline_center: tuple | None = None
    baseline_area: float = 0.0
    alert_generated: bool = False
    event_finalized: bool = False
    missed_frames: int = 0
    removal_confirm_frames: int = 0
    session_id: str | None = None
    last_seen: float = 0.0
    object_class: str = "DRUM_CONTAINER"
    track_id: int | None = None


class ObjectTheftStateMachine:
    """Simple temporal state machine for object theft decisions."""

    def __init__(self, confirm_frames=5, max_missed_frames=15):
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_missed_frames = max(1, int(max_missed_frames))

    def update(self, track, detection_visible, now):
        state = track.get("state", "ABSENT")
        if detection_visible:
            if state in {"ABSENT", "FINALIZED"}:
                state = "PRESENT"
                track["session_id"] = track.get("session_id") or f"session-{int(now * 1000)}"
            if track.get("baseline_bbox") is None:
                track["baseline_bbox"] = tuple(track.get("bbox") or ())
                track["baseline_center"] = track.get("center")
                track["baseline_area"] = max((track["bbox"][2] - track["bbox"][0]) * (track["bbox"][3] - track["bbox"][1]), 1.0)
                state = "PRESENT"

            track["missed_frames"] = 0
            track["state"] = "PRESENT"
            track["last_seen"] = now
            return state

        track["missed_frames"] = int(track.get("missed_frames", 0)) + 1
        if track.get("missed_frames", 0) >= self.max_missed_frames and not track.get("alert_generated", False):
            track["state"] = "REMOVAL_CONFIRMED"
            track["event_finalized"] = True
            track["alert_generated"] = True
            return "REMOVAL_CONFIRMED"
        if track.get("missed_frames", 0) >= max(1, self.max_missed_frames // 2):
            track["state"] = "REMOVAL_CANDIDATE"
            return "REMOVAL_CANDIDATE"
        track["state"] = "ABSENT"
        return "ABSENT"
