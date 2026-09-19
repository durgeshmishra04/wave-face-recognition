"""Bounded, camera-scoped temporal state for existing person tracks."""

from collections import deque
from dataclasses import dataclass, field
import time

from .fall_config import (
    FALL_CONFIRMATION_FRAMES, FALL_HISTORY_SIZE, FALL_MIN_NORMAL_FRAMES,
    FALL_RECOVERY_FRAMES, FALL_TRACK_TIMEOUT_SECONDS,
)


@dataclass
class TrackFallState:
    state: str = "NORMAL"
    normal_frames: int = 0
    candidate_frames: int = 0
    recovery_frames: int = 0
    last_seen: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=FALL_HISTORY_SIZE))


class FallTracker:
    """Produces one confirmation per fall, then permits a later fall after recovery."""

    def __init__(self, confirmation_frames=FALL_CONFIRMATION_FRAMES,
                 recovery_frames=FALL_RECOVERY_FRAMES,
                 min_normal_frames=FALL_MIN_NORMAL_FRAMES,
                 timeout_seconds=FALL_TRACK_TIMEOUT_SECONDS):
        self.confirmation_frames = confirmation_frames
        self.recovery_frames_required = recovery_frames
        self.min_normal_frames = min_normal_frames
        self.timeout_seconds = timeout_seconds
        self.states = {}

    def update(self, camera_id, track_id, posture, now=None):
        """Update one existing track. Returns True only for a newly confirmed fall."""
        now = time.time() if now is None else now
        key = (str(camera_id), int(track_id))
        state = self.states.setdefault(key, TrackFallState())
        state.last_seen = now
        state.history.append(posture)
        upright_transition_ready = state.normal_frames >= self.min_normal_frames

        if posture == "upright":
            state.normal_frames += 1
        elif posture == "fallen":
            state.normal_frames = 0

        if state.state == "NORMAL":
            if posture == "fallen" and state.normal_frames == 0:
                # A fall may only start after an observed upright posture.
                if upright_transition_ready:
                    state.state, state.candidate_frames = "CANDIDATE", 1
            return False

        if state.state == "CANDIDATE":
            if posture == "fallen":
                state.candidate_frames += 1
                if state.candidate_frames >= self.confirmation_frames:
                    state.state = "ALERTED"
                    state.recovery_frames = 0
                    return True
            elif posture == "upright":
                state.state = "NORMAL"
                state.candidate_frames = 0
            # Invalid/ambiguous poses do not force a state transition.
            return False

        if state.state == "ALERTED":
            if posture == "upright":
                state.state, state.recovery_frames = "RECOVERING", 1
            return False

        if state.state == "RECOVERING":
            if posture == "upright":
                state.recovery_frames += 1
                if state.recovery_frames >= self.recovery_frames_required:
                    state.state = "NORMAL"
                    state.candidate_frames = 0
            elif posture == "fallen":
                state.state, state.recovery_frames = "ALERTED", 0
            return False
        return False

    def cleanup(self, now=None):
        now = time.time() if now is None else now
        self.states = {
            key: value for key, value in self.states.items()
            if now - value.last_seen <= self.timeout_seconds
        }
