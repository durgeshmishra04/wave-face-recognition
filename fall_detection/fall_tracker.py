"""Bounded, camera-scoped temporal state for existing person tracks."""

from collections import deque
from dataclasses import dataclass, field
import time

from .fall_config import (
    FALL_CONFIRMATION_FRAMES, FALL_HISTORY_SIZE, FALL_MIN_NORMAL_FRAMES,
    FALL_RECOVERY_FRAMES, FALL_REQUIRE_UPRIGHT_TRANSITION,
    FALL_TRACK_TIMEOUT_SECONDS,
)


@dataclass
class TrackFallState:
    state: str = "NORMAL"
    normal_frames: int = 0
    candidate_frames: int = 0
    recovery_frames: int = 0
    last_seen: float = 0.0
    last_posture: str | None = None
    last_center_y: float | None = None
    last_bottom_y: float | None = None
    last_torso_angle: float | None = None
    last_aspect_ratio: float | None = None
    last_transition_ts: float | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=FALL_HISTORY_SIZE))


class FallTracker:
    """Produces one confirmation per fall, then permits a later fall after recovery."""

    def __init__(self, confirmation_frames=FALL_CONFIRMATION_FRAMES,
                 recovery_frames=FALL_RECOVERY_FRAMES,
                 min_normal_frames=FALL_MIN_NORMAL_FRAMES,
                 require_upright_transition=FALL_REQUIRE_UPRIGHT_TRANSITION,
                 timeout_seconds=FALL_TRACK_TIMEOUT_SECONDS):
        self.confirmation_frames = confirmation_frames
        self.recovery_frames_required = recovery_frames
        self.min_normal_frames = min_normal_frames
        self.require_upright_transition = require_upright_transition
        self.timeout_seconds = timeout_seconds
        self.states = {}

    @staticmethod
    def _is_non_fall_posture(posture):
        return posture in {"upright", "bending", "sitting", "crouching", "ambiguous", "invalid"}

    def _transition_signal(self, state, posture, center_y, bottom_y, torso_angle, aspect_ratio):
        if posture != "fallen":
            return False
        if center_y is None or bottom_y is None:
            return False
        if state.last_posture not in {"upright", "ambiguous"}:
            return False
        if state.last_bottom_y is None or state.last_center_y is None:
            return False

        delta_center = float(center_y) - float(state.last_center_y)
        delta_bottom = float(bottom_y) - float(state.last_bottom_y)
        delta_torso = 0.0 if state.last_torso_angle is None else abs(float(torso_angle) - float(state.last_torso_angle)) if torso_angle is not None else 0.0
        delta_aspect = 0.0 if state.last_aspect_ratio is None else float(aspect_ratio) - float(state.last_aspect_ratio) if aspect_ratio is not None else 0.0

        transition = (
            delta_bottom >= 8.0
            and delta_center >= 6.0
            and (delta_torso >= 12.0 or delta_aspect >= 0.10)
            and (state.last_posture == "upright" or state.last_posture == "ambiguous")
            and (torso_angle is not None and torso_angle >= 35.0)
        )
        return transition

    def update(self, camera_id, track_id, posture, now=None, center_y=None, bottom_y=None,
               torso_angle=None, aspect_ratio=None, bbox=None, transition=None):
        """Update one existing track. Returns True only for a newly confirmed fall."""
        now = time.time() if now is None else now
        key = (str(camera_id), int(track_id))
        state = self.states.setdefault(key, TrackFallState())
        state.last_seen = now
        state.history.append({
            "timestamp": now,
            "posture": posture,
            "center_y": center_y,
            "bottom_y": bottom_y,
            "torso_angle": torso_angle,
            "aspect_ratio": aspect_ratio,
            "bbox": bbox,
        })

        if transition is None:
            transition = self._transition_signal(state, posture, center_y, bottom_y, torso_angle, aspect_ratio)

        if posture == "upright":
            state.normal_frames += 1
        else:
            state.normal_frames = 0

        print(
            f"[FALL MOTION] camera={camera_id} track_id={track_id} delta_y={((center_y - state.last_center_y) if state.last_center_y is not None and center_y is not None else 0.0):.2f} "
            f"delta_bottom_y={((bottom_y - state.last_bottom_y) if state.last_bottom_y is not None and bottom_y is not None else 0.0):.2f} "
            f"previous_posture={state.last_posture} current_posture={posture} transition={bool(transition)}"
        )

        if state.state == "NORMAL":
            if posture == "fallen" and transition:
                state.state = "TRANSITION"
                state.candidate_frames = 1
                state.last_transition_ts = now
                if self.confirmation_frames <= 1:
                    state.state = "ALERTED"
                    state.recovery_frames = 0
                    print(f"[FALL CONFIRMED] camera={camera_id} track_id={track_id} reason=FALL_TRANSITION_PLUS_SUSTAINED_FALLEN_POSTURE")
                    state.last_posture = posture
                    state.last_center_y = center_y
                    state.last_bottom_y = bottom_y
                    state.last_torso_angle = torso_angle
                    state.last_aspect_ratio = aspect_ratio
                    return True
                print(f"[FALL STATE] camera={camera_id} track_id={track_id} state={state.state} candidate_frames={state.candidate_frames} normal_frames={state.normal_frames}")
                state.last_posture = posture
                state.last_center_y = center_y
                state.last_bottom_y = bottom_y
                state.last_torso_angle = torso_angle
                state.last_aspect_ratio = aspect_ratio
                return False
            if posture in {"sitting", "bending", "crouching"}:
                print(f"[FALL REJECTED] camera={camera_id} track_id={track_id} reason=NORMAL_{posture.upper()}")
            elif posture == "ambiguous" and not transition:
                print(f"[FALL REJECTED] camera={camera_id} track_id={track_id} reason=INSUFFICIENT_FALL_TRANSITION")
            state.last_posture = posture
            state.last_center_y = center_y
            state.last_bottom_y = bottom_y
            state.last_torso_angle = torso_angle
            state.last_aspect_ratio = aspect_ratio
            return False

        if state.state == "TRANSITION":
            if posture == "fallen" and transition:
                state.candidate_frames += 1
                if state.candidate_frames >= self.confirmation_frames:
                    state.state = "ALERTED"
                    state.recovery_frames = 0
                    print(f"[FALL CONFIRMED] camera={camera_id} track_id={track_id} reason=FALL_TRANSITION_PLUS_SUSTAINED_FALLEN_POSTURE")
                    state.last_posture = posture
                    state.last_center_y = center_y
                    state.last_bottom_y = bottom_y
                    state.last_torso_angle = torso_angle
                    state.last_aspect_ratio = aspect_ratio
                    return True
                state.state = "CANDIDATE"
                print(f"[FALL STATE] camera={camera_id} track_id={track_id} state={state.state} candidate_frames={state.candidate_frames} normal_frames={state.normal_frames}")
            elif posture == "fallen":
                state.candidate_frames = 1
                state.state = "CANDIDATE"
            elif self._is_non_fall_posture(posture):
                state.state = "NORMAL"
                state.candidate_frames = 0
                print(f"[FALL REJECTED] camera={camera_id} track_id={track_id} reason=NORMAL_{posture.upper()}")
            state.last_posture = posture
            state.last_center_y = center_y
            state.last_bottom_y = bottom_y
            state.last_torso_angle = torso_angle
            state.last_aspect_ratio = aspect_ratio
            return False

        if state.state == "CANDIDATE":
            if posture == "fallen":
                state.candidate_frames += 1
                print(f"[FALL STATE] camera={camera_id} track_id={track_id} state={state.state} candidate_frames={state.candidate_frames} normal_frames={state.normal_frames}")
                if state.candidate_frames >= self.confirmation_frames:
                    state.state = "ALERTED"
                    state.recovery_frames = 0
                    print(f"[FALL CONFIRMED] camera={camera_id} track_id={track_id} reason=FALL_TRANSITION_PLUS_SUSTAINED_FALLEN_POSTURE")
                    state.last_posture = posture
                    state.last_center_y = center_y
                    state.last_bottom_y = bottom_y
                    state.last_torso_angle = torso_angle
                    state.last_aspect_ratio = aspect_ratio
                    return True
            elif self._is_non_fall_posture(posture):
                state.state = "NORMAL"
                state.candidate_frames = 0
                print(f"[FALL REJECTED] camera={camera_id} track_id={track_id} reason=NORMAL_{posture.upper()}")
            state.last_posture = posture
            state.last_center_y = center_y
            state.last_bottom_y = bottom_y
            state.last_torso_angle = torso_angle
            state.last_aspect_ratio = aspect_ratio
            return False

        if state.state == "ALERTED":
            if posture == "upright":
                state.state, state.recovery_frames = "RECOVERING", 1
            state.last_posture = posture
            state.last_center_y = center_y
            state.last_bottom_y = bottom_y
            state.last_torso_angle = torso_angle
            state.last_aspect_ratio = aspect_ratio
            return False

        if state.state == "RECOVERING":
            if posture == "upright":
                state.recovery_frames += 1
                if state.recovery_frames >= self.recovery_frames_required:
                    state.state = "NORMAL"
                    state.candidate_frames = 0
                    state.normal_frames = 0
            elif posture == "fallen":
                state.state, state.recovery_frames = "ALERTED", 0
            state.last_posture = posture
            state.last_center_y = center_y
            state.last_bottom_y = bottom_y
            state.last_torso_angle = torso_angle
            state.last_aspect_ratio = aspect_ratio
            return False
        return False

    def cleanup(self, now=None):
        now = time.time() if now is None else now
        self.states = {
            key: value for key, value in self.states.items()
            if now - value.last_seen <= self.timeout_seconds
        }
