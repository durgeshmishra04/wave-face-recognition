import numpy as np

from fall_detection.fall_detector import FallDetector
from fall_detection.fall_tracker import FallTracker


class DummyModel:
    def predict(self, *args, **kwargs):
        return []


def _pose_keypoints():
    # shoulders, hips, knees, ankles. 17 keypoints total, with body bent forward but not lying down.
    pts = np.zeros((17, 2), dtype=float)
    pts[5] = [120.0, 110.0]
    pts[6] = [80.0, 110.0]
    pts[11] = [118.0, 170.0]
    pts[12] = [82.0, 170.0]
    pts[13] = [110.0, 210.0]
    pts[14] = [90.0, 210.0]
    pts[15] = [115.0, 255.0]
    pts[16] = [85.0, 255.0]
    return pts


def test_posture_does_not_mark_bent_pose_as_fallen():
    detector = FallDetector(model_path="dummy.pt", model_loader=lambda path: DummyModel())
    posture = detector._posture(
        pose_box=[20.0, 20.0, 180.0, 220.0],
        keypoints=_pose_keypoints(),
        confidences=np.full(17, 0.95),
    )

    assert posture not in {"fallen"}
    assert posture in {"upright", "bending", "sitting", "crouching", "ambiguous"}


def test_tracker_requires_upright_transition_before_fall_candidate():
    tracker = FallTracker(confirmation_frames=2)

    assert tracker.update("cam1", 7, "upright", now=1.0) is False
    assert tracker.update("cam1", 7, "fallen", now=2.0) is False
    assert tracker.update("cam1", 7, "fallen", now=3.0) is False
    assert tracker.update("cam1", 7, "fallen", now=4.0) is False


def test_tracker_rejects_static_sitting_or_bending_as_fall():
    tracker = FallTracker(confirmation_frames=2)

    assert tracker.update("cam1", 8, "upright", now=1.0, center_y=200.0, bottom_y=240.0, torso_angle=10.0, aspect_ratio=0.5) is False
    assert tracker.update("cam1", 8, "sitting", now=2.0, center_y=220.0, bottom_y=250.0, torso_angle=60.0, aspect_ratio=0.8) is False
    assert tracker.update("cam1", 8, "bending", now=3.0, center_y=225.0, bottom_y=255.0, torso_angle=55.0, aspect_ratio=0.9) is False
    assert tracker.update("cam1", 8, "upright", now=4.0, center_y=200.0, bottom_y=240.0, torso_angle=10.0, aspect_ratio=0.5) is False


def test_tracker_confirms_long_fall_only_after_transition():
    tracker = FallTracker(confirmation_frames=2)

    assert tracker.update("cam1", 9, "upright", now=1.0, center_y=180.0, bottom_y=220.0, torso_angle=10.0, aspect_ratio=0.5) is False
    assert tracker.update("cam1", 9, "fallen", now=2.0, center_y=210.0, bottom_y=260.0, torso_angle=70.0, aspect_ratio=1.1, transition=True) is False
    assert tracker.update("cam1", 9, "fallen", now=3.0, center_y=212.0, bottom_y=262.0, torso_angle=72.0, aspect_ratio=1.15, transition=True) is True


def test_tracker_blocks_duplicate_alerts_until_recovery():
    tracker = FallTracker(confirmation_frames=1, recovery_frames=2)

    assert tracker.update("cam1", 10, "upright", now=1.0, center_y=180.0, bottom_y=220.0, torso_angle=10.0, aspect_ratio=0.5) is False
    assert tracker.update("cam1", 10, "fallen", now=2.0, center_y=220.0, bottom_y=270.0, torso_angle=75.0, aspect_ratio=1.2, transition=True) is True
    assert tracker.update("cam1", 10, "fallen", now=3.0, center_y=222.0, bottom_y=272.0, torso_angle=74.0, aspect_ratio=1.2) is False
    assert tracker.update("cam1", 10, "upright", now=4.0, center_y=180.0, bottom_y=220.0, torso_angle=10.0, aspect_ratio=0.5) is False
    assert tracker.update("cam1", 10, "upright", now=5.0, center_y=180.0, bottom_y=220.0, torso_angle=10.0, aspect_ratio=0.5) is False
    assert tracker.update("cam1", 10, "fallen", now=6.0, center_y=220.0, bottom_y=270.0, torso_angle=75.0, aspect_ratio=1.2, transition=True) is True
