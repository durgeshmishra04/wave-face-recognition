from types import SimpleNamespace

import numpy as np

from events.detection_events import DetectionEventManager


class MemoryDatabase:
    def __init__(self):
        self.events = []

    def save(self, event):
        self.events.append(event.copy())
        return len(self.events)


class MemorySocket:
    def __init__(self):
        self.events = []

    def emit(self, name, event):
        self.events.append((name, event.copy()))


def _manager(tmp_path, fcm_sender=None):
    database = MemoryDatabase()
    socket = MemorySocket()
    manager = DetectionEventManager(
        database=database,
        image_dir=tmp_path,
        public_base_url="https://example.invalid",
        socketio=socket,
        fcm_sender=fcm_sender,
        dedup_seconds=0.1,
        exit_frame_offset=0,
        camera_id="CAM001",
        camera_name="Gate 1",
    )
    return manager, database, socket


def _valid_face_validation():
    return {
        "valid_person": True,
        "person_class": 0,
        "person_inside_roi": True,
        "person_crop_valid": True,
        "valid_face": True,
        "face_inside_person": True,
        "human_face_valid": True,
        "face_belongs_to_person": True,
        "unknown_face_score_valid": True,
        "source": "YOLO_PERSON_CROP",
    }


def test_pending_person_exit_does_not_create_unknown_event(tmp_path):
    manager, database, socket = _manager(tmp_path)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    pending_person = SimpleNamespace(
        recognized_name="Pending",
        recognition_state="PENDING",
        recognition_decision="PENDING",
        face_validation=None,
        associated_person_box=np.array([10, 10, 50, 70]),
        annotation_box=np.array([20, 15, 35, 30]),
        recognition_score=0.44,
        track_id=7,
    )

    manager.process_frame(frame, [pending_person], [], "Gate 1", 1.0)
    records = manager.process_frame(frame, [], [], "Gate 1", 2.0)

    assert records == []
    assert database.events == []
    assert socket.events == []
    assert list(tmp_path.iterdir()) == []


def test_unknown_publisher_rejects_missing_validation(tmp_path):
    manager, database, socket = _manager(tmp_path)
    event = {
        "detection_type": "unknown_person",
        "recognition_state": "UNKNOWN",
        "recognition_decision": "UNKNOWN",
        "face_validation": None,
        "detected_at": 1.0,
        "gate_name": "Gate 1",
        "title": "Unknown Person Detected",
        "message": "Unknown person detected at Gate 1",
    }

    result = manager._publish(event, np.zeros((80, 80, 3), dtype=np.uint8), notify=True)

    assert result is None
    assert database.events == []
    assert socket.events == []
    assert list(tmp_path.iterdir()) == []


def test_validated_unknown_can_use_existing_alert_pipeline(tmp_path):
    notifications = []

    def fcm_sender(**payload):
        notifications.append(payload)
        return {"sent": 1}

    manager, database, socket = _manager(tmp_path, fcm_sender=fcm_sender)
    event = {
        "detection_type": "unknown_person",
        "recognition_state": "UNKNOWN",
        "recognition_decision": "UNKNOWN",
        "face_validation": _valid_face_validation(),
        "detected_at": 1.0,
        "gate_name": "Gate 1",
        "title": "Unknown Person Detected",
        "message": "Unknown person detected at Gate 1",
        "confidence": 0.8,
        "unknown_count": 1,
    }

    result = manager._publish(event, np.zeros((80, 80, 3), dtype=np.uint8), notify=True)

    assert result is not None
    assert len(database.events) == 1
    assert [name for name, _ in socket.events] == ["detection_event", "face_alert"]
    assert len(notifications) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_only_validated_unknown_track_publishes_on_exit(tmp_path):
    notifications = []

    def fcm_sender(**payload):
        notifications.append(payload)
        return {"sent": 1}

    manager, database, socket = _manager(tmp_path, fcm_sender=fcm_sender)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    unknown_person = SimpleNamespace(
        recognized_name="Unknown",
        recognition_state="UNKNOWN",
        recognition_decision="UNKNOWN",
        face_validation=_valid_face_validation(),
        associated_person_box=np.array([10, 10, 50, 70]),
        annotation_box=np.array([20, 15, 35, 30]),
        recognition_score=0.8,
        person_id=None,
        track_id=9,
    )

    manager.process_frame(frame, [unknown_person], [], "Gate 1", 1.0)
    records = manager.process_frame(frame, [], [], "Gate 1", 2.0)

    assert len(records) == 1
    assert records[0]["detection_type"] == "unknown_person"
    assert len(database.events) == 1
    assert [name for name, _ in socket.events] == ["detection_event", "face_alert"]
    assert len(notifications) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_event_tracks_follow_yolo_track_ids_not_nearest_boxes(tmp_path):
    manager, _, _ = _manager(tmp_path)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def person(track_id, box):
        return SimpleNamespace(
            recognized_name="Pending",
            recognition_state="PENDING",
            recognition_decision="PENDING",
            associated_person_box=np.asarray(box),
            annotation_box=np.array([box[0] + 5, box[1] + 5, box[0] + 15, box[1] + 15]),
            track_id=track_id,
        )

    manager.process_frame(
        frame,
        [person(1, [5, 10, 40, 90]), person(2, [60, 10, 95, 90])],
        [],
        "Gate 1",
        1.0,
    )
    manager.process_frame(
        frame,
        [person(1, [60, 10, 95, 90]), person(2, [5, 10, 40, 90])],
        [],
        "Gate 1",
        1.05,
    )

    assert len(manager.active_person_events) == 2
    assert {
        (track["track_id"], tuple(track["last_box"]))
        for track in manager.active_person_events
    } == {
        (1, (60, 10, 95, 90)),
        (2, (5, 10, 40, 90)),
    }