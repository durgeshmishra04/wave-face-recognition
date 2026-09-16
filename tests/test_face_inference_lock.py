import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import alcon_stream_new
from roi.camera_rois import CAMERA_ROIS, get_camera_roi


def test_recognize_face_uses_best_individual_template(monkeypatch):
    monkeypatch.setattr(
        alcon_stream_new,
        "known_face_templates",
        {
            "REG_1": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
            "REG_2": [
                np.array([0.7, 0.7], dtype=np.float32),
                np.array([0.2, 0.98], dtype=np.float32),
            ],
        },
    )
    monkeypatch.setattr(
        alcon_stream_new,
        "known_person_metadata",
        {
            "REG_1": {"employee_name": "Alice", "person_name": "Alice", "employee_id": "E-1"},
            "REG_2": {"employee_name": "Bob", "person_name": "Bob", "employee_id": "E-2"},
        },
    )
    monkeypatch.setattr(alcon_stream_new, "known_embeddings", {"REG_1": [1.0, 0.0], "REG_2": [0.7, 0.7]})

    face = SimpleNamespace(embedding=np.array([1.0, 0.0], dtype=np.float32), person_id=None)

    name, score = alcon_stream_new.recognize_face(face, camera_id="CAM01")

    assert name == "Alice"
    assert score == pytest.approx(1.0)
    assert face.person_id == "E-1"


def test_recognize_face_uses_second_best_person_for_margin(monkeypatch):
    monkeypatch.setattr(
        alcon_stream_new,
        "known_face_templates",
        {
            "REG_1": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.99, 0.1], dtype=np.float32),
            ],
            "REG_2": [np.array([0.98, 0.2], dtype=np.float32)],
        },
    )
    monkeypatch.setattr(
        alcon_stream_new,
        "known_person_metadata",
        {
            "REG_1": {"employee_name": "Alice", "employee_id": "E-1"},
            "REG_2": {"employee_name": "Bob", "employee_id": "E-2"},
        },
    )

    face = SimpleNamespace(
        embedding=np.array([1.0, 0.0], dtype=np.float32), person_id=None
    )
    name, score = alcon_stream_new.recognize_face(face, camera_id="CAM01")

    assert name == "Unknown"
    assert score == pytest.approx(1.0)
    assert face.recognition_second_score == pytest.approx(0.98)
    assert face.recognition_margin == pytest.approx(0.02)
    assert face.recognition_decision == "PENDING"


def test_recognize_face_keeps_candidate_and_unknown_distinct(monkeypatch):
    monkeypatch.setattr(
        alcon_stream_new,
        "known_face_templates",
        {"REG_1": [np.array([1.0, 0.0], dtype=np.float32)]},
    )
    monkeypatch.setattr(
        alcon_stream_new,
        "known_person_metadata",
        {"REG_1": {"employee_name": "Alice", "employee_id": "E-1"}},
    )

    weak_face = SimpleNamespace(
        embedding=np.array([0.46, np.sqrt(1 - 0.46**2)], dtype=np.float32),
        person_id=None,
    )
    name, _ = alcon_stream_new.recognize_face(weak_face, camera_id="CAM01")
    assert name == "Unknown"
    assert weak_face.recognition_candidate_name == "Alice"
    assert weak_face.recognition_decision == "PENDING"

    unknown_face = SimpleNamespace(
        embedding=np.array([0.38, np.sqrt(1 - 0.38**2)], dtype=np.float32),
        person_id=None,
    )
    name, _ = alcon_stream_new.recognize_face(unknown_face, camera_id="CAM01")
    assert name == "Unknown"
    assert unknown_face.recognition_candidate_name == "Alice"
    assert unknown_face.recognition_decision == "UNKNOWN"


def test_identity_track_requires_spatial_continuity():
    assert alcon_stream_new._same_face_track(
        np.array([102, 102, 198, 198]),
        np.array([100, 100, 200, 200]),
    )
    assert not alcon_stream_new._same_face_track(
        np.array([420, 420, 520, 520]),
        np.array([100, 100, 200, 200]),
    )


def test_identity_track_configuration_preserves_strong_threshold():
    assert alcon_stream_new.RECOGNITION_THRESHOLD == pytest.approx(0.50)
    assert alcon_stream_new.FACE_RECOGNITION_MIN_DET_SCORE == pytest.approx(0.40)
    assert alcon_stream_new.KNOWN_IDENTITY_MEMORY_TIMEOUT == pytest.approx(2.0)


def test_face_association_requires_a_person_box():
    face_box = np.array([100, 100, 180, 190], dtype=np.int32)
    person_box = np.array([60, 60, 240, 400], dtype=np.int32)

    assert alcon_stream_new.face_associated_with_person(
        face_box,
        [person_box],
        (720, 1280, 3),
    ) is not None
    assert alcon_stream_new.face_associated_with_person(
        face_box,
        [],
        (720, 1280, 3),
    ) is None


def test_vehicle_alert_roi_uses_configured_rectangle():
    assert alcon_stream_new.vehicle_in_roi(
        np.array([400, 300, 520, 450]),
        1280,
        720,
    )
    assert not alcon_stream_new.vehicle_in_roi(
        np.array([400, 520, 520, 700]),
        1280,
        720,
    )


def test_vehicle_roi_rejects_box_outside_configured_polygon(monkeypatch):
    original_vehicle = CAMERA_ROIS["CAM002"]["vehicle"]
    CAMERA_ROIS["CAM002"]["vehicle"] = {
        "points": np.array([
            [0.10, 0.10],
            [0.90, 0.10],
            [0.10, 0.90],
        ], dtype=np.float32),
    }
    try:
        assert alcon_stream_new.vehicle_in_roi(
            np.array([100, 100, 180, 180]),
            1000,
            1000,
            camera_id="CAM002",
        )
        assert not alcon_stream_new.vehicle_in_roi(
            np.array([700, 700, 780, 780]),
            1000,
            1000,
            camera_id="CAM002",
        )
    finally:
        CAMERA_ROIS["CAM002"]["vehicle"] = original_vehicle


def test_camera_rois_are_isolated(monkeypatch):
    original_cam001 = get_camera_roi("CAM001")
    original_cam002 = get_camera_roi("CAM002")
    try:
        CAMERA_ROIS["CAM001"]["points"][0] = [0.50, 0.50]
        CAMERA_ROIS["CAM001"]["vehicle"]["left"] = 0.25

        cam001 = get_camera_roi("CAM001")
        cam002 = get_camera_roi("CAM002")

        assert cam001["points"][0].tolist() == [0.5, 0.5]
        assert cam001["vehicle"]["left"] == pytest.approx(0.25)
        assert cam002["points"][0].tolist() == original_cam002["points"][0].tolist()
        assert cam002["vehicle"]["left"] == original_cam002["vehicle"]["left"]
        np.testing.assert_array_equal(
            cam002["vehicle"]["points"],
            original_cam002["vehicle"]["points"],
        )
    finally:
        CAMERA_ROIS["CAM001"] = original_cam001


def test_person_box_deduplication_keeps_highest_confidence():
    boxes = alcon_stream_new.deduplicate_person_boxes(
        [
            np.array([100, 100, 300, 500]),
            np.array([110, 105, 295, 495]),
            np.array([500, 100, 700, 500]),
        ],
        [0.60, 0.90, 0.70],
        iou_threshold=0.50,
    )

    assert len(boxes) == 2
    assert any(np.array_equal(box, np.array([110, 105, 295, 495])) for box in boxes)
    assert any(np.array_equal(box, np.array([500, 100, 700, 500])) for box in boxes)


def test_vehicle_deduplication_keeps_separate_nearby_vehicles():
    vehicles = alcon_stream_new.deduplicate_vehicle_detections(
        [
            {"box": np.array([100, 100, 300, 300]), "vehicle_type": "four_wheeler", "confidence": 0.60},
            {"box": np.array([110, 105, 295, 295]), "vehicle_type": "four_wheeler", "confidence": 0.90},
            {"box": np.array([500, 100, 700, 300]), "vehicle_type": "four_wheeler", "confidence": 0.70},
        ],
        iou_threshold=0.50,
    )

    assert len(vehicles) == 2


def test_person_track_preserves_established_identity_fields():
    tracks = []
    next_track_id = alcon_stream_new._update_person_tracks(
        tracks,
        [np.array([100, 100, 300, 500])],
        now=1.0,
        next_track_id=1,
    )
    tracks[0].update({
        "identity_status": "KNOWN",
        "employee_name": "Ramesh",
        "employee_id": "E-1",
        "last_verified_score": 0.65,
        "annotation_box": np.array([150, 130, 250, 240]),
    })

    next_track_id = alcon_stream_new._update_person_tracks(
        tracks,
        [np.array([108, 105, 305, 505])],
        now=2.0,
        next_track_id=next_track_id,
    )

    assert next_track_id == 2
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == 1
    assert tracks[0]["identity_status"] == "KNOWN"
    assert tracks[0]["employee_name"] == "Ramesh"


def test_registration_quality_rejects_tiny_face():
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    face = SimpleNamespace(
        bbox=np.array([480, 480, 540, 540], dtype=np.float32),
        det_score=0.95,
    )

    with pytest.raises(ValueError, match="too small"):
        alcon_stream_new._validate_registration_face_quality(face, image, 1)


def test_registration_quality_accepts_large_sharp_face():
    image = np.indices((1000, 1000), dtype=np.uint8).sum(axis=0) % 2
    image = (np.repeat(image[:, :, None], 3, axis=2) * 255).astype(np.uint8)
    face = SimpleNamespace(
        bbox=np.array([250, 200, 750, 800], dtype=np.float32),
        det_score=0.95,
    )

    alcon_stream_new._validate_registration_face_quality(face, image, 1)


def test_refresh_known_faces_cache_reloads_when_forced(monkeypatch):
    calls = {"count": 0}

    def fake_load_known_faces():
        calls["count"] += 1
        alcon_stream_new.known_face_templates = {"REG_1": [np.array([1.0, 0.0], dtype=np.float32)]}
        alcon_stream_new.known_person_metadata = {"REG_1": {"employee_name": "Alice", "employee_id": "E-1"}}

    monkeypatch.setattr(alcon_stream_new, "load_known_faces", fake_load_known_faces)
    monkeypatch.setattr(alcon_stream_new, "known_face_templates", {"OLD": [np.array([0.0, 1.0], dtype=np.float32)]})

    result = alcon_stream_new.refresh_known_faces_cache(force=True)

    assert result is True
    assert calls["count"] == 1
    assert "REG_1" in alcon_stream_new.known_face_templates


def test_safe_face_inference_serializes_gpu_call(monkeypatch):
    events = []

    class DummyLock:
        def __enter__(self):
            events.append("lock-enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("lock-exit")

    class DummyFaceApp:
        def __init__(self):
            self.calls = []

        def get(self, frame):
            events.append(("infer", frame))
            self.calls.append(frame)
            return ["face"]

    dummy_lock = DummyLock()
    dummy_face_app = DummyFaceApp()

    monkeypatch.setattr(alcon_stream_new, "face_inference_lock", dummy_lock)
    monkeypatch.setattr(alcon_stream_new, "face_app", dummy_face_app)

    result = alcon_stream_new.safe_face_inference("frame-123")

    assert result == ["face"]
    assert events[0] == "lock-enter"
    assert events[1] == ("infer", "frame-123")
    assert events[-1] == "lock-exit"


def test_gpu_inference_helpers_serialize_all_gpu_models(monkeypatch):
    events = []

    class DummyLock:
        def __enter__(self):
            events.append("lock-enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append("lock-exit")

    class DummyDetector:
        def predict(self, frame, **kwargs):
            events.append(("predict", frame, kwargs))
            return ["person"]

    dummy_lock = DummyLock()
    monkeypatch.setattr(alcon_stream_new, "gpu_inference_lock", dummy_lock)
    monkeypatch.setattr(alcon_stream_new, "person_detector", DummyDetector())

    result = alcon_stream_new.safe_person_inference("frame-person")

    assert result == ["person"]
    assert events[0] == "lock-enter"
    assert events[1] == (
        "predict",
        "frame-person",
        {
            "classes": [0],
            "conf": alcon_stream_new.PERSON_CONFIDENCE,
            "iou": alcon_stream_new.PERSON_IOU,
            "verbose": False,
        },
    )
    assert events[-1] == "lock-exit"


def test_register_person_refreshes_known_face_cache(monkeypatch):
    called = {"count": 0}

    def fake_load_known_faces():
        called["count"] += 1
        alcon_stream_new.known_embeddings = {"REG_123": [1.0, 0.0]}
        alcon_stream_new.known_person_metadata = {"REG_123": {"employee_name": "Alice", "employee_id": "E-1"}}

    monkeypatch.setattr(alcon_stream_new, "load_known_faces", fake_load_known_faces)
    monkeypatch.setattr(alcon_stream_new, "_validate_registration_images", lambda: ([], []))
    monkeypatch.setattr(alcon_stream_new, "_save_registered_person", lambda *args, **kwargs: "REG_123")

    with alcon_stream_new.app.test_client() as client:
        response = client.post(
            "/api/register-person",
            data={
                "gate_no": "G1",
                "employee_name": "Alice",
                "designation": "Manager",
                "employee_id": "E-1",
            },
        )

    assert response.status_code == 201
    assert called["count"] == 1


def test_registration_decoder_returns_opencv_image():
    image_path = Path(__file__).parents[1] / "tools" / "image" / "r.jpeg"
    if not image_path.is_file():
        pytest.skip("registration diagnostic image is not present")

    image, orientation = alcon_stream_new._decode_registration_image(
        image_path.read_bytes()
    )

    assert image is not None
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    assert orientation is None


def test_registration_face_inference_retries_larger_detector_size(monkeypatch):
    calls = []
    face = SimpleNamespace(
        bbox=np.array([10, 10, 100, 100], dtype=np.float32),
        det_score=0.9,
        embedding=np.ones(512, dtype=np.float32),
    )

    class FakeFaceApp:
        models = {}

        def prepare(self, **kwargs):
            calls.append(("prepare", kwargs["det_size"], kwargs["det_thresh"]))

        def get(self, _frame):
            calls.append(("get",))
            return [] if len([call for call in calls if call[0] == "get"]) == 1 else [face]

    monkeypatch.setattr(alcon_stream_new, "face_app", FakeFaceApp())
    monkeypatch.setattr(
        alcon_stream_new,
        "_face_runtime_diagnostics",
        lambda: {"providers": ["CPUExecutionProvider"]},
    )

    result = alcon_stream_new.safe_registration_face_inference(
        np.zeros((200, 200, 3), dtype=np.uint8)
    )

    assert result == [face]
    assert calls == [
        ("get",),
        ("prepare", alcon_stream_new.REGISTRATION_FALLBACK_DET_SIZE, alcon_stream_new.DET_THRESH),
        ("get",),
        ("prepare", alcon_stream_new.DET_SIZE, alcon_stream_new.DET_THRESH),
    ]


def test_detect_vehicle_boxes_rejects_wall_like_false_positive(monkeypatch):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    class DummyArray:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class DummyResults:
        def __init__(self):
            self.boxes = SimpleNamespace(
                xyxy=DummyArray([
                    [10, 10, 150, 700],
                    [250, 420, 550, 540],
                ]),
                cls=DummyArray([2, 2]),
                conf=DummyArray([0.92, 0.91]),
            )
            self.names = {2: "car"}

    monkeypatch.setattr(alcon_stream_new, "safe_vehicle_inference", lambda frame: [DummyResults()])

    vehicles = alcon_stream_new.detect_vehicle_boxes(frame)

    assert len(vehicles) == 1
    assert vehicles[0]["class_name"] == "car"
    assert vehicles[0]["box"][0] == 250
    assert vehicles[0]["box"][2] == 550
