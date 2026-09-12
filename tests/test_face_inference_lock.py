import threading
from types import SimpleNamespace

import numpy as np
import pytest

import alcon_stream_new


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
    assert events[1] == ("predict", "frame-person", {})
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
