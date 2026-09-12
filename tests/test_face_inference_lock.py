import threading

import alcon_stream_new


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
