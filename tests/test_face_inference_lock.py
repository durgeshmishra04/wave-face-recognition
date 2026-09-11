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
