"""Shared model initialization compatibility facade."""

from alcon_stream_new import initialize_face_model


class AIModels:
    """References the process-wide models loaded by the existing application."""

    def load(self):
        initialize_face_model()
        return self

    @property
    def person_model(self):
        from alcon_stream_new import person_detector
        return person_detector

    @property
    def vehicle_model(self):
        from alcon_stream_new import vehicle_detector
        return vehicle_detector

    @property
    def face_model(self):
        from alcon_stream_new import face_app
        return face_app


__all__ = ["AIModels"]
