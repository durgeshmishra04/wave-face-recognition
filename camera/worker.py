"""Compatibility export for the existing camera worker pipeline."""

from alcon_stream_new import camera_worker

CameraWorker = camera_worker

__all__ = ["CameraWorker", "camera_worker"]
