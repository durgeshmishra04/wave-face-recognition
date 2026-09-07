"""Camera worker lifecycle management."""

import threading


class CameraManager:
    """Create, start, stop, and inspect one worker per enabled camera."""

    def __init__(self, cameras, worker_target, shutdown_event=None):
        self.cameras = list(cameras)
        self.worker_target = worker_target
        self.shutdown_event = shutdown_event or threading.Event()
        self.workers = {}

    def start(self):
        for camera_config in self.cameras:
            camera_id = camera_config["camera_id"]
            if camera_id in self.workers:
                continue
            worker = threading.Thread(
                target=self.worker_target,
                args=(camera_config,),
                name=f"camera-{camera_id}",
                daemon=True,
            )
            self.workers[camera_id] = worker
            worker.start()

    def stop(self, timeout=5):
        self.shutdown_event.set()
        for worker in self.workers.values():
            worker.join(timeout=timeout)

    def worker(self, camera_id):
        return self.workers.get(camera_id)


__all__ = ["CameraManager"]
