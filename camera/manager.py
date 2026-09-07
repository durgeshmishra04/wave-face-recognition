"""Camera worker lifecycle management."""

import threading
import time


class ManagedWorker:
    def __init__(self, camera_config, target, release_resources):
        self.camera_config = camera_config
        self.camera_id = camera_config["camera_id"]
        self.stop_event = threading.Event()
        self.release_resources = release_resources
        self.thread = threading.Thread(
            target=target,
            args=(camera_config,),
            name=f"camera-{self.camera_id}",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def stop(self):
        print(f"[INFO][{self.camera_id}] Stopping")
        self.stop_event.set()
        self.release_resources(self.camera_id)

    def join(self, timeout):
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()


class CameraManager:
    """Create, start, stop, and inspect one worker per enabled camera."""

    def __init__(self, cameras, worker_target, shutdown_event=None,
                 release_resources=None):
        self.cameras = list(cameras)
        self.worker_target = worker_target
        self.shutdown_event = shutdown_event or threading.Event()
        self.release_resources = release_resources or (lambda camera_id: None)
        self.workers = {}
        self._stop_lock = threading.Lock()
        self._stopped = False

    def start(self):
        for camera_config in self.cameras:
            camera_id = camera_config["camera_id"]
            if camera_id in self.workers:
                continue
            worker = ManagedWorker(
                camera_config,
                self.worker_target,
                self.release_resources,
            )
            self.workers[camera_id] = worker
            worker.start()

    def stop_all(self, timeout=5):
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
            self.shutdown_event.set()
            for worker in self.workers.values():
                worker.stop()
        deadline = time.monotonic() + timeout
        for worker in self.workers.values():
            remaining = max(0.0, deadline - time.monotonic())
            if worker.join(timeout=remaining):
                print(f"[INFO][{worker.camera_id}] Worker stopped")
            else:
                print(
                    f"[WARN][{worker.camera_id}] "
                    "Worker did not stop within timeout"
                )
        print("[INFO] All camera workers stopped")

    def stop(self, timeout=5):
        self.stop_all(timeout=timeout)

    def worker(self, camera_id):
        return self.workers.get(camera_id)


__all__ = ["CameraManager"]
