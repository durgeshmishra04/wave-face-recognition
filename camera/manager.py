"""Camera worker lifecycle management."""

import threading
import time


class LatestFrameStore:
    """Bounded latest-frame storage shared by camera capture and AI workers."""

    def __init__(self):
        self._frames = {}
        self._conditions = {}
        self._lock = threading.Lock()

    def publish(self, camera_id, stream_type, frame):
        key = (str(camera_id), str(stream_type).lower())
        with self._lock:
            frame_id = self._frames.get(key, {}).get("frame_id", 0) + 1
            packet = {
                "frame": frame,
                "frame_id": frame_id,
                "capture_timestamp": time.time(),
                "stream_type": key[1],
                "camera_id": key[0],
            }
            self._frames[key] = packet
            condition = self._conditions.setdefault(key, threading.Condition())
        with condition:
            condition.notify_all()
        return packet

    def get(self, camera_id, stream_type="main", after_frame_id=None):
        key = (str(camera_id), str(stream_type).lower())
        with self._lock:
            packet = self._frames.get(key)
            if packet is None:
                return None
            if after_frame_id is not None and packet["frame_id"] <= after_frame_id:
                return None
            return packet

    def wait_for_new(self, camera_id, stream_type="main", after_frame_id=None,
                     timeout=0.2):
        packet = self.get(camera_id, stream_type, after_frame_id)
        if packet is not None:
            return packet
        key = (str(camera_id), str(stream_type).lower())
        with self._lock:
            condition = self._conditions.setdefault(key, threading.Condition())
        with condition:
            condition.wait(timeout=max(0.0, float(timeout)))
        return self.get(camera_id, stream_type, after_frame_id)


class ManagedWorker:
    def __init__(self, camera_config, target, release_resources, rtsp_url=None):
        self.camera_config = camera_config
        self.camera_id = camera_config["camera_id"]
        self.rtsp_url = rtsp_url
        self.stop_event = threading.Event()
        self.release_resources = release_resources
        self.thread = threading.Thread(
            target=target,
            args=(camera_config, self.rtsp_url),
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
                 release_resources=None, rtsp_url_builder=None,
                 frame_store=None):
        self.cameras = list(cameras)
        self.worker_target = worker_target
        self.shutdown_event = shutdown_event or threading.Event()
        self.release_resources = release_resources or (lambda camera_id: None)
        self.rtsp_url_builder = rtsp_url_builder
        self.frame_store = frame_store or LatestFrameStore()
        self.workers = {}
        self._stop_lock = threading.Lock()
        self._stopped = False

    def start(self):
        for camera_config in self.cameras:
            camera_id = camera_config["camera_id"]
            if camera_id in self.workers:
                continue
            rtsp_url = None
            if self.rtsp_url_builder is not None:
                try:
                    rtsp_url = self.rtsp_url_builder(camera_config)
                except Exception as exc:
                    print(f"[WARN][{camera_id}] RTSP URL generation failed: {exc}")
            worker = ManagedWorker(
                camera_config,
                self.worker_target,
                self.release_resources,
                rtsp_url=rtsp_url,
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

    def get_latest_frame(self, camera_id, stream="main"):
        return self.frame_store.get(camera_id, stream)


__all__ = ["CameraManager"]
