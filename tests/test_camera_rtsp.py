import unittest

from camera.manager import ManagedWorker


class ManagedWorkerRTSPUrlTest(unittest.TestCase):
    def test_managed_worker_retains_unique_rtsp_url_for_each_camera(self):
        def target(camera, rtsp_url):
            return camera["camera_id"], rtsp_url

        camera_a = {
            "camera_id": "CAM001",
            "nvr_ip": "115.247.225.82",
            "channel": 1,
            "subtype": 1,
        }
        camera_b = {
            "camera_id": "CAM002",
            "nvr_ip": "115.247.225.82",
            "channel": 2,
            "subtype": 1,
        }

        worker_a = ManagedWorker(
            camera_a,
            target,
            lambda camera_id: None,
            rtsp_url="rtsp://user:pass@115.247.225.82:554/cam/realmonitor?channel=1&subtype=1",
        )
        worker_b = ManagedWorker(
            camera_b,
            target,
            lambda camera_id: None,
            rtsp_url="rtsp://user:pass@115.247.225.82:554/cam/realmonitor?channel=2&subtype=1",
        )

        self.assertEqual(
            worker_a.rtsp_url,
            "rtsp://user:pass@115.247.225.82:554/cam/realmonitor?channel=1&subtype=1",
        )
        self.assertEqual(
            worker_b.rtsp_url,
            "rtsp://user:pass@115.247.225.82:554/cam/realmonitor?channel=2&subtype=1",
        )
        self.assertEqual(worker_a.camera_id, "CAM001")
        self.assertEqual(worker_b.camera_id, "CAM002")


if __name__ == "__main__":
    unittest.main()
