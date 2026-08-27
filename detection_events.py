import json
import os
import time
import uuid
from pathlib import Path

import cv2
from firebase_admin import messaging

from detection_database import DetectionDatabase


class DetectionEventManager:

    def __init__(self, database, image_dir, public_base_url, socketio,
                 firebase_topic, dedup_seconds=10.0, exit_frame_offset=5):
        self.database = database
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")
        self.socketio = socketio
        self.firebase_topic = firebase_topic
        self.dedup_seconds = dedup_seconds
        self.exit_frame_offset = max(0, int(exit_frame_offset))
        self.active_person_events = []
        self.active_vehicle_events = []
        try:
            self.person_ids = json.loads(os.getenv("PERSON_IDS_JSON", "{}"))
        except json.JSONDecodeError:
            self.person_ids = {}

    def _save_image(self, frame):
        filename = f"{uuid.uuid4()}.jpg"
        image_path = self.image_dir / filename
        if not cv2.imwrite(
            str(image_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        ):
            raise RuntimeError("Could not save detection image")
        return f"{self.public_base_url}/alerts/{filename}"

    def _center(self, box):
        return (
            (int(box[0]) + int(box[2])) / 2,
            (int(box[1]) + int(box[3])) / 2,
        )

    def _match_track(self, tracks, key, box):
        center = self._center(box)
        best_track = None
        best_distance = None
        for track in tracks:
            if track["key"] != key:
                continue
            distance = (
                (center[0] - track["center"][0]) ** 2
                + (center[1] - track["center"][1]) ** 2
            ) ** 0.5
            threshold = max(
                100,
                int(max(
                    int(box[2]) - int(box[0]),
                    int(box[3]) - int(box[1]),
                )),
            )
            if distance <= threshold and (
                best_distance is None or distance < best_distance
            ):
                best_track = track
                best_distance = distance
        return best_track

    def _update_track(self, tracks, key, box, event, frame, now):
        track = self._match_track(tracks, key, box)
        if track is None:
            track = {
                "key": key,
                "event": event.copy(),
                "center": self._center(box),
                "last_box": box,
                "last_seen": now,
                "frames": [],
            }
            tracks.append(track)
        track["event"] = event.copy()
        track["center"] = self._center(box)
        track["last_box"] = box
        track["last_seen"] = now
        track["frames"].append(frame.copy())
        max_frames = self.exit_frame_offset + 1
        if len(track["frames"]) > max_frames:
            del track["frames"][:-max_frames]

    def _clear_unknown_near(self, box):
        center = self._center(box)
        remaining = []
        for track in self.active_person_events:
            if track["key"] != "unknown_person":
                remaining.append(track)
                continue
            distance = (
                (center[0] - track["center"][0]) ** 2
                + (center[1] - track["center"][1]) ** 2
            ) ** 0.5
            threshold = max(
                100,
                int(max(
                    int(box[2]) - int(box[0]),
                    int(box[3]) - int(box[1]),
                )),
            )
            if distance > threshold:
                remaining.append(track)
        self.active_person_events[:] = remaining

    def _finalize_track(self, track):
        event = track["event"].copy()
        frames = track["frames"]
        frame_index = max(0, len(frames) - 1 - self.exit_frame_offset)
        alert_frame = frames[frame_index] if frames else None
        notify = event["detection_type"] != "known_person"
        return self._record(None, event, notify, alert_frame)

    def _finalize_missing(self, tracks, now):
        records = []
        remaining = []
        expired = []
        for track in tracks:
            if now - track["last_seen"] >= self.dedup_seconds:
                expired.append(track)
            else:
                remaining.append(track)
        tracks[:] = remaining

        unknown_tracks = [
            track
            for track in expired
            if track["event"]["detection_type"] == "unknown_person"
        ]
        other_tracks = [
            track
            for track in expired
            if track["event"]["detection_type"] != "unknown_person"
        ]

        if unknown_tracks:
            event = unknown_tracks[0]["event"].copy()
            count = len(unknown_tracks)
            event["person_id"] = None
            event["person_name"] = f"{count} Unknown Persons"
            event["unknown_count"] = count
            event["title"] = f"{count} Unknown Persons Detected"
            event["message"] = (
                f"{count} unknown person"
                f"{'s' if count != 1 else ''} detected at {event['gate_name']}"
            )
            event["box"] = unknown_tracks[0]["last_box"]
            frames = unknown_tracks[0]["frames"]
            frame_index = max(0, len(frames) - 1 - self.exit_frame_offset)
            alert_frame = frames[frame_index] if frames else None
            records.append(self._record(None, event, True, alert_frame))

        records.extend(
            self._finalize_track(track)
            for track in other_tracks
        )
        return records

    def _vehicle_key(self, vehicle):
        vehicle_key = (
            vehicle["vehicle_type"],
            vehicle["class_name"],
        )
        return vehicle_key

    def _notify(self, event, image_url):
        title = event["title"]
        message = event["message"]
        notification_id = str(event["id"])
        messaging.send(
            messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=message,
                    image=image_url or None,
                ),
                android=messaging.AndroidConfig(priority="high"),
                data={
                    "type": event["detection_type"],
                    "notification_id": notification_id,
                    "title": title,
                    "message": message,
                    "detection_id": notification_id,
                    "gate": event["gate_name"],
                    "image_url": image_url,
                    "confidence": str(event.get("confidence", "")),
                    "detected_at": str(event["detected_at"]),
                },
                topic=self.firebase_topic,
            )
        )

    def _record(self, frame, event, notify, alert_frame=None):
        image_source = alert_frame if alert_frame is not None else frame
        if image_source is None:
            return None
        image_frame = image_source.copy()
        if alert_frame is None:
            if event["detection_type"] in {"known_person", "unknown_person"}:
                self._draw_person_label(image_frame, event)
            elif event["detection_type"] == "vehicle":
                self._draw_vehicle_label(image_frame, event)

        image_url = self._save_image(image_frame)
        event["image_url"] = image_url
        event_id = self.database.save(event)
        event["id"] = event_id
        event["type"] = event["detection_type"]
        event["time"] = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(event["detected_at"]),
        )
        event["gate"] = event["gate_name"]
        self.socketio.emit("detection_event", event)
        if event["detection_type"] == "unknown_person":
            self.socketio.emit("face_alert", event)
        if notify:
            try:
                self._notify(event, image_url)
            except Exception as error:
                print(f"[ERROR] Detection notification failed: {error}")
        return event

    def _draw_person_label(self, frame, event):
        label = event["person_name"]
        if event.get("person_id"):
            label += f" ID: {event['person_id']}"
        label += f" {event['confidence']:.2f}"
        cv2.putText(
            frame,
            label,
            (event["box"][0], max(20, event["box"][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_vehicle_label(self, frame, event):
        label = (
            f"{event['vehicle_type']} / {event['vehicle_class']} / "
            f"{event['confidence']:.2f}"
        )
        cv2.putText(
            frame,
            label,
            (event["box"][0], max(20, event["box"][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def process_frame(
        self,
        frame,
        faces,
        vehicles,
        gate_name,
        detected_at,
        alert_frame=None,
    ):
        records = self._finalize_missing(
            self.active_person_events,
            detected_at,
        )
        records.extend(
            self._finalize_missing(
                self.active_vehicle_events,
                detected_at,
            )
        )
        for face in faces:
            name = getattr(face, "recognized_name", "Unknown")
            confidence = float(getattr(face, "recognition_score", 0.0))
            box = tuple(int(value) for value in face.bbox)
            event_type = "known_person" if name != "Unknown" else "unknown_person"
            event = {
                "detected_at": detected_at,
                "detection_type": event_type,
                "person_id": getattr(
                    face,
                    "person_id",
                    self.person_ids.get(name),
                ),
                "person_name": name,
                "confidence": confidence,
                "gate_name": gate_name,
                "box": box,
                "title": "Known Person Detected" if name != "Unknown" else "Unknown Person Detected",
                "message": f"{name} detected at {gate_name}",
            }
            if event_type == "known_person":
                self._clear_unknown_near(box)
            self._update_track(
                self.active_person_events,
                (event_type, name) if event_type == "known_person" else event_type,
                box,
                event,
                alert_frame if alert_frame is not None else frame,
                detected_at,
            )

        for vehicle in vehicles:
            box = tuple(int(value) for value in vehicle["box"])
            vehicle_type = vehicle["vehicle_type"]
            title = (
                "Two Wheeler Detected"
                if vehicle_type == "two_wheeler"
                else "Four Wheeler Detected"
            )
            event = {
                "detected_at": detected_at,
                "detection_type": "vehicle",
                "vehicle_type": vehicle_type,
                "vehicle_class": vehicle["class_name"],
                "confidence": vehicle["confidence"],
                "gate_name": gate_name,
                "box": box,
                "title": title,
                "message": f"{title} at {gate_name}",
            }
            self._update_track(
                self.active_vehicle_events,
                self._vehicle_key(vehicle),
                box,
                event,
                alert_frame if alert_frame is not None else frame,
                detected_at,
            )
        return records
