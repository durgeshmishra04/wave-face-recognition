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
                 firebase_topic, dedup_seconds=10.0):
        self.database = database
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")
        self.socketio = socketio
        self.firebase_topic = firebase_topic
        self.dedup_seconds = dedup_seconds
        self.last_seen = {}
        self.active_person_events = {}
        self.active_vehicle_events = {}
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

    def _is_new_event(self, key, now):
        previous = self.last_seen.get(key, 0.0)
        self.last_seen[key] = now
        return now - previous >= self.dedup_seconds

    def _is_new_vehicle(self, vehicle, now):
        vehicle_key = (
            vehicle["vehicle_type"],
            vehicle["class_name"],
        )
        box = vehicle["box"]
        center = (
            (int(box[0]) + int(box[2])) / 2,
            (int(box[1]) + int(box[3])) / 2,
        )

        for active_key, active in list(self.active_vehicle_events.items()):
            if now - active["last_seen"] >= self.dedup_seconds:
                del self.active_vehicle_events[active_key]
                continue
            if active["vehicle_key"] != vehicle_key:
                continue
            distance = (
                (center[0] - active["center"][0]) ** 2
                + (center[1] - active["center"][1]) ** 2
            ) ** 0.5
            if distance <= max(
                100,
                int(max(
                    int(box[2]) - int(box[0]),
                    int(box[3]) - int(box[1]),
                )),
            ):
                active["center"] = center
                active["last_seen"] = now
                return False

        event_key = (vehicle_key, id(box))
        self.active_vehicle_events[event_key] = {
            "vehicle_key": vehicle_key,
            "center": center,
            "last_seen": now,
        }
        return True

    def _is_new_person(self, event_type, name, box, now):
        center = (
            (int(box[0]) + int(box[2])) / 2,
            (int(box[1]) + int(box[3])) / 2,
        )

        if event_type == "unknown_person":
            for active_key, active in list(self.active_person_events.items()):
                if now - active["last_seen"] >= self.dedup_seconds:
                    del self.active_person_events[active_key]
                    continue
                distance = (
                    (center[0] - active["center"][0]) ** 2
                    + (center[1] - active["center"][1]) ** 2
                ) ** 0.5
                if active["person_key"] == event_type and distance <= max(
                    100,
                    int(max(
                        int(box[2]) - int(box[0]),
                        int(box[3]) - int(box[1]),
                    )),
                ):
                    active["center"] = center
                    active["last_seen"] = now
                    return False
            person_key = event_type
        else:
            person_key = (event_type, name)
            active = self.active_person_events.get(person_key)
            if active is not None and now - active["last_seen"] < self.dedup_seconds:
                active["center"] = center
                active["last_seen"] = now
                return False

        event_key = (person_key, id(box)) if event_type == "unknown_person" else person_key
        self.active_person_events[event_key] = {
            "person_key": person_key,
            "center": center,
            "last_seen": now,
        }
        return True

    def _clear_unknown_near(self, box, now):
        center = (
            (int(box[0]) + int(box[2])) / 2,
            (int(box[1]) + int(box[3])) / 2,
        )
        for active_key, active in list(self.active_person_events.items()):
            if active["person_key"] != "unknown_person":
                continue
            distance = (
                (center[0] - active["center"][0]) ** 2
                + (center[1] - active["center"][1]) ** 2
            ) ** 0.5
            if distance <= max(
                100,
                int(max(
                    int(box[2]) - int(box[0]),
                    int(box[3]) - int(box[1]),
                )),
            ):
                del self.active_person_events[active_key]

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
        image_frame = (
            alert_frame.copy()
            if notify and alert_frame is not None
            else frame.copy()
        )
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
        records = []
        for face in faces:
            name = getattr(face, "recognized_name", "Unknown")
            confidence = float(getattr(face, "recognition_score", 0.0))
            box = tuple(int(value) for value in face.bbox)
            event_type = "known_person" if name != "Unknown" else "unknown_person"
            if event_type == "known_person":
                self._clear_unknown_near(box, detected_at)
            if not self._is_new_person(event_type, name, box, detected_at):
                continue
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
            records.append(
                self._record(
                    frame,
                    event,
                    name == "Unknown",
                    alert_frame,
                )
            )

        for vehicle in vehicles:
            box = tuple(int(value) for value in vehicle["box"])
            if not self._is_new_vehicle(vehicle, detected_at):
                continue
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
            records.append(self._record(frame, event, True, alert_frame))
        return records
