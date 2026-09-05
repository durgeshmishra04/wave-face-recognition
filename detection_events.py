"""ROI entry/exit event tracking. Records only confirmed exits, never frames."""
import json
import os
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from firebase_admin import messaging


class DetectionEventManager:
    def __init__(self, database, image_dir, public_base_url, socketio,
                 firebase_topic=None, fcm_sender=None,
                 dedup_seconds=10.0, exit_frame_offset=5):
        self.database = database
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.public_base_url = public_base_url.rstrip("/")
        self.socketio = socketio
        self.firebase_topic = firebase_topic
        self.fcm_sender = fcm_sender
        self.exit_confirm_seconds = max(0.1, float(dedup_seconds))
        self.exit_frame_offset = max(0, int(exit_frame_offset))
        self.active_person_events = []
        self.active_vehicle_events = []
        # Unknown people in one concurrent ROI visit are deliberately held
        # until the final member exits, then emitted as a single group event.
        self.pending_unknown_exits = []
        try:
            self.person_ids = json.loads(os.getenv("PERSON_IDS_JSON", "{}"))
        except json.JSONDecodeError:
            self.person_ids = {}

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _iou(a, b):
        left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _buffer_frame(frame):
        """
        Compress frame before storing it in the tracking buffer.
        """
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if not success:
            raise RuntimeError("Could not encode tracking frame")

        return encoded.tobytes()

    @staticmethod
    def _decode_buffered_frame(frame_data):
        if isinstance(frame_data, bytes):
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                raise RuntimeError("Could not decode buffered frame")

            return frame

        # Backward compatibility
        return frame_data.copy()

    def _match_track(self, tracks, category, box):
        center, best, best_score = self._center(box), None, -1.0
        for track in tracks:
            if track["category"] != category or track["matched_this_frame"]:
                continue
            iou = self._iou(track["last_box"], box)
            distance = ((center[0] - track["center"][0]) ** 2 + (center[1] - track["center"][1]) ** 2) ** 0.5
            scale = max(1, box[2] - box[0], box[3] - box[1], track["last_box"][2] - track["last_box"][0], track["last_box"][3] - track["last_box"][1])
            if iou < .10 and distance > max(100, scale * 1.5):
                continue
            score = iou - distance / (scale * 10)
            if score > best_score:
                best, best_score = track, score
        return best

    def _update_track(self, tracks, category, event, frame, now):
        track = self._match_track(tracks, category, event["box"])
        if track is None:
            track = {"event_id": str(uuid.uuid4()), "category": category, "state": "ENTERED_ROI", "first_seen": now, "last_seen": now, "last_box": event["box"], "center": self._center(event["box"]), "matched_this_frame": True, "known_detected_once": event["detection_type"] == "known_person", "vehicle_context_detected": bool(event.get("vehicle_context_detected")), "suppress_notify": bool(event.get("suppress_notify")), "event": event.copy(), "frames": deque(maxlen=self.exit_frame_offset + 2)}
            tracks.append(track)
        else:
            track.update({"state": "TRACKING", "last_seen": now, "last_box": event["box"], "center": self._center(event["box"]), "matched_this_frame": True})
        # A recognition is sticky for the entire physical ROI crossing.
        if event["detection_type"] == "known_person":
            track["known_detected_once"] = True
            track["event"] = event.copy()
        elif not track["known_detected_once"]:
            track["vehicle_context_detected"] = track.get("vehicle_context_detected", False) or bool(event.get("vehicle_context_detected"))
            track["suppress_notify"] = track.get("suppress_notify", False) or bool(event.get("suppress_notify"))
            track["event"] = event.copy()
        track["event"]["box"] = event["box"]
        if track.get("vehicle_context_detected"):
            track["event"]["vehicle_context_detected"] = True
        if track.get("suppress_notify"):
            track["event"]["suppress_notify"] = True
        try:
            buffered_frame = self._buffer_frame(frame)
        except Exception as error:
            print(f"[ERROR] Could not buffer tracking frame: {error}")
            return
        track["frames"].append({"frame": buffered_frame, "event": track["event"].copy()})

    def _save_image(self, frame):
        filename = f"{uuid.uuid4()}.jpg"
        if not cv2.imwrite(str(self.image_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise RuntimeError("Could not save detection image")
        return f"{self.public_base_url}/alerts/{filename}"

    @staticmethod
    def _draw_event(frame, event):
        if event["detection_type"] == "vehicle":
            x1, y1, x2, y2 = event["box"]
            label, color = f"{event['vehicle_type'].replace('_', ' ').title()} | {event['vehicle_class']} | {event['confidence']:.2f}", (0, 140, 255)
        elif event["detection_type"] == "known_person":
            # The person-model box is tracking-only. Only a face box may be
            # rendered on the image exposed to API/Firebase/Android.
            if not event.get("annotation_box"):
                return
            x1, y1, x2, y2 = event["annotation_box"]
            label, color = f"KNOWN | {event['person_name']} | ID: {event.get('person_id') or 'N/A'} | {event['confidence']:.2f}", (0, 255, 0)
        else:
            if not event.get("annotation_box"):
                return
            x1, y1, x2, y2 = event["annotation_box"]
            count = event.get("unknown_count", 1)
            label, color = ("UNKNOWN" if count == 1 else f"UNKNOWN PERSONS: {count}"), (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # A solid color label with thick white text remains readable in the
        # image downloaded by Android and in the Firebase notification.
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            .65,
            3,
        )
        label_bottom = max(text_height + baseline + 10, y1)
        label_top = max(0, label_bottom - text_height - baseline - 10)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 12, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 6, label_bottom - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            .65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    def _notify(self, event):
        image_url = event["image_url"]
        data = {key: str(value) for key, value in {"type": event["detection_type"], "detection_id": event["id"], "gate": event["gate_name"], "image_url": image_url, "confidence": event.get("confidence", ""), "unknown_count": event.get("unknown_count", ""), "detected_at": event["detected_at"]}.items()}
        if self.fcm_sender is not None:
            self.fcm_sender(
                title=event["title"],
                body=event["message"],
                data=data,
                image_url=image_url,
            )
            return
        print("[WARNING] No FCM sender configured; notification skipped.")

    def _publish(self, event, image, notify):
        event["image_url"] = self._save_image(image)
        event["id"] = self.database.save(event)
        event["type"], event["gate"] = event["detection_type"], event["gate_name"]
        event["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event["detected_at"]))
        self.socketio.emit("detection_event", event)
        if event["detection_type"] == "unknown_person":
            self.socketio.emit("face_alert", event)
        if notify:
            try:
                self._notify(event)
            except Exception as error:
                print(f"[ERROR] Detection notification failed: {error}")
        return event

    def _finalize_expired(self, tracks, now):
        expired, remaining = [], []
        for track in tracks:
            if now - track["last_seen"] < self.exit_confirm_seconds:
                remaining.append(track)
            elif track["frames"]:
                track["state"] = "FINALIZED"
                # With frames 101..108 and a confirmed exit at 108, index -7
                # is frame 102: exactly EXIT_FRAME - 6.
                index = (
                    max(-len(track["frames"]), -(self.exit_frame_offset + 1))
                    if self.exit_frame_offset
                    else -1
                )
                expired.append((track, track["frames"][index]))
        tracks[:] = remaining
        return expired

    def _person_event(self, face, gate_name, now):
        name, confidence = getattr(face, "recognized_name", "Unknown"), float(getattr(face, "recognition_score", 0.0))
        known = name != "Unknown"
        return {"detected_at": now, "detection_type": "known_person" if known else "unknown_person", "person_id": getattr(face, "person_id", None) or self.person_ids.get(name), "person_name": name, "confidence": confidence, "gate_name": gate_name, "box": tuple(int(v) for v in face.bbox), "annotation_box": getattr(face, "annotation_box", None), "title": "Known Person Detected" if known else "Unknown Person Detected", "message": f"{name} detected at {gate_name}"}

    def _publish_unknowns(self, unknowns, gate_name, now):
        track, chosen = unknowns[0]
        event = track["event"].copy()
        count = len(unknowns)
        vehicle_context = any(
            item[0].get("vehicle_context_detected")
            for item in unknowns
        )
        event.update({"detected_at": now, "detection_type": "unknown_person", "person_name": "Unknown", "person_id": None, "unknown_count": count, "title": "Unknown Person Detected" if count == 1 else "Unknown Persons Detected", "message": f"Unknown person detected at {gate_name}" if count == 1 else f"{count} unknown persons detected at {gate_name}"})
        if vehicle_context:
            event["title"] = "Unknown Person Detected at Vehicle" if count == 1 else "Unknown Persons Detected at Vehicle"
            event["message"] = f"Unknown person detected at vehicle at {gate_name}" if count == 1 else f"{count} unknown persons detected at vehicle at {gate_name}"
            event["vehicle_context_detected"] = True
        try:
            image = self._decode_buffered_frame(chosen["frame"])
        except Exception as error:
            print(f"[ERROR] Could not decode tracking frame: {error}")
            return None
        for item, _ in unknowns:
            marked = item["event"].copy()
            marked["unknown_count"] = count
            self._draw_event(image, marked)
        return self._publish(event, image, notify=True)

    def process_frame(self, frame, faces, vehicles, gate_name, detected_at, alert_frame=None):
        """Ingest one annotated ROI frame; return records finalized on this call."""
        annotated = alert_frame if alert_frame is not None else frame
        for track in self.active_person_events + self.active_vehicle_events:
            track["matched_this_frame"] = False
        for face in faces:
            self._update_track(self.active_person_events, "person", self._person_event(face, gate_name, detected_at), annotated, detected_at)
        for vehicle in vehicles:
            vehicle_type = vehicle["vehicle_type"]
            title = "Two Wheeler Detected" if vehicle_type == "two_wheeler" else "Four Wheeler Detected"
            event = {"detected_at": detected_at, "detection_type": "vehicle", "vehicle_type": vehicle_type, "vehicle_class": vehicle["class_name"], "confidence": float(vehicle["confidence"]), "gate_name": gate_name, "box": tuple(int(v) for v in vehicle["box"]), "title": title, "message": f"{title} at {gate_name}"}
            self._update_track(self.active_vehicle_events, f"vehicle:{vehicle_type}", event, annotated, detected_at)
        active_unknown_tracks = [
            track
            for track in self.active_person_events
            if not track["known_detected_once"]
        ]
        if active_unknown_tracks and self.active_vehicle_events:
            for track in active_unknown_tracks:
                track["vehicle_context_detected"] = True
                track["event"]["vehicle_context_detected"] = True
            for track in self.active_vehicle_events:
                track["suppress_notify"] = True
                track["event"]["suppress_notify"] = True
        # Keep the buffer aligned with real processing frames while a track is
        # temporarily missing. This makes the selection relative to the
        # confirmation frame (rather than merely the fifth prior detection).
        for track in self.active_person_events + self.active_vehicle_events:
            if not track["matched_this_frame"]:
                try:
                    buffered_frame = self._buffer_frame(annotated)
                except Exception as error:
                    print(f"[ERROR] Could not buffer tracking frame: {error}")
                    continue
                track["frames"].append({
                    "frame": buffered_frame,
                    "event": track["event"].copy(),
                })
        people = self._finalize_expired(self.active_person_events, detected_at)
        vehicles = self._finalize_expired(self.active_vehicle_events, detected_at)
        expired_unknowns = [
            (track, chosen)
            for track, chosen in people
            if not track["known_detected_once"]
        ]
        self.pending_unknown_exits.extend(expired_unknowns)
        unknowns_still_inside = any(
            not track["known_detected_once"]
            for track in self.active_person_events
        )
        # Do not publish when the first person leaves. Wait until every
        # unknown track from the same ROI group has confirmed its exit.
        records = []
        if self.pending_unknown_exits and not unknowns_still_inside:
            records.append(self._publish_unknowns(
                self.pending_unknown_exits,
                gate_name,
                detected_at,
            ))
            self.pending_unknown_exits.clear()
        for track, chosen in people:
            if track["known_detected_once"]:
                event = track["event"].copy(); event["detected_at"] = detected_at
                try:
                    image = self._decode_buffered_frame(chosen["frame"])
                except Exception as error:
                    print(f"[ERROR] Could not decode tracking frame: {error}")
                    continue
                self._draw_event(image, event)
                records.append(self._publish(event, image, notify=False))
        for track, chosen in vehicles:
            event = track["event"].copy(); event["detected_at"] = detected_at
            try:
                image = self._decode_buffered_frame(chosen["frame"])
            except Exception as error:
                print(f"[ERROR] Could not decode tracking frame: {error}")
                continue
            self._draw_event(image, event)
            records.append(self._publish(event, image, notify=not track.get("suppress_notify", False)))
        return records
