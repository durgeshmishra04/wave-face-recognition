import sqlite3
import time
from pathlib import Path


class DetectionDatabase:

    def __init__(self, database_path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def connect(self):
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        return connection

    def ensure_schema(self):
        connection = self.connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at REAL NOT NULL,
                    detection_type TEXT NOT NULL,
                    person_id TEXT,
                    person_name TEXT,
                    vehicle_type TEXT,
                    vehicle_class TEXT,
                    confidence REAL,
                    gate_name TEXT NOT NULL,
                    image_url TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_detection_records_detected_at
                ON detection_records (detected_at DESC)
                """
            )
            connection.commit()
        finally:
            connection.close()

    def save(self, event):
        connection = self.connect()
        try:
            cursor = connection.execute(
                """
                INSERT INTO detection_records (
                    detected_at,
                    detection_type,
                    person_id,
                    person_name,
                    vehicle_type,
                    vehicle_class,
                    confidence,
                    gate_name,
                    image_url,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.get("detected_at", time.time()),
                    event["detection_type"],
                    event.get("person_id"),
                    event.get("person_name"),
                    event.get("vehicle_type"),
                    event.get("vehicle_class"),
                    event.get("confidence"),
                    event.get("gate_name", "Main Gate 01"),
                    event.get("image_url", ""),
                    time.time(),
                ),
            )
            connection.commit()
            return cursor.lastrowid
        finally:
            connection.close()

    def latest(self, limit=50):
        limit = max(1, min(int(limit), 500))
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT id, detected_at, detection_type, person_id,
                       person_name, vehicle_type, vehicle_class,
                       confidence, gate_name, image_url, created_at
                FROM detection_records
                ORDER BY detected_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            connection.close()

        detections = []
        for row in rows:
            item = dict(row)
            item["type"] = item.pop("detection_type")
            item["time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(item.pop("detected_at")),
            )
            item["gate"] = item.pop("gate_name")
            item.pop("created_at", None)
            detections.append(item)
        return detections
