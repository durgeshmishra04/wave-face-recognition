import time

from database import get_db_connection


class DetectionDatabase:

    def __init__(self, database_path=None):
        self.database_path = database_path
        self.ensure_schema()

    def connect(self):
        return get_db_connection()

    def ensure_schema(self):
        connection = self.connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_records (
                    id SERIAL PRIMARY KEY,
                    detected_at DOUBLE PRECISION NOT NULL,
                    detection_type TEXT NOT NULL,
                    person_id TEXT,
                    person_name TEXT,
                    vehicle_type TEXT,
                    vehicle_class TEXT,
                    confidence DOUBLE PRECISION,
                    unknown_count INTEGER,
                    camera_id TEXT,
                    camera_name TEXT,
                    gate_name TEXT NOT NULL,
                    image_url TEXT,
                    created_at DOUBLE PRECISION NOT NULL
                )
                """
            )
            columns = {
                row[0]
                for row in connection.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'detection_records'
                    """
                ).fetchall()
            }
            if "unknown_count" not in columns:
                connection.execute(
                    "ALTER TABLE detection_records ADD COLUMN unknown_count INTEGER"
                )
            if "camera_id" not in columns:
                connection.execute(
                    "ALTER TABLE detection_records ADD COLUMN camera_id TEXT"
                )
            if "camera_name" not in columns:
                connection.execute(
                    "ALTER TABLE detection_records ADD COLUMN camera_name TEXT"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_detection_records_detected_at
                ON detection_records (detected_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_detection_records_camera_id
                ON detection_records (camera_id, detected_at DESC)
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
                    unknown_count,
                    camera_id,
                    camera_name,
                    gate_name,
                    image_url,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    event.get("detected_at", time.time()),
                    event["detection_type"],
                    event.get("person_id"),
                    event.get("person_name"),
                    event.get("vehicle_type"),
                    event.get("vehicle_class"),
                    event.get("confidence"),
                    event.get("unknown_count"),
                    event.get("camera_id"),
                    event.get("camera_name"),
                    event.get("gate_name", "Main Gate 01"),
                    event.get("image_url", ""),
                    time.time(),
                ),
            )
            connection.commit()
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            connection.close()

    def latest(self, limit=0, camera_id=None):
        """Return persisted finalized events; limit=0 intentionally means all."""
        limit = max(0, int(limit or 0))
        connection = self.connect()
        try:
            query = """
                SELECT id, detected_at, detection_type, person_id,
                       person_name, vehicle_type, vehicle_class,
                       confidence, unknown_count, camera_id, camera_name,
                       gate_name, image_url, created_at
                FROM detection_records
            """
            params = []
            if camera_id:
                query += " WHERE camera_id = %s OR camera_id IS NULL"
                params.append(camera_id)
            query += " ORDER BY detected_at DESC, id DESC"
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            cursor = connection.execute(query, tuple(params))
            rows = cursor.fetchall()
        finally:
            connection.close()

        detections = []
        names = [
            "id",
            "detected_at",
            "detection_type",
            "person_id",
            "person_name",
            "vehicle_type",
            "vehicle_class",
            "confidence",
            "unknown_count",
            "camera_id",
            "camera_name",
            "gate_name",
            "image_url",
            "created_at",
        ]
        for row in rows:
            item = dict(zip(names, row))
            item["type"] = item.pop("detection_type")
            item["time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(item.pop("detected_at")),
            )
            item["gate"] = item.pop("gate_name")
            item["camera_id"] = item.get("camera_id") or "CAM001"
            item["camera_name"] = item.get("camera_name") or item["gate"]
            item.pop("created_at", None)
            detections.append(item)
        return detections
