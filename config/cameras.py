"""Camera configuration loading and lookup helpers."""

import json
import os
from pathlib import Path

from config.settings import PROJECT_ROOT

DEFAULT_NVR_IP = os.getenv("NVR_IP", "115.247.225.82")
CAMERA_CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.json"


def load_cameras(config_path=CAMERA_CONFIG_PATH):
    """Load cameras while preserving the existing environment overrides."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        configured = json.load(config_file)

    cameras = []
    for item in configured:
        camera_id = str(item["camera_id"]).strip()
        if not camera_id or any(
            camera_id == existing["camera_id"] for existing in cameras
        ):
            raise ValueError(f"Duplicate or empty camera_id: {camera_id!r}")
        nvr_ip = os.getenv(
            f"{camera_id}_NVR_IP",
            item.get("nvr_ip", DEFAULT_NVR_IP),
        )
        channel = int(os.getenv(
            f"{camera_id}_CHANNEL", str(item.get("channel", 1))
        ))
        configured_streams = item.get("streams") or {}
        streams = {}
        for stream_type, stream_number in (("main", 1), ("sub", 2)):
            configured = configured_streams.get(stream_type) or {}
            streams[stream_type] = {
                "enabled": bool(configured.get("enabled", True)),
                "stream_type": stream_type,
                "stream_number": int(configured.get("stream_number", stream_number)),
                "rtsp_url": str(
                    configured.get("rtsp_url", "")
                ).strip(),
            }
        cameras.append({
            "camera_id": camera_id,
            "name": str(item["name"]).strip(),
            "ip": item.get("ip"),
            "nvr_ip": nvr_ip,
            "channel": channel,
            "subtype": int(os.getenv(
                f"{camera_id}_SUBTYPE", str(item.get("subtype", 1))
            )),
            "kpi": str(item.get("kpi", "person")).strip().lower(),
            "enabled": os.getenv(
                f"{camera_id}_ENABLED",
                "true" if item.get("enabled", True) else "false",
            ).strip().lower() not in {"false", "0", "no", "off"},
            "streams": streams,
            "rtsp_url": streams["main"]["rtsp_url"],
        })

        for stream_type in ("main", "sub"):
            stream = streams[stream_type]
            status = "configured" if stream["enabled"] and stream["rtsp_url"] else "missing"
            print(f"[CAMERA-CONFIG] {camera_id} {stream_type.upper()} {status}")
        print(
            f"[CAMERA-CONFIG] {camera_id} | "
            f"MAIN={'OK' if streams['main']['rtsp_url'] else 'MISSING'} | "
            f"SUB={'OK' if streams['sub']['rtsp_url'] else 'MISSING'} | "
            f"ENABLED={'YES' if cameras[-1]['enabled'] else 'NO'}"
        )

    enabled_override = os.getenv("ENABLED_CAMERAS", "").strip()
    if enabled_override:
        selected = {
            value.strip().upper()
            for value in enabled_override.split(",")
            if value.strip()
        }
        if selected != {"ALL"}:
            known_ids = {camera["camera_id"] for camera in cameras}
            unknown = selected - known_ids
            if unknown:
                raise ValueError(
                    "Unknown camera IDs in ENABLED_CAMERAS: "
                    + ", ".join(sorted(unknown))
                )
            for camera in cameras:
                camera["enabled"] = camera["camera_id"] in selected

    return cameras


CAMERAS = load_cameras()


def enabled_cameras():
    return [camera for camera in CAMERAS if camera.get("enabled", True)]


def camera_by_id(camera_id):
    if camera_id:
        normalized = str(camera_id).strip().lower()
        return next(
            (camera for camera in CAMERAS
             if camera["camera_id"].lower() == normalized),
            None,
        )
    return None


__all__ = ["CAMERAS", "CAMERA_CONFIG_PATH", "camera_by_id", "enabled_cameras", "load_cameras"]
