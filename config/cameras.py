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
        cameras.append({
            "camera_id": camera_id,
            "name": str(item["name"]).strip(),
            "nvr_ip": os.getenv(
                f"{camera_id}_NVR_IP",
                item.get("nvr_ip", DEFAULT_NVR_IP),
            ),
            "channel": int(os.getenv(
                f"{camera_id}_CHANNEL", str(item.get("channel", 1))
            )),
            "subtype": int(os.getenv(
                f"{camera_id}_SUBTYPE", str(item.get("subtype", 1))
            )),
            "kpi": str(item.get("kpi", "person")).strip().lower(),
            "enabled": os.getenv(
                f"{camera_id}_ENABLED",
                "true" if item.get("enabled", True) else "false",
            ).strip().lower() not in {"false", "0", "no", "off"},
        })

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
