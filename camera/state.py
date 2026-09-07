"""Small state object for consumers that need camera metadata."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraState:
    camera_id: str
    camera_name: str
    status: str = "Starting..."
    frame_counter: int = 0
    latest_frame: Optional[bytes] = None
    connected: bool = False
