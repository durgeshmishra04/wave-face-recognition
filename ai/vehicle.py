"""Compatibility exports for vehicle detection."""

from alcon_stream_new import detect_vehicle_boxes

detect_vehicles = detect_vehicle_boxes

__all__ = ["detect_vehicle_boxes", "detect_vehicles"]
