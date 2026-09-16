"""Independent ROI configuration for each camera.

Every camera starts with the existing production ROI values, but each entry
is copied independently so changing one camera does not change another.
"""

import os

import numpy as np

PERSON_ROI_POLYGON = np.array(
    [
        [0.10, 1.00],
        [0.12, 0.82],
        [0.18, 0.64],
        [0.28, 0.55],
        [0.42, 0.54],
        [0.62, 0.55],
        [0.76, 0.58],
        [0.88, 0.62],
        [0.94, 1.00],
    ],
    dtype=np.float32,
)

VEHICLE_ROI = {
    "left": float(os.getenv("VEHICLE_ROI_LEFT", "0.00")),
    "top": float(os.getenv("VEHICLE_ROI_TOP", "0.35")),
    "right": float(os.getenv("VEHICLE_ROI_RIGHT", "1.00")),
    "bottom": float(os.getenv("VEHICLE_ROI_BOTTOM", "0.70")),
}

VEHICLE_ROI_POINTS = np.array(
    [
        [VEHICLE_ROI["left"], VEHICLE_ROI["top"]],
        [VEHICLE_ROI["right"], VEHICLE_ROI["top"]],
        [VEHICLE_ROI["right"], VEHICLE_ROI["bottom"]],
        [VEHICLE_ROI["left"], VEHICLE_ROI["bottom"]],
    ],
    dtype=np.float32,
)


def _new_roi():
    return {
        "enabled": True,
        "points": PERSON_ROI_POLYGON.copy(),
        "vehicle": {
            **VEHICLE_ROI,
            "points": VEHICLE_ROI_POINTS.copy(),
        },
    }


# Keep one editable entry per configured camera. Every value is a separate
# object, so replace only the selected camera's points or vehicle rectangle.
CAMERA_ROIS = {
    "CAM001": {
    "enabled": True,

    "points": np.array([
        [0.286, 0.492],
        [0.365, 0.492],
        [0.911, 0.508],
        [0.950, 0.535],
        [0.963, 1.000],
        [0.054, 1.000],
    ], dtype=np.float32),

    "vehicle": {
        "points": np.array([
            [0.286, 0.492],
            [0.365, 0.492],
            [0.911, 0.508],
            [0.950, 0.535],
            [0.963, 1.000],
            [0.054, 1.000],
        ], dtype=np.float32),
    },
},
    "CAM002": _new_roi(),
    "CAM003": {
        "enabled": True,
        "points": np.array([
            [0.339, 0.529],
            [0.589, 0.548],
            [0.586, 0.595],
            [0.599, 0.983],
            [0.040, 0.983],
        ], dtype=np.float32),
        "vehicle": {
            **VEHICLE_ROI,
            "points": VEHICLE_ROI_POINTS.copy(),
        },
    },
    "CAM004": _new_roi(),
    "CAM005": _new_roi(),
    "CAM006": _new_roi(),
    "CAM007": _new_roi(),
    "CAM008": _new_roi(),
    "CAM009": _new_roi(),
    "CAM010": _new_roi(),
    "CAM011": _new_roi(),
    "CAM012": _new_roi(),
    "CAM013": {
    "enabled": True,

    "points": np.array([
        [0.000, 0.882],  # Bottom-left
        [0.095, 0.785],
        [0.195, 0.700],
        [0.265, 0.590],
        [0.339, 0.510],
        [0.465, 0.370],
        [0.583, 0.403],
        [0.831, 0.431],
        [0.747, 0.970],  # Bottom-right
    ], dtype=np.float32),

    
},
    "CAM014": {
    "enabled": True,

    "points": np.array([
        [0.000, 0.880],
        [0.250, 0.508],
        [0.426, 0.444],
        [0.478, 0.408],
        [0.560, 0.429],
        [0.615, 0.525],
        [0.689, 0.602],
        [0.783, 0.681],
        [0.923, 0.972],
    ], dtype=np.float32),

   
},
    "CAM015": {
    "enabled": True,

    "points": np.array([
        [0.000, 0.675],
        [0.145, 0.620],
        [0.295, 0.540],
        [0.405, 0.425],
        [0.470, 0.405],
        [0.535, 0.455],
        [0.635, 0.535],
        [0.710, 0.570],
        [0.790, 0.610],
        [0.755, 0.705],
        [0.705, 0.805],
        [0.635, 0.925],
        [0.590, 1.000],
    ], dtype=np.float32),

    
},
    "CAM016": {
    "enabled": True,

    "points": np.array([
        [0.265, 0.350],
        [0.355, 0.350],
        [0.535, 0.305],
        [0.665, 0.340],
        [0.830, 0.375],
        [1.000, 0.420],
        [1.000, 1.000],
        [0.265, 1.000],
    ], dtype=np.float32),

   
},
    "CAM017": _new_roi(),
    "CAM018": _new_roi(),
    "CAM019": _new_roi(),
    "CAM020": _new_roi(),
    "CAM021": _new_roi(),
    "CAM022": _new_roi(),
    "CAM023": _new_roi(),
    "CAM024": _new_roi(),
    "CAM025": _new_roi(),
    "CAM026": _new_roi(),
    "CAM027": _new_roi(),
    "CAM028": _new_roi(),
    "CAM029": _new_roi(),
    "CAM030": _new_roi(),
    "CAM031": _new_roi(),
    "CAM032": _new_roi(),
    "CAM033": _new_roi(),
    "CAM034": _new_roi(),
    "CAM035": _new_roi(),
    "CAM036": _new_roi(),
    "CAM037": _new_roi(),
    "CAM038": _new_roi(),
}


def get_camera_roi(camera_id):
    """Return an isolated copy of the selected camera's ROI configuration."""
    normalized_id = str(camera_id or "").strip().upper()
    configured = CAMERA_ROIS.get(normalized_id, _new_roi())
    configured_vehicle = configured.get("vehicle", {})
    vehicle_points = configured_vehicle.get("points", VEHICLE_ROI_POINTS)
    return {
        "enabled": bool(configured.get("enabled", True)),
        "points": np.asarray(configured["points"], dtype=np.float32).copy(),
        "vehicle": {
            **VEHICLE_ROI,
            **{
                key: value
                for key, value in configured_vehicle.items()
                if key != "points"
            },
            "points": np.asarray(vehicle_points, dtype=np.float32).copy(),
        },
    }


__all__ = [
    "CAMERA_ROIS",
    "PERSON_ROI_POLYGON",
    "VEHICLE_ROI",
    "VEHICLE_ROI_POINTS",
    "get_camera_roi",
]
