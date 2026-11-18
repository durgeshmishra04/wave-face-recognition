"""Incident workflow response mapping and service rules."""

from datetime import datetime, timezone
import math

from incident_workflow import repository
from incident_workflow.validators import alert_image_url


def utc_iso(timestamp):
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def list_incidents(page, limit, status, camera_id, start_epoch, end_epoch):
    total, rows = repository.list_incidents(
        page, limit, status, camera_id, start_epoch, end_epoch
    )
    data = []
    for row in rows:
        camera_name = row[4] or row[5] or row[3]
        data.append(
            {
                "id": row[0],
                "title": "Unknown Person Detected",
                "description": f"Unknown person detected at {camera_name or 'camera'}",
                "receivedAt": utc_iso(row[1]),
                "image": alert_image_url(row[6]),
                "category": "unknown_person",
                "cameraId": row[3],
                "cameraName": camera_name,
                "status": row[7],
            }
        )
    return {
        "data": data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "totalPages": math.ceil(total / limit),
        },
    }


def incident_details(incident_id):
    row, history_rows = repository.get_incident(incident_id)
    supervisor_status = "pending"
    if row[11] is not None:
        supervisor_status = "submitted"
    history = [
        {
            "action": item[0],
            "role": item[1],
            "username": item[2],
            "remark": item[3],
            "details": item[4],
            "createdAt": utc_iso(item[5]),
        }
        for item in history_rows
    ]
    return {
        "incidentId": row[0],
        "visitorName": row[7],
        "phone": row[8],
        "whomToMeet": row[9],
        "reason": row[10],
        "supervisorRemark": row[11],
        "supervisorName": row[13],
        "supervisorSubmittedAt": utc_iso(row[12]),
        "supervisorStatus": supervisor_status,
        "csoRemark": row[14],
        "csoName": row[16],
        "csoReviewedAt": utc_iso(row[15]),
        "csoStatus": row[17],
        "finalAction": row[18],
        "finalRemark": row[19],
        "finalActionAt": utc_iso(row[20]),
        "finalActionBy": row[21],
        "status": row[22],
        "detection": {
            "title": "Unknown Person Detected",
            "description": f"Unknown person detected at {row[4] or row[5] or row[3] or 'camera'}",
            "receivedAt": utc_iso(row[1]),
            "image": alert_image_url(row[6]),
            "category": "unknown_person",
            "cameraId": row[3],
            "cameraName": row[4] or row[5],
        },
        "history": history,
    }


def success(data, message="Operation completed successfully"):
    return {"success": True, "message": message, "data": data}


def error(message):
    return {"success": False, "message": message}
