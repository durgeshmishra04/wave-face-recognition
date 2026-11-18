"""Input validation for incident workflow requests."""

from datetime import date, datetime, time, timedelta, timezone
import re
from urllib.parse import unquote, urlsplit


class ValidationError(ValueError):
    pass


def required_text(payload, name, maximum=2000):
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} is required")
    value = value.strip()
    if len(value) > maximum:
        raise ValidationError(f"{name} must be {maximum} characters or fewer")
    return value


def optional_text(payload, name, maximum=2000):
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    value = value.strip()
    if len(value) > maximum:
        raise ValidationError(f"{name} must be {maximum} characters or fewer")
    return value or None


def incident_id_value(value):
    if not re.fullmatch(r"[1-9][0-9]*", str(value or "")):
        raise ValidationError("incident_id must be a positive integer")
    return int(value)


def page_value(value, name, default, maximum=None):
    try:
        result = default if value is None else int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be an integer") from error
    if result < 1:
        raise ValidationError(f"{name} must be at least 1")
    if maximum is not None and result > maximum:
        raise ValidationError(f"{name} must be at most {maximum}")
    return result


def utc_date_boundary(value, end=False):
    if not value:
        return None
    try:
        if len(value) == 10:
            parsed_date = date.fromisoformat(value)
            boundary = datetime.combine(
                parsed_date + (timedelta(days=1) if end else timedelta()),
                time.min,
                tzinfo=timezone.utc,
            )
        else:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            boundary = parsed.astimezone(timezone.utc)
        return boundary.timestamp()
    except (TypeError, ValueError) as error:
        raise ValidationError(f"Invalid {'end' if end else 'start'}_date") from error


def alert_image_url(value):
    if not value:
        return None
    path = unquote(urlsplit(value).path)
    segments = path.split("/")
    if (
        len(segments) != 3
        or segments[1] != "alerts"
        or not re.fullmatch(r"[A-Za-z0-9._-]+", segments[2])
        or segments[2] in {".", ".."}
    ):
        return None
    return f"/alerts/{segments[2]}"
