"""Compatibility exports for Firebase/FCM integration."""

from alcon_stream_new import (
    initialize_firebase,
    _send_fcm_to_active_tokens,
    _send_fcm_to_all_tokens,
)

__all__ = [
    "initialize_firebase",
    "_send_fcm_to_active_tokens",
    "_send_fcm_to_all_tokens",
]
