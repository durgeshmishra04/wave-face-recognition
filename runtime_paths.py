"""Paths for configuration and persistent data."""

import os
from pathlib import Path


def runtime_root() -> Path:
    configured_root = os.getenv("WAVE_RUNTIME_DIR", "").strip()
    if configured_root:
        path = Path(configured_root).expanduser()
        if not path.is_absolute():
            raise ValueError("WAVE_RUNTIME_DIR must be an absolute path.")
        return path.resolve()
    return Path(__file__).resolve().parent


def resolve_runtime_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return runtime_root() / path


def resolve_model_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    project_path = runtime_root() / path
    if project_path.exists():
        return project_path

    model_path = runtime_root() / "models" / path
    if model_path.exists():
        return model_path
    return project_path
