"""Paths for configuration and persistent data outside a frozen executable."""

import errno
import os
import sys
from pathlib import Path


def runtime_root() -> Path:
    configured_root = os.getenv("WAVE_RUNTIME_DIR", "").strip()
    if configured_root:
        path = Path(configured_root).expanduser()
        if not path.is_absolute():
            raise ValueError("WAVE_RUNTIME_DIR must be an absolute path.")
        return path.resolve()
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        if executable_dir.parent.name == "dist":
            project_root = executable_dir.parent.parent
            if (project_root / "config" / "cameras.json").is_file():
                return project_root
        return executable_dir
    return Path(__file__).resolve().parent


def acquire_runtime_lock():
    """Keep source and frozen processes from writing to one runtime concurrently."""
    lock_path = runtime_root() / ".wave-vms.lock"
    try:
        lock_file = lock_path.open("a+b")
    except OSError as error:
        raise RuntimeError(f"Cannot open Wave VMS runtime lock {lock_path}: {error}") from error

    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise RuntimeError(
            f"Another Wave VMS process already holds the runtime lock: {lock_path}"
        ) from error
    except OSError as error:
        lock_file.close()
        if os.name == "nt" and error.errno in {
            errno.EACCES,
            errno.EAGAIN,
            errno.EDEADLK,
        }:
            raise RuntimeError(
                f"Another Wave VMS process already holds the runtime lock: {lock_path}"
            ) from error
        raise RuntimeError(f"Cannot lock Wave VMS runtime directory {lock_path}: {error}") from error
    return lock_file


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
