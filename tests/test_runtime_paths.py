from pathlib import Path

import json

import pytest

from config import cameras as camera_config
from runtime_paths import (
    acquire_runtime_lock,
    resolve_model_path,
    resolve_runtime_path,
    runtime_root,
)


def test_runtime_root_uses_configured_external_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("WAVE_RUNTIME_DIR", str(tmp_path))

    assert runtime_root() == tmp_path.resolve()


def test_frozen_development_build_uses_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("WAVE_RUNTIME_DIR", raising=False)
    monkeypatch.setattr("runtime_paths.sys.frozen", True, raising=False)
    executable = tmp_path / "dist" / "wave-vms" / "wave-vms"
    executable.parent.mkdir(parents=True)
    config_directory = tmp_path / "config"
    config_directory.mkdir()
    (config_directory / "cameras.json").touch()
    monkeypatch.setattr("runtime_paths.sys.executable", str(executable))

    assert runtime_root() == tmp_path


def test_frozen_release_defaults_to_executable_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("WAVE_RUNTIME_DIR", raising=False)
    monkeypatch.setattr("runtime_paths.sys.frozen", True, raising=False)
    executable = tmp_path / "wave-vms" / "wave-vms"
    executable.parent.mkdir()
    monkeypatch.setattr("runtime_paths.sys.executable", str(executable))

    assert runtime_root() == executable.parent


def test_runtime_root_rejects_relative_configuration(monkeypatch):
    monkeypatch.setenv("WAVE_RUNTIME_DIR", "relative/path")

    with pytest.raises(ValueError, match="absolute path"):
        runtime_root()


def test_relative_runtime_paths_use_external_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("WAVE_RUNTIME_DIR", str(tmp_path))

    assert resolve_runtime_path("config/cameras.json") == tmp_path / "config/cameras.json"


def test_model_paths_support_project_and_models_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("WAVE_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "external.pt").touch()

    assert resolve_model_path("external.pt") == tmp_path / "models" / "external.pt"
    assert resolve_model_path("missing.pt") == tmp_path / "missing.pt"


def test_absolute_model_path_is_preserved(tmp_path):
    absolute_path = tmp_path / "custom.pt"

    assert resolve_model_path(absolute_path) == absolute_path


def test_runtime_lock_prevents_two_process_modes_sharing_data(monkeypatch, tmp_path):
    monkeypatch.setenv("WAVE_RUNTIME_DIR", str(tmp_path))
    first_lock = acquire_runtime_lock()
    try:
        with pytest.raises(RuntimeError, match="already holds"):
            acquire_runtime_lock()
    finally:
        first_lock.close()

    next_lock = acquire_runtime_lock()
    next_lock.close()


def test_camera_config_uses_global_nvr_for_sanitized_empty_value(
    monkeypatch, tmp_path
):
    camera_file = tmp_path / "cameras.json"
    camera_file.write_text(
        json.dumps(
            [
                {
                    "camera_id": "CAM001",
                    "name": "Camera 001",
                    "ip": "",
                    "nvr_ip": "",
                    "enabled": False,
                    "channel": 1,
                    "subtype": 1,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("CAM001_NVR_IP", raising=False)
    monkeypatch.setattr(camera_config, "DEFAULT_NVR_IP", "192.0.2.5")

    cameras = camera_config.load_cameras(camera_file)

    assert cameras[0]["nvr_ip"] == "192.0.2.5"
