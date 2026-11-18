import importlib

from fall_detection import fall_config
from ppe_detection import ppe_config


def test_ppe_camera_scope_is_configurable(monkeypatch):
    with monkeypatch.context() as scoped_env:
        scoped_env.delenv("PPE_CAMERAS", raising=False)
        importlib.reload(ppe_config)
        assert ppe_config.PPE_CAMERAS == {"CAM008"}

        scoped_env.setenv("PPE_CAMERAS", "cam003, CAM013")
        importlib.reload(ppe_config)

        assert ppe_config.PPE_CAMERAS == {"CAM003", "CAM013"}
        assert ppe_config.is_ppe_camera_enabled("cam013")
        assert not ppe_config.is_ppe_camera_enabled("CAM008")

    importlib.reload(ppe_config)


def test_fall_camera_scope_is_configurable(monkeypatch):
    with monkeypatch.context() as scoped_env:
        scoped_env.delenv("FALL_CAMERAS", raising=False)
        importlib.reload(fall_config)
        assert fall_config.FALL_CAMERAS == {"CAM008"}

        scoped_env.setenv("FALL_CAMERAS", "cam003, CAM013")
        importlib.reload(fall_config)

        assert fall_config.FALL_CAMERAS == {"CAM003", "CAM013"}
        assert fall_config.is_fall_camera_enabled("cam013")
        assert not fall_config.is_fall_camera_enabled("CAM008")

    importlib.reload(fall_config)
