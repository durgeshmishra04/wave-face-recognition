"""Linux onedir build; application source remains in the original checkout."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

SOURCE_ROOT = Path(os.environ["WAVE_SOURCE_ROOT"]).resolve()
if not (SOURCE_ROOT / "app.py").is_file():
    raise SystemExit(f"WAVE_SOURCE_ROOT does not contain app.py: {SOURCE_ROOT}")

hiddenimports = [
    "engineio.async_drivers.threading",
    "firebase_admin",
    "flask_socketio",
    "insightface.model_zoo.arcface_onnx",
    "insightface.model_zoo.retinaface",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "psycopg_binary",
    "socketio",
]
for package in (
    "insightface.model_zoo",
    "ultralytics.models",
    "ultralytics.nn",
    "ultralytics.utils.callbacks",
):
    hiddenimports.extend(collect_submodules(package))

datas = collect_data_files(
    "ultralytics",
    includes=["cfg/*.yaml", "cfg/**/*.yaml"],
)
datas += collect_data_files("firebase_admin")
binaries = []
for package in ("cv2", "onnxruntime", "torch", "torchvision"):
    binaries += collect_dynamic_libs(package)

a = Analysis(
    [str(SOURCE_ROOT / "app.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "notebook", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="wave-vms",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="wave-vms",
)
