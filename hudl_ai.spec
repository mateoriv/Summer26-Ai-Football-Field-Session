# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for building the Hudl AI desktop UI as a single-file binary.
The spec is configured to bundle the PySide6 Qt plugins plus the local assets
that the application expects to find relative to the project root:
    - yolo_models/  (Git LFS weights)
    - cache/        (precomputed detection + homography samples)
    - scripts/      (helper pipelines invoked from the UI)
    - CNN/          (static processing helper)

Hidden imports include cv2 (OpenCV) and numpy to ensure they are properly
bundled with the executable, as they are used throughout the application
and in subprocess scripts.
"""

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent
else:
    # When the spec is executed directly from the CLI, __file__ can be missing.
    PROJECT_ROOT = Path.cwd()
PROJECT_ROOT = str(PROJECT_ROOT)
APP_DIR = os.path.join(PROJECT_ROOT, "app")

def collect_directory(src_root: str, target_prefix: str):
    """Recursively collect files under src_root and map them under target_prefix."""
    collected = []
    for root, _, files in os.walk(src_root):
        for filename in files:
            src_path = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, src_root)
            if rel_dir == ".":
                rel_dir = ""
            dest_dir = os.path.join(target_prefix, rel_dir) if rel_dir else target_prefix
            collected.append((src_path, dest_dir))
    return collected

# Bundle project asset directories (plus app modules for subprocess access).
asset_dirs = ["app", "scripts", "yolo_models", "cache", "CNN"]
asset_datas = []
for rel_path in asset_dirs:
    src_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(src_path):
        asset_datas.extend(collect_directory(src_path, rel_path))

datas = asset_datas
binaries = []
hiddenimports = ["darkdetect", "cv2", "numpy"]

# Collect all dependencies from packages used in scripts
# This ensures PyInstaller includes all submodules and dependencies
ultralytics_datas, ultralytics_binaries, ultralytics_hiddenimports = collect_all("ultralytics")
torch_datas, torch_binaries, torch_hiddenimports = collect_all("torch")
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all("cv2")

# Merge collected data
datas += ultralytics_datas + torch_datas + cv2_datas
binaries += ultralytics_binaries + torch_binaries + cv2_binaries
hiddenimports += ultralytics_hiddenimports + torch_hiddenimports + cv2_hiddenimports

# Add script files to be analyzed (so PyInstaller detects their imports)
script_files = [
    "scripts/playerDetection.py",
    "scripts/positionDetection.py",
    "scripts/snapDetection.py",
    "scripts/yardMarkerDetection.py",
    "scripts/autoCorrespondancePoints.py",
    "scripts/perFrameHomographyTransform.py",
    "CNN/staticProcess.py",
]

# Add scripts to analysis so their imports are detected
script_paths = [os.path.join(PROJECT_ROOT, script) for script in script_files if os.path.exists(os.path.join(PROJECT_ROOT, script))]

# Create analysis with main app and scripts (scripts are analyzed but not executed)
analysis_scripts = ["app/application.py"] + script_paths

a = Analysis(
    analysis_scripts,
    pathex=[PROJECT_ROOT, APP_DIR, os.path.join(PROJECT_ROOT, "scripts"), os.path.join(PROJECT_ROOT, "CNN")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="hudl_ai",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # True to enable console window to see debug output, or False for no debug output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
