# Project Guide

This document expands on the quick-start steps in `README.md`. It covers dependency management, Git LFS installation, the machine learning tooling behind the scenes, and a guided tour of the repository.

## Prerequisites
- **Python**: Tested with Python 3.10+. Other 3.9+ builds should work, but keep the pinned requirements in sync.
- **pip**: Ships with Python; upgrade if you see installation errors.
- **Git LFS**: Required to download the YOLO model weights stored in `yolo_models/`.
- **Virtual environment**: Optional, but helps isolate dependencies.

## Git LFS Installation
Install Git LFS before cloning or immediately after, then run `git lfs install` and `git lfs pull`.

- **macOS (Homebrew)**
  ```bash
  brew install git-lfs
  git lfs install
  ```
- **Windows (Chocolatey or Installer)**
  ```powershell
  choco install git-lfs
  git lfs install
  ```
  Alternatively, download the [Git LFS installer](https://git-lfs.com/) and follow the wizard.
- **Linux (Debian/Ubuntu)**
  ```bash
  sudo apt update
  sudo apt install git-lfs
  git lfs install
  ```
- **Linux (Fedora/RHEL)**
  ```bash
  sudo dnf install git-lfs
  git lfs install
  ```

Once installed, pull the tracked artifacts (if contents of files in yolo_models are txt files):
```bash
git lfs pull
```

## Environment Setup
1. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   python app/application.py
   ```
4. Pick a working directory containing videos and CSV metadata via `File → Open Folder`. Use `File → Set Default Folder` to remember the location (on macOS the menu lives in the OS menu bar).

## YOLO & Ultralytics Components
- YOLO weights live under `yolo_models/` and are synced with Git LFS.
- The detection scripts in `scripts/` rely on **Ultralytics** utilities (`ultralytics` Python package) to run inference for players and yard markers.
- Generated detection JSONs land in `cache/<session>/players` or `cache/<session>/yard_markers`. The application loads these files to paint bounding boxes on top of video frames.
- Homography transforms are pre-computed per frame (`scripts/perFrameHomographyTransform.py`) and consumed by the virtual field overlay.

## Repository Walkthrough
- `app/`: PySide6 desktop UI. See `app/docs.md` for per-module details.
- `scripts/`: Processing scripts for detections, homography, and rendering. Documented in `scripts/docs.md`.
- `cache/`: Runtime cache populated by the scripts and UI (safe to delete between runs).
- `yolo_models/`: YOLO weight files tracked with Git LFS.
- `requirements.txt`: Locked Python dependencies for the UI and processing scripts.
- `AUTHORS.txt`: Contributors.
- `README.md`: Quick start reference.
- `docs.md`: This document.

## Running The Pipeline End-to-End
1. Add raw videos to a working folder.
2. Populate YOLO detections with the `scripts/playerDetection.py` and `scripts/yardMarkerDetection.py` helpers (see `scripts/docs.md`).
3. Generate per-frame homography data if you need the virtual field overlay.
4. Launch the desktop app and point it at the folder. The file browser auto-loads the first CSV/video, the video dock renders clips with bounding boxes, and the virtual field reflects homography data.

## Troubleshooting
- **Menus missing on macOS**: Qt apps place menus in the global menu bar; make sure the Hudl AI window has focus.
- **No videos listed**: Confirm the selected folder contains `.mp4` files; the browser filters for common video extensions.
- **Bounding boxes not showing**: Ensure the YOLO JSON files exist in `cache/<folder-name>/players` and toggle the UI buttons to enable overlays.
- **Virtual field empty**: Verify homography results are present under `cache/<folder-name>/homography`.

Questions or gaps? Add clarifying notes directly in the relevant `docs.md` file so future runs are smoother.
