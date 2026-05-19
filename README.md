# Hudl AI Analysis

Desktop tooling for browsing football clips, drawing detections, and syncing metadata from CSV files. The application is built with PySide6 and OpenCV and expects pre-generated YOLO outputs that ship via Git LFS.

## Quick Start
- **Clone the repository**
  ```bash
  git clone https://github.com/<your-org>/Fall25-Ai-Football-Field-Session.git
  cd Fall25-Ai-Football-Field-Session
  ```
- **Install Git LFS before pulling large model assets**
  ```bash
  git lfs install # set hooks
  git lfs pull
  ```
- **Create a virtual environment (optional but recommended)**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate  # Windows: .venv\Scripts\activate
  ```
- **Install Python dependencies**
  ```bash
  pip install -r requirements.txt
  ```
- **Launch the UI**
  ```bash
  python app/application.py
  ```
- **Choose a working folder**
  - Use `File → Open Folder` to point the app at a directory containing videos (`.mp4`, `.mov`, …) and CSV data.
  - Select `File → Set Default Folder` to remember that directory. On macOS these menus live in the system menu bar (top of the screen) rather than the window itself.

That's it—video thumbnails, CSV grids, and the virtual field should populate automatically once a folder is selected.

## Repository Layout
- `app/` – main PySide6 application, custom video player, file browser, data sheet, virtual field, and processing dialogs.
- `scripts/` – helper scripts for the processing pipeline (Player Detection through Static Process). See `scripts/docs.md`.
- `models/` – offense-position model + metadata used by Step 7 (`staticProcess.py`).
- `yolo_models/` – YOLO weights tracked with Git LFS.
- `modelTraining/` – training utilities for the offense-positions model (not bundled in the EXE).
- `CNN/` – legacy folder. `CNN/staticProcess.py` is a compatibility shim that forwards to `scripts/staticProcess.py`; the rest is experimental Kaggle data not used by the app.
- `cache/` – runtime JSON caches (YOLO detections, yard markers, homography data, per-folder data sheet CSV). Generated at runtime, not checked in.
- `hudl_ai.spec` – PyInstaller spec for building the desktop EXE. See `build_docs.md`.
- `requirements.txt` – pinned Python dependencies for the GUI and processing scripts.
- `AUTHORS.txt` – contributors and acknowledgements.

See `docs.md` for expanded setup guidance and component notes, and `build_docs.md` for distributable build instructions.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before working on the project.
