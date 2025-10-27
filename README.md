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
- `app/` – main PySide6 application, custom video player, file browser, data sheet, and virtual field widgets.
- `scripts/` – helper scripts for processing videos and detections.
- `cache/` – runtime JSON caches (YOLO detections, yard markers, homography data).
- `yolo_models/` – YOLO weights tracked with Git LFS.
- `requirements.txt` – pinned Python dependencies for the GUI.
- `AUTHORS.txt` – contributors and acknowledgements.

See `docs.md` for expanded setup guidance and component notes.
