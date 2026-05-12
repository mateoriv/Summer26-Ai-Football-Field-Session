# Scripts Overview

The `scripts/` folder collects the command-line utilities that generate model outputs and intermediate assets consumed by the desktop app. They are designed to be run individually while iterating on a single clip, or orchestrated via the processing dialogs in the UI (`app/processingDialog.py` for a single clip, `app/batchProcessingDialog.py` for a folder).

## Common Requirements
- Python 3.10+
- Dependencies from the project `requirements.txt` (`opencv-python`, `ultralytics`, `numpy`, `pandas`, `scipy`, `torch`, etc.).
- YOLO weights synced via `git lfs pull` under `yolo_models/`.

## Script Catalog

The UI processing pipeline runs these scripts in order (steps 1–7):

- `playerDetection.py` *(Step 1)*
  Runs an Ultralytics YOLO model (`yolo_models/bestPlayerDetectorM.pt`) over a video and produces frame-by-frame player bounding boxes. Saves JSON with `frames[]` entries that the UI overlays in `video.py`.
  ```bash
  python scripts/playerDetection.py --video path/to/game.mp4 --output cache/session/players/game_detection.json
  ```

- `snapDetection.py` *(Step 2)*
  Analyzes the player detection JSON to find the snap frame for each clip. Writes a snap-detection JSON with one or more `snaps[]` entries (frame number + timestamp) used by later steps (position detection, static process) and by the UI timeline markers.
  ```bash
  python scripts/snapDetection.py --player-detections cache/session/players/game_detection.json --output cache/session/snap_detection/game_snap_detection.json
  ```

- `positionDetection.py` *(Step 3)*
  Runs a separate YOLO model (`yolo_models/positionDetection.pt`) on the snap frame(s) to classify offense/defense/ref/etc. Produces a `position` JSON consumed by static process to determine offense side and orientation.
  ```bash
  python scripts/positionDetection.py --video path/to/game.mp4 --output cache/session/positions/game_position.json --snap-detection cache/session/snap_detection/game_snap_detection.json
  ```

- `yardMarkerDetection.py` *(Step 4)*
  Detects sideline yard markers using `yolo_models/yardMarkerDetection.pt`. Outputs a JSON payload mirroring the player format but focused on marker classes like `fl1`, `nr3`, etc. These detections drive correspondence estimation and field calibration.
  ```bash
  python scripts/yardMarkerDetection.py --video path/to/game.mp4 --output cache/session/yard_markers/game_yard_markers.json
  ```

- `autoCorrespondancePoints.py` *(Step 5)*
  Consumes yard-marker detections and maps them to canonical NCAA field coordinates. Filters by confidence, averages repeated detections, and emits per-frame `frame_correspondences` describing pixel ↔ field pairs suitable for homography.
  ```bash
  python scripts/autoCorrespondancePoints.py \
    --detection-json cache/session/yard_markers/game_yard_markers.json \
    --output cache/session/correspondence/game_correspondence.json \
    --confidence 0.7 \
    --per-frame
  ```

- `perFrameHomographyTransform.py` *(Step 6)*
  Applies the correspondence points to player detections on a per-frame basis. Each player's bounding-box center is projected into field coordinates (yards), producing `normalized_positions` consumed by `virtualField.py`.
  ```bash
  python scripts/perFrameHomographyTransform.py \
    --position-detections cache/session/players/game_detection.json \
    --correspondence-points cache/session/correspondence/game_correspondence.json \
    --output cache/session/homography/game_normalized_positions.json
  ```

- `staticProcess.py` *(Step 7)*
  Final analysis step. Loads the snap-frame data and homography output, picks the 11 offense players, runs the offense-positions MLP from `models/offense_positions/` to predict the play type (`OFF FORM`), computes yard line + hash side from median positions, and updates the folder-level metadata CSV (`cache/<folder>/<folder>_data.csv`). Also appends a row to `cache/<folder>/offense_positions.csv` for training-data reuse.
  ```bash
  python scripts/staticProcess.py \
    --video-name "Wide - Clip 001" \
    --folder-name DemoFolder \
    --cache-dir cache
  ```
  A thin compatibility shim still exists at `CNN/staticProcess.py` that forwards to this script.

### Other utilities

- `renderFieldVideo.py`
  Turns normalized field coordinates into a rendered video of dots moving across a stylized football field. Useful for debugging homography pipelines or sharing analytics clips.
  ```bash
  python scripts/renderFieldVideo.py \
    --input cache/session/homography/game_normalized_positions.json \
    --output cache/session/results/game_field.mp4 \
    --fps 30 --frame-skip 2
  ```

- `extractSnapFrames.py`
  Helper for dataset construction — extracts the snap frame image(s) from a video for labeling or model training. Not part of the runtime UI pipeline.

- `__init__.py`
  Empty initializer; keeps the folder importable so dialogs can run scripts as modules.

## Typical Processing Flow (manual)
1. `playerDetection.py` → `snapDetection.py` → `positionDetection.py` (player + snap + position data)
2. `yardMarkerDetection.py` → `autoCorrespondancePoints.py` → `perFrameHomographyTransform.py` (field calibration + projection)
3. `staticProcess.py` (final metadata + form prediction in the data sheet CSV)
4. Optionally visualize the result with `renderFieldVideo.py`.
5. Launch the desktop app and open the session folder so the UI can pick up the newly generated JSON files.

The processing dialogs in `app/processingDialog.py` and `app/batchProcessingDialog.py` stitch steps 1–7 together end-to-end, either per-clip or across a whole folder. When the build is frozen by PyInstaller, the UI re-invokes the bundled executable with `PYINSTALLER_RUN_SCRIPT` set so each script runs inside the frozen interpreter rather than requiring system Python.
