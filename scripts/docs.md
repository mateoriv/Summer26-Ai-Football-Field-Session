# Scripts Overview

The `scripts/` folder collects the command-line utilities that generate model outputs and intermediate assets consumed by the desktop app. They are designed to be run individually while iterating on a single clip, or orchestrated via the batch dialogs in the UI.

## Common Requirements
- Python 3.10+
- Dependencies from the project `requirements.txt` (`opencv-python`, `ultralytics`, `numpy`, `pandas`, etc.).
- YOLO weights synced via `git lfs pull` under `yolo_models/`.

## Script Catalog
- `playerDetection.py`  
  Runs an Ultralytics YOLO model (`yolo_models/bestPlayerDetectorM.pt`) over a video and produces frame-by-frame player bounding boxes. Saves JSON with `frames[]` entries that the UI overlays in `video.py`.
  ```bash
  python scripts/playerDetection.py --video path/to/game.mp4 --output cache/session/players/game_detection.json
  ```

- `yardMarkerDetection.py`  
  Detects sideline yard markers using a specialized YOLO model (`yolo_models/yardMarker2.pt`). Outputs a JSON payload mirroring the player format but focused on marker classes like `fl1`, `nr3`, etc. These detections can drive correspondence estimation and field calibration.
  ```bash
  python scripts/yardMarkerDetection.py --video path/to/game.mp4 --output cache/session/yard_markers/game_yard_markers.json
  ```

- `autoCorrespondancePoints.py`  
  Consumes yard-marker detections and maps them to canonical NCAA field coordinates. Filters by confidence, averages repeated detections, and emits `frame_correspondences` describing pixel ↔ field pairs suitable for homography.
  ```bash
  python scripts/autoCorrespondancePoints.py \
    --detection-json cache/session/yard_markers/game_yard_markers.json \
    --output cache/session/correspondence/game_correspondence.json \
    --confidence 0.7
  ```

- `perFrameHomographyTransform.py`  
  Applies the correspondence points to player detections on a per-frame basis. Each player's bounding-box center is projected into field coordinates (yards), producing `normalized_positions` consumed by `virtualField.py`.
  ```bash
  python scripts/perFrameHomographyTransform.py \
    --players cache/session/players/game_detection.json \
    --correspondence cache/session/correspondence/game_correspondence.json \
    --output cache/session/homography/game_normalized.json
  ```

- `renderFieldVideo.py`  
  Turns the normalized field coordinates into a rendered video of dots moving across a stylized football field. Useful for debugging homography pipelines or sharing analytics clips.
  ```bash
  python scripts/renderFieldVideo.py \
    --input cache/session/homography/game_normalized.json \
    --output cache/session/results/game_field.mp4 \
    --fps 30 --frame-skip 2
  ```

- `__init__.py`  
  Empty initializer; keeps the folder importable so dialogs can run scripts as modules.

## Typical Processing Flow
1. Run `playerDetection.py` and `yardMarkerDetection.py` to create base detections.
2. Derive correspondence points with `autoCorrespondancePoints.py`.
3. Project detections with `perFrameHomographyTransform.py` to generate normalized field positions.
4. Optionally visualize the result with `renderFieldVideo.py`.
5. Launch the desktop app and open the session folder so the UI can pick up the newly generated JSON files.

The batch-processing dialog in `app/batchProcessingDialog.py` stitches steps 1–2 together for an entire folder if you prefer a GUI trigger.
