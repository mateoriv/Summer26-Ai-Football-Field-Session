#!/bin/bash
# Headless single-clip pipeline runner — mirrors app/processingDialog.py exactly.
# Usage: ./run_one_clip.sh "data/CSAI_FORMATIONS/Wide - Clip 001.mp4"
set -e

VIDEO="$1"
FOLDER="CSAI_FORMATIONS"
CACHE="cache"
NAME="$(basename "$VIDEO")"; NAME="${NAME%.*}"

DET="$CACHE/$FOLDER/players/${NAME}_detection.json"
SNAP="$CACHE/$FOLDER/snap_detection/${NAME}_snap_detection.json"
POS="$CACHE/$FOLDER/positions/${NAME}_position.json"
YARD="$CACHE/$FOLDER/yard_markers/${NAME}_yard_markers.json"
CORR="$CACHE/$FOLDER/correspondence/${NAME}_correspondence.json"
HOMO="$CACHE/$FOLDER/homography/${NAME}_normalized_positions.json"

mkdir -p "$CACHE/$FOLDER"/{players,snap_detection,positions,yard_markers,correspondence,homography}

echo "=== [1/7] player detection ==="
python3 scripts/playerDetection.py --video "$VIDEO" --output "$DET"
echo "=== [2/7] snap detection ==="
python3 scripts/snapDetection.py --player-detections "$DET" --output "$SNAP"
echo "=== [3/7] position detection ==="
python3 scripts/positionDetection.py --video "$VIDEO" --output "$POS" --snap-detection "$SNAP"
echo "=== [4/7] yard marker detection ==="
python3 scripts/yardMarkerDetection.py --video "$VIDEO" --output "$YARD"
echo "=== [5/7] correspondence points ==="
python3 scripts/autoCorrespondancePoints.py --detection-json "$YARD" --output "$CORR" --confidence 0.7 --per-frame
echo "=== [6/7] homography transform ==="
python3 scripts/perFrameHomographyTransform.py --position-detections "$DET" --correspondence-points "$CORR" --output "$HOMO"
echo "=== [7/7] static process ==="
python3 scripts/staticProcess.py --video-name "$NAME" --folder-name "$FOLDER" --cache-dir "$CACHE"

echo "=== DONE — outputs under $CACHE/$FOLDER/ ==="
