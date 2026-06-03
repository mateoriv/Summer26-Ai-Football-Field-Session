#!/usr/bin/env python3
"""
Static Process Script

Takes a video name and loads snap detection and homography data,
then extracts player data for snap frames. Uses the same offense-11
logic as the dataset builder, then runs the offense positions model
to predict play type and updates the folder CSV.
"""

import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations

FIELD_WIDTH_YD = 160/3 

# --- Offense-11 extraction (same logic as build_offense_positions_dataset) ---

def _normalize_class(name):
    return (name or "").strip().lower().replace(" ", "_")


def get_snap_frame(snap_detection_path):
    """Return snap frame number from snap_detection JSON, or None."""
    if not os.path.exists(snap_detection_path):
        return None
    with open(snap_detection_path, "r") as f:
        data = json.load(f)
    snaps = data.get("snaps") or []
    if not snaps:
        return None
    return snaps[0].get("frame")


def get_positions_at_snap(positions_path, snap_frame):
    """Load position JSON; return (detections at snap frame, image width)."""
    if not os.path.exists(positions_path):
        return [], 1920.0
    with open(positions_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames") or []
    width = float((data.get("video_info") or {}).get("width") or 1920)
    for fr in frames:
        if fr.get("frame_number") == snap_frame:
            return (fr.get("detections") or [], width)
    return ([], width)


def get_offense_side_from_positions(position_detections, image_width):
    """Which side is offense: 'left' or 'right' (more offense players left/right of defense median)."""
    offense_x = []
    defense_x = []
    for det in position_detections:
        cls_name = _normalize_class(det.get("class") or "")
        bbox = det.get("bbox") or {}
        cx = bbox.get("center_x")
        if cx is None:
            continue
        if cls_name == "ref":
            continue
        if cls_name == "defense":
            defense_x.append(float(cx))
        else:
            offense_x.append(float(cx))
    if not offense_x:
        return None
    if defense_x:
        defense_median = float(pd.Series(defense_x).median())
        offense_left = sum(1 for x in offense_x if x < defense_median)
        offense_right = sum(1 for x in offense_x if x > defense_median)
        if offense_left > offense_right:
            return "left"
        if offense_right > offense_left:
            return "right"
    offense_median = float(pd.Series(offense_x).median())
    return "right" if offense_median > (image_width / 2.0) else "left"


def get_normalized_positions_at_snap(homography_path, snap_frame):
    """Return list of detections for snap frame from homography JSON (normalized_position + original_bbox)."""
    if not os.path.exists(homography_path):
        return []
    with open(homography_path, "r") as f:
        data = json.load(f)
    frames = data.get("normalized_positions") or {}
    return frames.get(str(int(snap_frame))) or []


def take_first_11_on_side(detections, side):
    """From homography detections (normalized_position, original_bbox), take first 11 on side by normalized x."""
    points = []

    for det in detections:
        npos = det.get("normalized_position") or {}
        bbox = det.get("original_bbox") or {}
        nx = npos.get("x")
        ny = npos.get("y")
        ox = bbox.get("center_x")
        oy = bbox.get("center_y")
        if ox is None or oy is None or nx is None or ny is None:
            continue
        # Use a mutable list here so we can apply the same in-place
        # right-side normalization as build_offense_positions_dataset.
        points.append([float(nx), float(ny), float(ox), float(oy)])
    if not points:
        return []
    points.sort(key=lambda p: p[0])
    points = points[:11] if side == "left" else points[-11:]
    points.sort(key=lambda p: p[1])

    #Normalize attack from the left side to the right side
    if side == "right":
        x_min = []
        #Find Min x value
        for point in points:
            x_min.append(point[0])
        x_min_val = np.array(x_min).min()
        print(x_min_val)
        #Subtract 2 distance from LOS from all x values
        for point in points:
            dist = point[0] - x_min_val
            point[0] = float(point[0] - 2*dist)
    return points


def get_offense_points_for_video(video_name, folder_name, base_cache_dir):
    """
    Same pipeline as build_offense_positions_dataset.py, but for one clip.
    Returns (points_11, None) or (None, error_msg). points are (nx, ny, ox, oy).
    """
    snap_path = os.path.join(base_cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    positions_path = os.path.join(base_cache_dir, folder_name, "positions", f"{video_name}_position.json")
    homography_path = os.path.join(base_cache_dir, folder_name, "homography", f"{video_name}_normalized_positions.json")

    snap_frame = get_snap_frame(snap_path)
    if snap_frame is None:
        return None, "No snap frame", None

    position_detections, image_width = get_positions_at_snap(positions_path, snap_frame)
    offense_side = get_offense_side_from_positions(position_detections, image_width)
    if offense_side is None:
        return None, "Could not determine offense side", None

    normalized = get_normalized_positions_at_snap(homography_path, snap_frame)
    print(normalized)
    points = take_first_11_on_side(normalized, offense_side)
    print(points)
    if len(points) < 11:
        return None, f"Only {len(points)} players on offense side", None

    return points, offense_side, None


def get_offense_features_for_video(video_name, folder_name, base_cache_dir):
    """
    Returns (feature_vec_22, None) or (None, error_msg).
    Feature order matches the training CSV/metadata: nx1, ny1, nx2, ny2, ..., nx11, ny11.
    """
    points, _o_side, err = get_offense_points_for_video(video_name, folder_name, base_cache_dir)
    if points is None:
        return None, err

    interleaved = []
    for i in range(11):
        interleaved.append(points[i][0])  # nx
        interleaved.append(points[i][1])  # ny
    return np.array(interleaved, dtype=np.float32), None


def _update_offense_positions_csv(base_cache_dir, folder_name, video_name, points_11, label_value=""):
    """
    Update (or create) cache/<folder>/offense_positions.csv with this clip's offense points.
    Matches build_offense_positions_dataset.py output columns.
    """
    out_path = os.path.join(base_cache_dir, folder_name, "offense_positions.csv")

    columns = ["clip_name"]
    for i in range(11):
        columns.extend([f"nx{i+1}", f"ny{i+1}", f"ox{i+1}", f"oy{i+1}"])
    columns.append("label")

    if os.path.exists(out_path):
        out_df = pd.read_csv(out_path)
        for col in columns:
            if col not in out_df.columns:
                out_df[col] = ""
    else:
        out_df = pd.DataFrame(columns=columns)

    row = {"clip_name": str(video_name), "label": ("" if pd.isna(label_value) else str(label_value).strip())}
    for i in range(11):
        nx, ny, ox, oy = points_11[i]
        row[f"nx{i+1}"] = nx
        row[f"ny{i+1}"] = ny
        row[f"ox{i+1}"] = ox
        row[f"oy{i+1}"] = oy

    if "clip_name" in out_df.columns:
        matches = out_df.index[out_df["clip_name"].astype(str) == str(video_name)].tolist()
    else:
        matches = []

    if matches:
        idx = matches[0]
        existing_label = out_df.at[idx, "label"] if "label" in out_df.columns else ""
        if existing_label is not None and str(existing_label).strip():
            row["label"] = existing_label
        for k, v in row.items():
            out_df.at[idx, k] = v
    else:
        out_df = pd.concat([out_df, pd.DataFrame([row], columns=out_df.columns)], ignore_index=True)

    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Updated offense positions CSV: {out_path}")

def extract_geometric_features(features: torch.Tensor) -> torch.Tensor:
        # NORMALIZE FEATURES
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        features = features.float()
        coords = features.reshape(-1, 11, 2)

        # 1. Center (translation invariance)
        centroid = coords.mean(axis=1, keepdims=True)
        coords = coords - centroid
        
        B = coords.shape[0]

        # -------------------------------------------------
        # 1. Base Flattened Coordinates (11 * 2 = 22)
        # -------------------------------------------------
        base_features = coords.reshape(B, -1)

        # -------------------------------------------------
        # 2. Width & Depth (Span Features)
        # -------------------------------------------------
        x_max = torch.amax(coords[:, :, 0], dim=1)
        x_min = torch.amin(coords[:, :, 0], dim=1)
        x_span = x_max - x_min

        y_max = torch.amax(coords[:, :, 1], dim=1)
        y_min = torch.amin(coords[:, :, 1], dim=1)
        y_span = y_max - y_min

        span_features = torch.stack([x_span, y_span], dim=1)

        # -------------------------------------------------
        # 3. Mean Pairwise Distance
        # -------------------------------------------------
        pairwise_dists = []

        for i, j in combinations(range(11), 2):
            diff = coords[:, i] - coords[:, j]
            dist = torch.norm(diff, dim=1)
            pairwise_dists.append(dist)

        pairwise_dists = torch.stack(pairwise_dists, dim=1)

        mean_pairwise_distance = pairwise_dists.mean(dim=1, keepdim=True)

        # -------------------------------------------------
        # 4. Distance to Centroid Statistics
        # -------------------------------------------------
        d_to_centroid = torch.norm(coords - centroid, dim=2)

        centroid_mean = d_to_centroid.mean(dim=1, keepdim=True)
        centroid_std = d_to_centroid.std(dim=1, keepdim=True)

        centroid_features = torch.cat([centroid_mean, centroid_std], dim=1)

        # -------------------------------------------------
        # 5. PCA Eigenvalues (Top 2)
        # -------------------------------------------------
        # Compute covariance matrix per batch
        centered = coords - centroid
        cov = torch.bmm(centered.transpose(1, 2), centered) / 11.0 + 1e-6 * torch.eye(2).to(coords.device)

        eigvals = []

        eigvals_all = torch.linalg.eigvalsh(cov)
        eigvals_sorted = torch.sort(eigvals_all, dim=1).values
        eigvals = eigvals_sorted[:, -2:]

        # -------------------------------------------------
        # Concatenate Everything
        # -------------------------------------------------
        features = torch.cat(
            [
                base_features,
                span_features,
                mean_pairwise_distance,
                centroid_features,
                eigvals,
            ],
            dim=1,
        )

        return features
# --- Offense positions model inference (same architecture as train_offense_positions) ---

def _predict_play(features_22, model_dir):
    """
    Load model + metadata from model_dir, normalize features, run forward pass.
    Returns (predicted_label_str, confidence_float) or (None, None) if model missing/invalid.
    """

    # Trainer writes "model.pt"; keep legacy fallback for older names.
    model_path = os.path.join(model_dir, "formModel.pt")
    if not os.path.exists(model_path):
        return None, None
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return None, None

    with open(metadata_path, "r") as f:
        meta = json.load(f)
    index_to_label = meta.get("index_to_label") or {}
    # JSON keys may be strings "0", "1", ...
    index_to_label = {int(k): v for k, v in index_to_label.items()}
    mean = meta.get("mean")
    std = meta.get("std")
    num_features = int(meta.get("num_features", 22))
    num_classes = int(meta.get("num_classes", 2))
    hidden_dims = tuple((meta.get("train_args") or {}).get("hidden_dims") or [32, 16])
    dropout = float((meta.get("train_args") or {}).get("dropout") or 0.1)

    # Normalize
    
    x = np.array(features_22, dtype=np.float32).reshape(1, -1)
    x = extract_geometric_features(x)

    # Minimal MLP matching trainer
    class PositionsNet(torch.nn.Module):
        def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(torch.nn.Linear(prev, h))
                layers.append(torch.nn.ReLU())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
                prev = h
            layers.append(torch.nn.Linear(prev, num_classes))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PositionsNet(num_features, hidden_dims, num_classes, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(probs.argmax(dim=1).item())
    confidence = float(probs[0, pred_idx].item())
    label = index_to_label.get(pred_idx, str(pred_idx))
    return label, confidence


def load_data(file_path):
    """
    Load JSON data from a file.
    
    Args:
        file_path: Absolute path to the JSON file
    
    Returns:
        Dictionary with loaded JSON data, or None if not found
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"[INFO] Loaded data from: {file_path}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load data from {file_path}: {e}")
        return None


def get_player_data_for_frame(video_name, folder_name=None, cache_dir="cache", project_root=None):
    """
    Get player detection data for snap frames.
    
    Args:
        video_name: Name of the video (without extension)
        folder_name: Name of the folder containing the video (optional, will try to find)
        cache_dir: Cache directory name (default: "cache")
        project_root: Project root directory (defaults to parent of script directory)
    
    Returns:
        Dictionary with processed frame data for all snap frames, or None if not found
    """
    # Determine cache directory path
    # If cache_dir is absolute, use it directly; otherwise use project_root
    if os.path.isabs(cache_dir):
        base_cache_dir = cache_dir
    else:
        # Get project root if not provided
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
        base_cache_dir = os.path.join(project_root, cache_dir)
    
    if folder_name is None:
        print(f"[ERROR] Could not find folder containing snap detection for video: {video_name}")
        return None
    
    # Construct absolute paths
    snap_file_path = os.path.join(base_cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    homography_file_path = os.path.join(base_cache_dir, folder_name, "homography", f"{video_name}_normalized_positions.json")
    
    # Load snap detection data
    snap_data = load_data(snap_file_path)
    if snap_data is None:
        return None
    
    # Load homography data
    homography_data = load_data(homography_file_path)
    if homography_data is None:
        return None
    
    # Get snap frames
    snaps = snap_data.get('snaps', [])
    if not snaps:
        print(f"[WARNING] No snap frames found in snap detection")
        return None
    
    # Get normalized positions
    normalized_positions = homography_data.get('normalized_positions', {})
    
    # Process each snap frame
    results = []

    snap_frame_number = snaps[0].get('frame')
    snap_time = snaps[0].get('time', 0.0)
    
    # Find frame data in normalized positions (keys are strings)
    frame_key = str(snap_frame_number)
    frame_detections = normalized_positions.get(frame_key, [])
    
    if frame_detections:
        print(f"[SUCCESS] Found player data for snap frame {snap_frame_number}")
        results = {
            "snap_frame": snap_frame_number,
            "snap_time": snap_time,
            "detections": frame_detections
        }
    else:
        # Try to find closest frame
        closest_frame = None
        min_diff = float('inf')
        for frame_key_str, detections in normalized_positions.items():
            try:
                frame_num = int(frame_key_str)
                diff = abs(frame_num - snap_frame_number)
                if diff < min_diff:
                    min_diff = diff
                    closest_frame = (frame_num, detections)
            except ValueError:
                continue
        
        if closest_frame and min_diff <= 5:  # Within 5 frames
            frame_num, detections = closest_frame
            print(f"[INFO] Using closest frame {frame_num} for snap frame {snap_frame_number} (difference: {min_diff})")
            results.append({
                "snap_frame": snap_frame_number,
                "snap_time": snap_time,
                "actual_frame": frame_num,
                "frame_difference": min_diff,
                "detections": detections
            })
        else:
            print(f"[WARNING] Could not find player data for snap frame {snap_frame_number}")
    
    if not results:
        print(f"[ERROR] No player data found for any snap frames")
        return None
    return results


def _get_project_root():
    """Project root (parent of scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def process_frame_data(frame_data, video_name, folder_name=None, cache_dir="cache", project_root=None):
    """
    Process frame data and update the associated CSV file.
    Also runs offense positions model and writes predicted_play (and confidence) when possible.

    Args:
        frame_data: Dictionary with frame data containing player detections
        video_name: Name of the video (without extension) - used to find the CSV row
        folder_name: Name of the folder containing the video
        cache_dir: Cache directory name (default: "cache")
        project_root: Project root directory (defaults to parent of script directory)

    Returns:
        Dictionary with processed frame data and update status
    """
    if frame_data is None:
        return None

    # Determine cache directory path
    if os.path.isabs(cache_dir):
        base_cache_dir = cache_dir
    else:
        if project_root is None:
            project_root = _get_project_root()
        base_cache_dir = os.path.join(project_root, cache_dir)

    if folder_name is None:
        print(f"[ERROR] Folder name required to access CSV file")
        return None

    # Construct CSV file path
    csv_file_path = os.path.join(base_cache_dir, folder_name, f"{folder_name}_data.csv")
    print(f"[INFO] CSV file path: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        print(f"[ERROR] CSV file not found: {csv_file_path}")
        return None
    
    try:
        # Load CSV file
        df = pd.read_csv(csv_file_path)
        print(f"[INFO] Loaded CSV file: {csv_file_path}")
        
        # Find the row matching the video name
        # The CSV has a 'CLIP NAME' column that should match the video name
        video_row_index = None
        for idx, row in df.iterrows():
            clip_name = str(row.get('CLIP NAME', '')).strip()
            if clip_name == video_name:
                video_row_index = idx
                break
        
        if video_row_index is None:
            print(f"[WARNING] Video '{video_name}' not found in CSV file")
            print(f"[INFO] Available clip names: {df['CLIP NAME'].tolist()[:10]}...")  # Show first 10
            return frame_data
        
        print(f"[INFO] Found video '{video_name}' at row {video_row_index}")

        # Extract x/y positions from player detections.
        # frame_data may be either:
        # - dict {"detections": [...]}
        # - list [ {"detections": [...]}, ... ] (legacy/closest-frame path)
        if isinstance(frame_data, dict):
            detections = frame_data.get("detections", [])
        elif isinstance(frame_data, list) and frame_data:
            first_item = frame_data[0] if isinstance(frame_data[0], dict) else {}
            detections = first_item.get("detections", [])
        else:
            detections = []
            print("[WARNING] Unexpected frame_data format; no detections available for yard/hash update.")

        x_positions = []
        y_positions = []
        for detection in detections:
            normalized_pos = detection.get('normalized_position', {})
            x = normalized_pos.get('x')
            y = normalized_pos.get('y')
            if x is not None:
                x_positions.append(x)
                y_positions.append(y)

        # Update cache/<folder>/offense_positions.csv for this clip; get offense_side for yard-line logic
        points_11, o_side, pts_err = get_offense_points_for_video(video_name, folder_name, base_cache_dir)
        if points_11 is not None:
            # Use OFF FORM (if present) as the training label column in offense_positions.csv
            label_value = df.at[video_row_index, "OFF FORM"] if "OFF FORM" in df.columns else ""
            _update_offense_positions_csv(base_cache_dir, folder_name, video_name, points_11, label_value=label_value)
        elif pts_err:
            print(f"[INFO] Offense positions CSV skipped: {pts_err}")

        # Offense positions model: same 11-offense features as dataset builder, then predict
        proot = project_root if project_root is not None else _get_project_root()
        model_dir = os.path.join(proot, "models", "offense_positions")
        features, fe_err = get_offense_features_for_video(video_name, folder_name, base_cache_dir)
        if features is not None and os.path.isdir(model_dir):
            pred_label, confidence = _predict_play(features, model_dir)
            if pred_label is not None:
                df["OFF FORM"] = df["OFF FORM"].astype(object)
                df.at[video_row_index, "OFF FORM"] = pred_label
                df.at[video_row_index, "OFF FORM CONFIDENCE"] = confidence
                print(f"[INFO] Offense positions model: predicted_play={pred_label}, confidence={confidence:.3f}")
        elif fe_err:
            print(f"[INFO] Offense model skipped: {fe_err}")

        # Calculate median x position and round to nearest integer (use o_side from get_offense_points_for_video)
        if x_positions:
            median_x = np.median(x_positions)
            yard_line = int(round(median_x))
            print(f"[INFO] Calculated median x position: {median_x:.2f}, rounded to yard line: {yard_line}")

            if o_side is not None:
                if o_side == "left":
                    if yard_line >= 50:
                        yard_line = yard_line *(-1) + 100
                    else:
                        yard_line = yard_line * (-1)
                else:
                    if yard_line >= 50:
                        yard_line = yard_line - 100
                    else:
                        yard_line = yard_line

            # Update CSV row with yard line
            df.at[video_row_index, 'YARD LINE'] = yard_line
        else:
            print(f"[WARNING] No x positions found in detections, skipping yard line update")

        # Calculate Hash Side
        if y_positions:
           
            median_y = np.median(y_positions)
            top_hash = FIELD_WIDTH_YD/2 + 5
            bottom_hash = FIELD_WIDTH_YD/2 - 5
           
            if o_side == "left":
                if median_y > top_hash:
                    hash_side = "L"
                elif median_y < bottom_hash:
                    hash_side = "M"
                else:
                    hash_side = "R"
            else:
                if median_y > top_hash:
                    hash_side = "R"
                elif median_y < bottom_hash:
                    hash_side = "M"
                else:
                    hash_side = "L"
         
            df.at[video_row_index, 'HASH'] = hash_side
        else:
            print(f"[WARNING] No y positions found in detections, skipping hash side update")


        # Save the updated CSV
        df.to_csv(csv_file_path, index=False)
        print(f"[SUCCESS] Updated and saved CSV file: {csv_file_path}")
        
        return {
            "frame_data": frame_data,
            "csv_updated": True,
            "row_index": video_row_index
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Load player detection data for snap frames")
    parser.add_argument("--video-name", type=str, required=True,
                        help="Name of the video (without extension)")
    parser.add_argument("--folder-name", type=str, default=None,
                        help="Name of the folder containing the video (optional)")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Cache directory path (can be absolute or relative to project root)")
    args = parser.parse_args()
    
    print(f"Cache directory: {args.cache_dir}")
    print(f"Video name: {args.video_name}")
    print(f"Folder name: {args.folder_name}")
    
    # Get player data for snap frames
    frame_data = get_player_data_for_frame(
        video_name=args.video_name,
        folder_name=args.folder_name,
        cache_dir=args.cache_dir
    )
    
    if frame_data is None:
        print("[ERROR] Failed to retrieve player data for snap frames")
        sys.exit(1)
    
    # Process frame data and update CSV
    processed_frame_data = process_frame_data(
        frame_data=frame_data,
        video_name=args.video_name,
        folder_name=args.folder_name,
        cache_dir=args.cache_dir
    )
    
    if processed_frame_data:
        print("[SUCCESS] Processing completed successfully")
    else:
        print("[WARNING] Processing completed but CSV update may have failed")
    
    return processed_frame_data


if __name__ == "__main__":
    main()
