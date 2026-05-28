#!/usr/bin/env python3
"""
Snap Detection Script

Detects the snap frame in a football play clip by finding the
calm-to-burst transition in player motion.

Handles the real-world events that can cause false positives:

  Event                       How it's handled
  --------------------------  -----------------------------------------------
  Huddle break                Skip the first `--skip-start-seconds` of the clip
  Men in motion (pre-snap)    Require calm for at least `--set-window-seconds`
                              (longer than the smooth window)
  Hard count / offsides move  Detect "spike then return" patterns and reject them
  Camera pan / tilt           Remove median frame-level shift before computing
                              per-player velocity
  Referee walking             No class filter available; mitigated by requiring
                              collective (multi-player) motion
  Clustered duplicate frames  Deduplicate: keep only the best candidate per
                              `--cluster-gap-seconds` window
  Snap near end of clip       Hard cap at `--max-snap-fraction` of total frames
"""

import json
import os
import sys
import argparse
import numpy as np
from scipy.spatial.distance import cdist


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def load_player_detections(path):
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------------
# Player Matching
# ------------------------------------------------------------------

def match_players(prev_centers, curr_centers):
    """
    Nearest-neighbour matching between two sets of player centers.
    Returns an (N, 2) array of displacement vectors for matched pairs.
    """
    if len(prev_centers) == 0 or len(curr_centers) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    dists = cdist(prev_centers, curr_centers)
    used_curr = set()
    deltas = []
    for i in range(len(prev_centers)):
        j = int(np.argmin(dists[i]))
        if j in used_curr:
            continue
        used_curr.add(j)
        deltas.append(curr_centers[j] - prev_centers[i])

    return np.array(deltas, dtype=np.float32) if deltas else np.zeros((0, 2), dtype=np.float32)


# ------------------------------------------------------------------
# Velocity Computation
# ------------------------------------------------------------------

def compute_velocity(detections, min_players_per_frame=15):
    """
    Returns:
        velocities       – per-frame scalar (camera-corrected, outlier-filtered)
        mover_counts     – per-frame count of players whose corrected motion
                           exceeds a small threshold (used for multi-player check)
        player_counts    – per-frame raw player detection count
        fps

    Frames with fewer than *min_players_per_frame* detections are treated as
    unreliable: velocity and mover count are set to 0 and prev_centers is reset
    so adjacent sparse frames cannot contaminate the velocity signal.
    """
    fps = detections.get("video_info", {}).get("fps", 30.0)
    frames = detections.get("frames", [])

    velocities = []
    mover_counts = []
    player_counts = []
    prev_centers = None
    small_move_threshold = 1.5   # pixels; below this = "not really moving"

    for frame in frames:
        centers = []
        for det in frame.get("detections", []):
            bbox = det.get("bbox", {})
            if "center_x" in bbox and "center_y" in bbox:
                centers.append([bbox["center_x"], bbox["center_y"]])

        n_players = len(centers)
        player_counts.append(n_players)
        centers = np.array(centers, dtype=np.float32)

        # Frames below the minimum player threshold are unreliable – skip and reset
        if n_players < min_players_per_frame:
            velocities.append(0.0)
            mover_counts.append(0)
            prev_centers = None   # reset so next valid frame starts fresh
            continue

        if prev_centers is None or len(prev_centers) == 0 or len(centers) == 0:
            velocities.append(0.0)
            mover_counts.append(0)
            prev_centers = centers
            continue

        matched = match_players(prev_centers, centers)

        if len(matched) == 0:
            velocities.append(0.0)
            mover_counts.append(0)
            prev_centers = centers
            continue

        # Camera motion = median displacement across all matched players
        cam_shift = np.median(matched, axis=0)
        corrected = matched - cam_shift

        # Weight horizontal motion higher (football is primarily horizontal)
        weights = np.array([1.0, 0.4])
        magnitudes = np.linalg.norm(corrected * weights, axis=1)

        # Remove per-frame outliers (e.g. a single ref running across field)
        if len(magnitudes) > 3:
            q1, q3 = np.percentile(magnitudes, [25, 75])
            iqr = q3 - q1
            cap = q3 + 1.5 * iqr if iqr > 0 else np.inf
            filtered = magnitudes[magnitudes <= cap]
            avg_vel = np.mean(filtered) if len(filtered) > 0 else np.median(magnitudes)
        else:
            avg_vel = float(np.mean(magnitudes))

        # Count how many players are genuinely moving (for multi-player check)
        movers = int(np.sum(magnitudes > small_move_threshold))

        velocities.append(avg_vel)
        mover_counts.append(movers)
        prev_centers = centers

    return (
        np.array(velocities, dtype=np.float32),
        np.array(mover_counts, dtype=np.int32),
        np.array(player_counts, dtype=np.int32),
        float(fps),
    )


# ------------------------------------------------------------------
# Hard-Count / Offsides Detection
# ------------------------------------------------------------------

def build_hardcount_mask(smoothed, fps, threshold_mult=0.6, return_frames=8):
    """
    Mark frames that follow a "spike-then-return" pattern as hard-count zones.
    A spike that returns to calm within `return_frames` is NOT a snap.

    Returns a boolean mask (True = likely hard count, do not call snap here).
    """
    motion_threshold = np.percentile(smoothed, 70) * threshold_mult
    mask = np.zeros(len(smoothed), dtype=bool)

    i = 0
    while i < len(smoothed):
        if smoothed[i] > motion_threshold:
            # Find how long the spike lasts
            j = i
            while j < len(smoothed) and smoothed[j] > motion_threshold:
                j += 1
            spike_len = j - i
            # If the spike is short and velocity returns to near-calm, it's a hard count
            if spike_len < return_frames and j < len(smoothed):
                after_calm = np.mean(smoothed[j:min(j + return_frames, len(smoothed))])
                calm_base = np.percentile(smoothed, 30)
                if after_calm < calm_base * 1.5:
                    # Mark the spike and a small buffer around it
                    buffer = return_frames
                    mask[max(0, i - buffer):min(len(smoothed), j + buffer)] = True
            i = j
        else:
            i += 1

    return mask


# ------------------------------------------------------------------
# Candidate Deduplication
# ------------------------------------------------------------------

def cluster_candidates(candidates, gap_frames):
    """
    Keep only the highest-confidence candidate within each `gap_frames` window.
    This collapses "frame 1058, 1059, 1060 all from the same snap" into one result.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["frame"])
    clusters = []
    group = [candidates[0]]

    for c in candidates[1:]:
        if c["frame"] - group[-1]["frame"] <= gap_frames:
            group.append(c)
        else:
            clusters.append(max(group, key=lambda x: x["confidence"]))
            group = [c]
    clusters.append(max(group, key=lambda x: x["confidence"]))

    return clusters


# ------------------------------------------------------------------
# Snap Detection
# ------------------------------------------------------------------

def detect_snaps(
    velocities,
    mover_counts,
    fps,
    player_counts=None,
    skip_start_frames=0,
    set_window_seconds=1.5,
    min_movers=4,
    min_players_per_frame=15,
    max_snap_fraction=0.95,
    cluster_gap_seconds=0.5,
    top_k=1,
):
    """
    Find the snap frame.

    Parameters
    ----------
    velocities            : camera-corrected per-frame velocity array
    mover_counts          : per-frame count of players genuinely moving
    fps                   : frames per second
    player_counts         : per-frame raw player detection count (from compute_velocity)
    skip_start_frames     : ignore this many frames at the start (huddle break)
    set_window_seconds    : how long (in seconds) of calm is required before a
                            snap candidate is accepted (set / offsides guard)
    min_movers            : minimum number of players that must move simultaneously
                            (filters refs, isolated pre-snap shifts)
    min_players_per_frame : frame must have at least this many detected players to
                            be eligible as a snap candidate (default: 15)
    max_snap_fraction     : ignore candidates beyond this fraction of total frames
    cluster_gap_seconds   : deduplicate candidates closer than this
    top_k                 : how many final snap candidates to return
    """
    n = len(velocities)
    if n == 0:
        return []

    # Smooth velocity (~0.4 s window)
    smooth_win = max(5, int(fps * 0.4))
    kernel = np.ones(smooth_win) / smooth_win
    smoothed = np.convolve(velocities, kernel, mode="same")

    # Smooth mover counts the same way for stability
    smoothed_movers = np.convolve(mover_counts.astype(np.float32), kernel, mode="same")

    # Eligibility mask: frames below the player threshold were forced to
    # velocity 0 in compute_velocity. On long clips that are ~half wide/huddle
    # shots, those zeros dominate and drag the percentiles toward 0 --
    # calm_threshold becomes exactly 0, the calm gate `pre_mean < 0` can never
    # pass, and the detector finds 0 snaps. Compute thresholds over the eligible
    # (enough-players) frames only so they reflect real motion.
    if player_counts is not None and len(player_counts) == n:
        eligible_mask = np.asarray(player_counts) >= min_players_per_frame
    else:
        eligible_mask = np.ones(n, dtype=bool)
    valid_smoothed = smoothed[eligible_mask]
    if len(valid_smoothed) < 5:
        valid_smoothed = smoothed

    # Adaptive thresholds (over eligible frames only)
    calm_threshold  = np.percentile(valid_smoothed, 30)
    motion_threshold = np.percentile(valid_smoothed, 70)
    accel = np.gradient(np.gradient(smoothed))
    accel_threshold = np.percentile(np.abs(accel), 80)

    print(
        f"[INFO] Thresholds: calm={calm_threshold:.2f}, "
        f"motion={motion_threshold:.2f}, accel={accel_threshold:.4f}"
    )

    # Hard-count mask: avoid calling snap during spike-then-return events
    hardcount_mask = build_hardcount_mask(smoothed, fps)

    # Set window: how many frames of calm are required before the snap
    set_window_frames = max(smooth_win, int(fps * set_window_seconds))

    # Look-ahead: confirm motion is actually coming
    look_ahead = max(5, int(fps * 0.3))

    # Max frame cap
    max_frame = int(n * max_snap_fraction)

    # Build a boolean mask: True where the frame has enough players to be eligible
    if player_counts is not None and len(player_counts) == n:
        enough_players = player_counts >= min_players_per_frame
    else:
        enough_players = np.ones(n, dtype=bool)

    candidates = []

    start = max(skip_start_frames, set_window_frames)
    for i in range(start, max_frame - look_ahead):

        # Gate 6: must have enough detected players (hard gate, never relaxed)
        if not enough_players[i]:
            continue

        # Skip hard-count / offsides zones
        if hardcount_mask[i]:
            continue

        pre_window = smoothed[i - set_window_frames:i]
        future_window = smoothed[i:i + look_ahead]

        pre_mean = float(np.mean(pre_window))
        pre_std  = float(np.std(pre_window))
        future_avg = float(np.mean(future_window))
        future_max = float(np.max(future_window))

        # Gate 1: long calm before snap
        calm_ok = pre_mean < calm_threshold * 1.2

        # Gate 2: stability of calm period (not mid-motion-shift)
        stable_ok = pre_std < (calm_threshold * 0.6)

        # Gate 3: motion clearly coming
        motion_ok = future_avg > motion_threshold * 0.4 or future_max > motion_threshold * 0.6

        # Gate 4: acceleration spike at this frame
        accel_ok = np.abs(accel[i]) > accel_threshold * 0.8

        # Gate 5: multiple players moving simultaneously (not just 1-2)
        movers_ok = smoothed_movers[i] >= min_movers

        if calm_ok and stable_ok and motion_ok and (accel_ok or movers_ok):
            confidence = (
                (motion_threshold - pre_mean)             # reward calm before
                + np.abs(accel[i]) * 2.0                  # reward sharp transition
                + (future_avg - pre_mean)                  # reward big velocity jump
                + float(smoothed_movers[i]) * 0.3         # reward many movers
            )
            frame = int(i) - 2*fps
            candidates.append({
                "frame": int(frame),
                "time": round(frame/ fps, 3),
                "confidence": float(confidence),
            })

    if not candidates:
        print("[WARNING] No snap candidates found with all gates. Relaxing stability gate.")
        # Fallback: relax the stability gate (but keep Gate 6 — player count is never relaxed)
        for i in range(start, max_frame - look_ahead):
            if not enough_players[i]:
                continue
            if hardcount_mask[i]:
                continue
            pre_window = smoothed[i - set_window_frames:i]
            future_window = smoothed[i:i + look_ahead]
            pre_mean = float(np.mean(pre_window))
            future_avg = float(np.mean(future_window))
            future_max = float(np.max(future_window))
            calm_ok   = pre_mean < calm_threshold * 1.5
            motion_ok = future_avg > motion_threshold * 0.3 or future_max > motion_threshold * 0.5
            accel_ok  = np.abs(accel[i]) > accel_threshold * 0.5
            if calm_ok and motion_ok and accel_ok:
                confidence = (
                    (motion_threshold - pre_mean)
                    + np.abs(accel[i]) * 2.0
                    + (future_avg - pre_mean)
                )
                frame = max(0, int(i) - 2*fps)
                candidates.append({
                    "frame": int(frame),
                    "time": round(frame / fps, 3),
                    "confidence": float(confidence),
                })

    if not candidates:
        return []

    # Deduplicate: collapse adjacent frames from the same snap event
    cluster_gap = max(1, int(fps * cluster_gap_seconds))
    candidates = cluster_candidates(candidates, cluster_gap)

    # Sort by confidence and return top-K
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates[:top_k]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Snap Detection")
    parser.add_argument("--player-detections", required=True,
                        help="Path to player detection JSON")
    parser.add_argument("--output", required=True,
                        help="Path to output JSON")
    parser.add_argument("--skip-start-seconds", type=float, default=0,
                        help="Ignore first N seconds (huddle break). Default 0")
    parser.add_argument("--set-window-seconds", type=float, default=1,
                        help="Required calm window before snap (seconds). Default 1")
    parser.add_argument("--min-movers", type=int, default=4,
                        help="Minimum players moving at snap. Default 4.")
    parser.add_argument("--min-players", type=int, default=15,
                        help="Minimum detected players required for a frame to be a snap candidate. Default 15.")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Number of snap candidates to return. Default 1.")
    args = parser.parse_args()

    print(f"[INFO] Loading detections from: {args.player_detections}")
    raw = load_player_detections(args.player_detections)

    print("[INFO] Computing camera-corrected velocities...")
    velocities, mover_counts, player_counts, fps = compute_velocity(raw, min_players_per_frame=args.min_players)

    if len(velocities) == 0:
        print("[ERROR] No velocity data computed.")
        sys.exit(1)

    eligible = int(np.sum(player_counts >= args.min_players))
    print(f"[INFO] Velocity stats: mean={np.mean(velocities):.2f}, max={np.max(velocities):.2f}")
    print(f"[INFO] {eligible}/{len(player_counts)} frames have >= {args.min_players} players (eligible for snap).")

    skip_frames = int(fps * args.skip_start_seconds)
    print(f"[INFO] Skipping first {skip_frames} frames (huddle break guard).")

    snaps = detect_snaps(
        velocities,
        mover_counts,
        fps,
        player_counts=player_counts,
        skip_start_frames=skip_frames,
        set_window_seconds=args.set_window_seconds,
        min_movers=args.min_movers,
        min_players_per_frame=args.min_players,
        top_k=args.top_k,
    )

    print(f"[SUCCESS] Found {len(snaps)} snap(s):")
    for i, s in enumerate(snaps, 1):
        print(f"   {i}. Frame {s['frame']}  ({s['time']:.2f}s)  confidence={s['confidence']:.2f}")

    output_data = {
        "video_info": raw.get("video_info", {}),
        "snaps": snaps,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[SUCCESS] Saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
