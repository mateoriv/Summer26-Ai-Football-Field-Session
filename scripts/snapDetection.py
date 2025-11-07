#!/usr/bin/env python3
"""
Snap Detection Script (Improved)

Detects when the football is snapped by analyzing player movement patterns,
accounting for camera motion and pre-snap movement.

Input: player detection JSON file
Output: JSON file with detected snap frames
"""

import json
import os
import sys
import argparse
import numpy as np

def load_player_detections(detection_path):
    """Load player detection JSON file"""
    with open(detection_path, 'r') as f:
        return json.load(f)

def compute_velocity(detections):
    """
    Compute player velocities per frame, compensating for global (camera) motion.

    Returns:
        corrected_velocities: np.array of global average velocities per frame
        fps: frames per second
    """
    fps = detections.get('video_info', {}).get('fps', 30.0)
    frames = detections.get('frames', [])
    if not frames:
        return np.array([]), fps

    prev_centers = {}
    velocities = []
    prev_frame_centers = None

    for frame_idx, frame_data in enumerate(frames):
        frame_detections = frame_data.get('detections', [])
        curr_centers = []

        for det in frame_detections:
            bbox = det.get('bbox', {})
            if 'center_x' in bbox and 'center_y' in bbox:
                curr_centers.append((bbox['center_x'], bbox['center_y']))

        curr_centers = np.array(curr_centers)
        if len(curr_centers) == 0:
            velocities.append(0.0)
            prev_frame_centers = curr_centers
            continue

        # Estimate camera motion if we have previous frame
        if prev_frame_centers is not None and len(prev_frame_centers) > 0:
            # Match players roughly by nearest neighbors
            min_len = min(len(curr_centers), len(prev_frame_centers))
            deltas = curr_centers[:min_len] - prev_frame_centers[:min_len]

            # Estimate camera translation as median of all deltas
            cam_shift = np.median(deltas, axis=0)

            # Subtract camera motion
            motion_corrected = deltas - cam_shift

            # Prioritize horizontal motion (x-axis)
            vertical_weight = 0.3
            magnitudes = np.sqrt(motion_corrected[:, 0] ** 2 + (motion_corrected[:, 1] * vertical_weight) ** 2)

            # Remove outliers (single fast-moving players) before averaging
            if len(magnitudes) > 2:
                # Use IQR to identify outliers
                q75 = np.percentile(magnitudes, 75)
                q25 = np.percentile(magnitudes, 25)
                iqr = q75 - q25
                
                # Remove velocities that are outliers (more than 1.5 * IQR above Q75)
                # This filters out single fast-moving players while keeping scale similar
                outlier_threshold = q75 + 1.5 * iqr if iqr > 0 else np.inf
                filtered_magnitudes = magnitudes[magnitudes <= outlier_threshold]
                
                # Use mean of filtered values to maintain similar scale
                # This preserves the velocity magnitude while removing outliers
                if len(filtered_magnitudes) > 0:
                    avg_velocity = np.mean(filtered_magnitudes)
                else:
                    # Fallback to median if all values filtered
                    avg_velocity = np.median(magnitudes)
            else:
                avg_velocity = np.mean(magnitudes) if len(magnitudes) > 0 else 0.0
        else:
            avg_velocity = 0.0

        velocities.append(avg_velocity)
        prev_frame_centers = curr_centers

    return np.array(velocities), fps

def detect_snaps(velocities, fps,
                 calm_window=60, motion_window=45,
                 calm_threshold=None, motion_threshold=None,
                 gradient_threshold=None):
    """
    Detect snap moment as the transition from calm (low movement) to rapid motion.
    Designed to detect at the START of motion, not after it's fully developed.
    """
    """
    Detect snap moment as the transition from calm (low movement) to rapid motion.

    - Uses long averaging windows to smooth out pre-snap jitters.
    - Detects large gradients in motion over time.
    - Uses adaptive thresholds based on velocity distribution.
    """
    if len(velocities) == 0:
        return []

    # Smooth velocity over time (longer window = more stable)
    smooth_win = int(fps * 0.5)  # ~0.5s smoothing
    smoothed = np.convolve(velocities, np.ones(smooth_win) / smooth_win, mode='same')

    # Calculate adaptive thresholds based on velocity distribution
    if calm_threshold is None:
        # Calm threshold: 30th percentile (below average movement)
        calm_threshold = np.percentile(smoothed, 30)
    if motion_threshold is None:
        # Motion threshold: 70th percentile (above average movement)
        motion_threshold = np.percentile(smoothed, 70)
    if gradient_threshold is None:
        # Gradient threshold: 75th percentile of gradient magnitude
        grad_all = np.abs(np.gradient(smoothed))
        gradient_threshold = np.percentile(grad_all, 75)

    print(f"[INFO] Adaptive thresholds: calm={calm_threshold:.2f}, motion={motion_threshold:.2f}, gradient={gradient_threshold:.2f}")

    snap_frames = []
    total_frames = len(smoothed)

    # Gradient of smoothed velocity to detect rapid change
    grad = np.gradient(smoothed)

    snap_candidates = []
    
    # Detect snap at the START of velocity increase from calm state
    # Look for the first frame where velocity transitions from calm to rising
    # Use slightly shorter look-ahead to detect ~0.5s earlier
    look_ahead_frames = max(10, int(fps * 0.4))  # Look ~0.4s ahead to confirm motion
    
    for i in range(calm_window, total_frames - look_ahead_frames):
        # Check calm period before (long window)
        pre_calm = np.mean(smoothed[i - calm_window:i])
        
        # Current velocity at this frame
        current_vel = smoothed[i]
        
        # Look ahead to confirm motion is coming
        future_window = smoothed[i:i + look_ahead_frames]
        future_avg = np.mean(future_window)
        future_max = np.max(future_window)
        
        # Gradient at this point (how fast velocity is changing NOW)
        current_grad = grad[i]
        
        # Check if velocity is starting to rise (current > calm baseline)
        velocity_rising = current_vel > pre_calm * 1.05  # 5% increase from calm
        
        # Check if motion will develop (future confirms motion)
        # Slightly lower thresholds to detect earlier
        motion_coming = future_avg > motion_threshold * 0.3 or future_max > motion_threshold * 0.5

        # Conditions for snap (detect slightly earlier - ~0.5s before full motion):
        # 1. Calm before snap (below threshold * 1.5)
        calm_ok = pre_calm < calm_threshold * 1.5
        # 2. Velocity starting to rise NOW (not waiting for it to develop)
        rising_ok = velocity_rising or current_grad > 0
        # 3. Positive gradient (velocity increasing NOW, even if small) - slightly lower threshold
        grad_ok = current_grad > gradient_threshold * 0.13  # Slightly lower to catch ~0.5s earlier
        # 4. Motion confirmed in future (we know motion is coming)
        motion_ok = motion_coming
        
        # Require calm + (rising OR gradient) + motion confirmation
        if calm_ok and motion_ok and (rising_ok or grad_ok):
            # Calculate confidence score (higher is better)
            # Prioritize early detection: strong gradient + calm before + motion ahead
            calm_score = 1.0 - (pre_calm / calm_threshold) if calm_threshold > 0 else 0
            gradient_score = current_grad / gradient_threshold if gradient_threshold > 0 else 0
            jump_score = (future_avg - pre_calm) / calm_threshold if calm_threshold > 0 else 0
            future_score = (future_avg - motion_threshold * 0.4) / motion_threshold if motion_threshold > 0 else 0
            
            # Bonus for early detection (earlier frames get slightly higher score)
            early_bonus = 1.0 - (i / total_frames) if total_frames > 0 else 0
            
            # Strong gradient is key indicator of transition starting
            # Slightly favor earlier detection
            confidence = calm_score + gradient_score * 3 + jump_score + future_score + early_bonus * 0.4
            
            snap_candidates.append({
                'frame': int(i), 
                'time': i / fps,
                'confidence': confidence,
                'pre_calm': pre_calm,
                'current_vel': current_vel,
                'gradient': current_grad
            })

    # Filter out unlikely late detections (camera pans, etc.)
    snap_candidates = [s for s in snap_candidates if s['frame'] < total_frames * 0.6]

    # Return only the best snap (highest confidence)
    if len(snap_candidates) == 0:
        return []
    
    # Sort by confidence (highest first)
    snap_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Return only the best snap
    best_snap = snap_candidates[0]
    return [{'frame': best_snap['frame'], 'time': best_snap['time']}]

def main():
    parser = argparse.ArgumentParser(description="Snap Detection")
    parser.add_argument("--player-detections", type=str, required=True,
                        help="Path to player detection JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSON file")
    parser.add_argument("--calm-threshold", type=float, default=None,
                        help="Below this = calm (None = auto)")
    parser.add_argument("--motion-threshold", type=float, default=None,
                        help="Above this = active motion (None = auto)")

    args = parser.parse_args()

    print(f"[INFO] Loading player detections from: {args.player_detections}")
    detections = load_player_detections(args.player_detections)

    print("[INFO] Computing motion-compensated velocities...")
    velocities, fps = compute_velocity(detections)

    if len(velocities) == 0:
        print("[ERROR] No detections found — cannot compute snap.")
        sys.exit(1)

    print(f"[INFO] Velocity stats: mean={np.mean(velocities):.2f}, max={np.max(velocities):.2f}")

    print("[INFO] Detecting snap frames...")
    snap_frames = detect_snaps(
        velocities, fps,
        calm_threshold=args.calm_threshold,
        motion_threshold=args.motion_threshold
    )

    print(f"[SUCCESS] Found {len(snap_frames)} snap(s):")
    for i, snap in enumerate(snap_frames, 1):
        print(f"   {i}. Frame {snap['frame']} ({snap['time']:.2f}s)")

    # Save output
    output_data = {
        'video_info': detections.get('video_info', {}),
        'snaps': snap_frames,
        'detection_info': {
            'calm_threshold': args.calm_threshold,
            'motion_threshold': args.motion_threshold
        }
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"[SUCCESS] Results saved to {args.output}")
    print("[SUCCESS] Snap detection completed successfully!")

if __name__ == "__main__":
    main()
