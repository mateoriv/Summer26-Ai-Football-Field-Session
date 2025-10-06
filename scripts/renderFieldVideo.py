#!/usr/bin/env python3
"""
Render Field Video Script
Creates a video showing players moving around on the digital football field
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import cv2

# Field constants (yards) - same as drawPlayers.py
FIELD_LENGTH = 120.0                # 120 yards (100 + 2 endzones)
FIELD_WIDTH = 160.0 / 3.0           # 160 ft -> yards (160/3 ~= 53.3333)
HASH_DIST_FT = 40.0                 # hash marks are 40 ft from sideline
HASH_NEAR_YD = HASH_DIST_FT / 3.0   # in yards (~13.3333)
HASH_TOP_YD = FIELD_WIDTH - HASH_NEAR_YD
HASH_LEN = 0.5

def draw_field(ax):
    """Draw a college football field to scale (yards)."""
    # Base rectangle
    field = patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, linewidth=2,
                              edgecolor='black', facecolor='green', zorder=0)
    ax.add_patch(field)

    # Yard lines every 5 yards (thinner) and every 10 (thicker)
    for x in range(10, int(FIELD_LENGTH), 5):
        lw = 2 if x % 10 == 0 else 1
        ax.plot([x, x], [0, FIELD_WIDTH], color='white', linewidth=lw, zorder=1)

    # Hash marks (every yard between 10 and 110 except multiples of 5)
    for x in range(11, 110):
        if x % 5 == 0:
            continue
        ax.plot([x, x], [HASH_NEAR_YD - HASH_LEN / 2, HASH_NEAR_YD + HASH_LEN / 2],
                color='white', linewidth=1, zorder=2)
        ax.plot([x, x], [HASH_TOP_YD - HASH_LEN / 2, HASH_TOP_YD + HASH_LEN / 2],
                color='white', linewidth=1, zorder=2)

    # End zones
    ez1 = patches.Rectangle((0, 0), 10, FIELD_WIDTH, linewidth=1,
                            edgecolor='white', facecolor='darkblue', alpha=0.6, zorder=1)
    ez2 = patches.Rectangle((110, 0), 10, FIELD_WIDTH, linewidth=1,
                            edgecolor='white', facecolor='darkred', alpha=0.6, zorder=1)
    ax.add_patch(ez1); ax.add_patch(ez2)

    # Yard numbers (every 10)
    for x in range(20, 110, 10):
        ax.text(x, 4, str(x-10), color='white', fontsize=8, ha='center', va='center', zorder=3)
        ax.text(x, FIELD_WIDTH-4, str(x-10), color='white', fontsize=8, ha='center', va='center', rotation=180, zorder=3)

    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')

def feet_to_yards(x_ft, y_ft):
    """Convert feet to yards for plotting"""
    return x_ft / 3.0, y_ft / 3.0

def get_track_color(track_id, max_tracks=20):
    """Get a consistent color for a track ID"""
    # Use a fixed color scheme to avoid flashing
    colors = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'cyan', 'magenta', 
              'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'teal', 'maroon',
              'gold', 'silver', 'coral', 'indigo']
    return colors[track_id % len(colors)]

def plot_frame(ax, frame, radius_yd=0.6, show_labels=False):
    """Plot players in a single frame."""
    # Handle both 'tracked' and 'detections' data structures
    detections = frame.get('detections', [])
    if not detections and 'tracked' in frame:
        detections = frame['tracked']
    
    # Clear previous player circles
    for patch in ax.patches:
        if hasattr(patch, '_is_player') and patch._is_player:
            patch.remove()
    
    for text in ax.texts:
        if hasattr(text, '_is_player_label') and text._is_player_label:
            text.remove()
    
    count = 0
    for i, detection in enumerate(detections):
        # Check if we have field coordinates (homography data) or bbox coordinates
        if 'field_coords' in detection:
            # Use field coordinates directly (already in feet)
            x_ft = detection['field_coords']['x']
            y_ft = detection['field_coords']['y']
        else:
            # Use bbox coordinates (pixel coordinates that need conversion)
            bbox = detection['bbox']
            x = bbox['center_x']
            y = bbox['y2']  # Use bottom of bbox
            x_ft, y_ft = x, y  # Assume already in feet if no field_coords
        
        # Convert to yards for plotting
        x_yd, y_yd = feet_to_yards(x_ft, y_ft)
        
        # Correct for endzone offset
        x_yd += 10.0   # shift everything forward 10 yards
        
        # Use a single color for all players to avoid flashing
        color = 'red'  # or 'blue', 'yellow', etc.
        
        # Create player circle
        circ = plt.Circle((x_yd, y_yd), radius_yd, color=color, alpha=0.8, zorder=5)
        circ._is_player = True  # Mark for removal
        ax.add_patch(circ)
        
        # Add detection ID label
        if show_labels:
            text = ax.text(x_yd + 0.5, y_yd + 0.2, str(i), color='white', fontsize=6, zorder=6)
            text._is_player_label = True  # Mark for removal
        
        count += 1
    
    return count

def create_field_video(input_json, output_video, fps=30, radius_yd=0.6, show_labels=False, 
                      frame_skip=1, max_frames=None):
    """
    Create a video showing players moving on the digital field
    
    Args:
        input_json: Path to tracked players JSON file
        output_video: Path to output video file
        fps: Frames per second for output video
        radius_yd: Radius of player circles in yards
        show_labels: Whether to show detection ID labels
        frame_skip: Process every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to process (None = all)
    """
    
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    if not frames:
        raise ValueError("No frames found in JSON input.")
    
    # Apply frame skipping and max frames
    if frame_skip > 1:
        frames = frames[::frame_skip]
    
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]
    
    print(f"Processing {len(frames)} frames for field video creation")
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    draw_field(ax)
    
    # Animation function
    def animate(frame_idx):
        frame = frames[frame_idx]
        player_count = plot_frame(ax, frame, radius_yd=radius_yd, 
                                show_labels=show_labels)
        
        # Update title with frame info
        timestamp = frame.get('timestamp', frame_idx / fps)
        ax.set_title(f"Frame {frame_idx} | Time: {timestamp:.2f}s | Tracked Players: {player_count}", 
                    fontsize=14, color='white', pad=20)
        
        return ax.patches + ax.texts
    
    # Create animation
    print("Creating field animation...")
    anim = FuncAnimation(fig, animate, frames=len(frames), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save as video
    print(f"Saving video to {output_video}...")
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Use matplotlib's animation writer
    Writer = plt.matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='AI Football Analysis'), bitrate=1800)
    
    anim.save(output_video, writer=writer, dpi=100)
    print(f"Field video saved successfully to {output_video}")
    
    return anim

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Create video of players moving on digital football field")
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to tracked players JSON file')
    parser.add_argument('--output', '-o', default='cache/videos/field_animation.mp4',
                       help='Path to output video file')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Frames per second for output video (default: 30)')
    parser.add_argument('--radius', type=float, default=0.6, 
                       help='Circle radius in yards (default: 0.6)')
    parser.add_argument('--labels', action='store_true', 
                       help='Show detection ID labels next to each circle')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    try:
        anim = create_field_video(
            input_json=args.input,
            output_video=args.output,
            fps=args.fps,
            radius_yd=args.radius,
            show_labels=args.labels,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
        print("Field video creation completed successfully!")
        
    except Exception as e:
        print(f"Error creating video: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
