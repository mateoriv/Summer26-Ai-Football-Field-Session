# plotPlayersOnField.py
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Field constants (yards)
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
    return x_ft / 3.0, y_ft / 3.0

def plot_frame(ax, frame, radius_yd=0.6, color='red', show_label=False):
    """Plot all player detections in a single frame object."""
    detections = frame.get('detections', [])
    count = 0
    for det in detections:
        # Prefer 'field_coords' produced by homographyTransform.py
        fc = det.get('field_coords')
        if fc is None:
            # fallback: maybe homographyTransform didn't run — try bbox center (assume it's already field coords)
            bbox = det.get('bbox', {})
            cx = bbox.get('center_x')
            cy = bbox.get('center_y')
            if cx is None or cy is None:
                continue
            # assume these would be in feet if they were transformed; else this will be wrong — warn below
            x_ft, y_ft = float(cx), float(cy)
        else:
            x_ft = float(fc.get('x'))
            y_ft = float(fc.get('y'))

        # convert to yards for plotting (field drawing uses yards)
        x_yd, y_yd = feet_to_yards(x_ft, y_ft)

         # ⚠️ Correct for endzone offset
        x_yd += 10.0   # shift everything forward 10 yards

        circ = plt.Circle((x_yd, y_yd), radius_yd, color=color, alpha=0.85, zorder=5)
        ax.add_patch(circ)

        if show_label:
            lab = det.get('label') or det.get('class') or ""
            ax.text(x_yd + 0.5, y_yd + 0.2, str(lab), color='white', fontsize=6, zorder=6)

        count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Plot homography-transformed player detections on a football field")
    parser.add_argument('--input', '-i', required=True, help='Path to homographyTransform JSON (transformed detections)')
    parser.add_argument('--frame', '-f', type=int, default=None,
                        help='Frame index to plot (0-based). Default = last frame. Use -1 to overlay all frames.')
    parser.add_argument('--radius', type=float, default=0.6, help='Circle radius in yards (default 0.6)')
    parser.add_argument('--save', type=str, default=None, help='Optional path to save the plotted image (PNG)')
    parser.add_argument('--labels', action='store_true', help='Show small labels next to each circle')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    with open(args.input, 'r') as f:
        data = json.load(f)

    frames = data.get('frames', [])
    if not frames:
        raise ValueError("No frames found in JSON input.")

    # Choose frames
    if args.frame is None:
        frames_to_plot = [frames[-1]]       # default -> last frame
        title = f"Frame {len(frames)-1} (last)"
    elif args.frame == -1:
        frames_to_plot = frames             # overlay all
        title = "All frames (overlay)"
    else:
        if args.frame < 0 or args.frame >= len(frames):
            raise IndexError("frame index out of range")
        frames_to_plot = [frames[args.frame]]
        title = f"Frame {args.frame}"

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)

    total = 0
    if args.frame == -1:
        # overlay different colors
        cmap = plt.cm.get_cmap('tab20', len(frames_to_plot))
        for idx, fr in enumerate(frames_to_plot):
            c = cmap(idx % cmap.N)
            total += plot_frame(ax, fr, radius_yd=args.radius, color=c, show_label=args.labels)
    else:
        total = plot_frame(ax, frames_to_plot[0], radius_yd=args.radius, color='red', show_label=args.labels)

    ax.set_title(f"{title} — plotted {total} detections")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
        print("Saved image to", args.save)

    plt.show()

if __name__ == "__main__":
    main()