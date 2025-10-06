# selectCorrespondences.py
import cv2
import json
import argparse

FIELD_LENGTH_FT = 360.0  # 120 yards * 3 ft
FIELD_WIDTH_FT = 160.0   # 53.3 yards * 3 ft

# Hash marks: 60 ft from each sideline
HASH_NEAR = 40.0
HASH_FAR = FIELD_WIDTH_FT - 40.0

correspondences = []
image_copy = None  # keep global copy for drawing

def get_field_coords(yardline, side, hashmark):
    """
    Convert yardline, side, and hash info into field coordinates in feet.
    """
    if side == "left":
        x = yardline * 3.0
    elif side == "right":
        x = FIELD_LENGTH_FT - yardline * 3.0
    else:
        raise ValueError("side must be 'left' or 'right'")
    
    if hashmark == "near":
        y = HASH_NEAR
    elif hashmark == "far":
        y = HASH_FAR
    else:
        raise ValueError("hash must be 'near' or 'far'")
    
    return [x, y]

def mouse_callback(event, x, y, flags, param):
    global image_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked image point: ({x}, {y})")

        # Prompt user for labels
        yardline = int(input("Enter yardline (10,20,30,40,50): "))
        side = input("Enter side ('left' or 'right'): ").strip().lower()
        hashmark = input("Enter hash ('near' or 'far'): ").strip().lower()

        field_point = get_field_coords(yardline, side, hashmark)

        entry = {
            "image_point": [float(x), float(y)],
            "field_point": field_point,
            "label": {
                "yardline": yardline,
                "side": side,
                "hash": hashmark
            }
        }
        correspondences.append(entry)

        # Draw circle + label on image copy
        label_text = f"{yardline}{'L' if side=='left' else 'R'}-{hashmark}"
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)  # red dot
        cv2.putText(image_copy, label_text, (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        print("Saved correspondence:", entry)

def main():
    global image_copy
    parser = argparse.ArgumentParser(description="Manual correspondence point selector")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="cache/correspondences.json",
                        help="Path to save JSON correspondences")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Error: Could not load image", args.image)
        return
    image_copy = img.copy()

    cv2.namedWindow("Select Correspondences")
    cv2.setMouseCallback("Select Correspondences", mouse_callback)

    print("Click at least 4 points. After each click, enter yardline/side/hash in terminal.")
    print("Press 'q' to finish and save.")

    while True:
        cv2.imshow("Select Correspondences", image_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Save correspondences to JSON
    output_data = {"correspondences": correspondences}
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Saved {len(correspondences)} correspondences to {args.output}")

if __name__ == "__main__":
    main()