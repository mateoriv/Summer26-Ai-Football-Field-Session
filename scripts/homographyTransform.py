# script to perform homography transformation on data
# input: json file with correspondence points, detection JSON file
# output: json file with homography transformed data
# saved as homographyTransform.json in the cache folder

import cv2
import json
import os
import numpy as np

def homographyTransform(correspondence_file, detection_data):
    """
    Perform homography transformation on detection data
    
    Args:
        correspondence_file: Path to correspondence points JSON file
        detection_data: Detection data dictionary
    
    Returns:
        Transformed detection data dictionary
    """
    # Load correspondence points
    with open(correspondence_file, "r") as f:
        corr = json.load(f)

    pixel_points = []
    field_points = []
    for c in corr["correspondences"]:
        pixel_points.append(c["image_point"])
        field_points.append(c["field_point"])

    pixel_points = np.array(pixel_points, dtype=np.float32)
    field_points = np.array(field_points, dtype=np.float32)

    # Compute homography matrix
    H, _ = cv2.findHomography(pixel_points, field_points)

    # Transform detections
    for frame in detection_data.get("frames", []):
        for det in frame.get("detections", []):
            bbox = det["bbox"]

            # bottom-center of the bbox
            x = (bbox["x1"] + bbox["x2"]) / 2.0
            y = bbox["y2"]

            # apply homography
            pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pt, H)[0][0]

            det["field_coords"] = {
                "x": float(transformed[0]),
                "y": float(transformed[1])
            }

    return detection_data


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Homography Transform Module')
    parser.add_argument('--input', type=str, required=True, help='Path to input detection JSON file')
    parser.add_argument('--correspondence', type=str, required=True, 
                       help='Path to correspondence points JSON file')
    parser.add_argument('--output', type=str, default='cache/homography/homographyTransform.json', 
                       help='Path to output transformed JSON file')
    
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        detection_data = json.load(f)

    transformed = homographyTransform(args.correspondence, detection_data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(transformed, f, indent=4)

    print(f"Transformed detections saved to {args.output}")


if __name__ == "__main__":
    main()