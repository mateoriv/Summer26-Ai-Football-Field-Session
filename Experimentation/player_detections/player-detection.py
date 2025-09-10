from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os


def fit_line(points, img_shape):
    """
    Fit a line x = m*y + b through given points (least squares).
    Returns two endpoints for drawing.
    """
    points = np.array(points)
    ys = points[:,1]
    xs = points[:,0]

    # Fit linear regression for x as function of y
    m, b = np.polyfit(ys, xs, 1)

    # Endpoints: top (y=0) and bottom (y=img_height)
    y1, y2 = 0, img_shape[0]
    x1 = int(m*y1 + b)
    x2 = int(m*y2 + b)

    return (x1, y1), (x2, y2)



model = YOLO("best.pt")
for file in os.listdir("inputs"):
    full_rel_path = f"inputs/{file}"
    img = cv2.imread(full_rel_path)
    # Run prediction
    results = model.predict(img)
    # Given from model
    classes = {0: 'defense', 1: 'oline', 2: 'qb', 3: 'ref',
            4: 'running_back', 5: 'tight_end', 6: 'wide_receiver'}

    boxes = []
    class_ids = []
    scores = []

    for result in results:
        temp_boxes = result.boxes.xyxy.cpu().numpy()  
        temp_class_ids = result.boxes.cls.cpu().numpy()
        temp_scores = result.boxes.conf.cpu().numpy()

        for (box, cls_id, score) in zip(temp_boxes, temp_class_ids, temp_scores):
            # Skip refs
            if int(cls_id) == 3:
                continue  
            x1, y1, x2, y2 = map(int, box)
            label = classes[int(cls_id)]
            conf = f"{score:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text: label + confidence
            cv2.putText(img, f"{label} {conf}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            boxes.append(box)
            class_ids.append(cls_id)
            scores.append(score)

    # Save image with that has labeled positions
    os.makedirs("outputs/positions", exist_ok=True)
    cv2.imwrite(f"outputs/positions/{file}", img)


    # Fresh image:
    img = cv2.imread(full_rel_path)
    
    # Try to identify teams?
    features = []
    player_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        player_boxes.append((x1, y1, x2, y2))

        # Crop to player region
        player = img[y1:y2, x1:x2]

        # Compute average color
        mean_color = cv2.mean(player)[:3]

        # Player center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Feature vector: position + color
        features.append([cx, cy, mean_color[0], mean_color[1], mean_color[2]])


    features = np.array(features)

    # Try to identify 2 teams
    kmeans = KMeans(n_clusters=2, random_state=42).fit(features)
    labels = kmeans.labels_


    # Output images
    for (x1, y1, x2, y2), label in zip(player_boxes, labels):
        color = (0, 0, 255) if label == 0 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
    os.makedirs("outputs/clusters", exist_ok=True)
    cv2.imwrite(f"outputs/clusters/{file}", img)


    # # Try to draw line of scrimmage
    # defense_centers = []
    # offense_centers = []
    # img = cv2.imread(full_rel_path)

    # for (x1, y1, x2, y2), cls_id in zip(player_boxes, class_ids):
    #     cx = (x1 + x2) // 2
    #     cy = (y1 + y2) // 2
    #     if cls_id == 0:  # defense
    #         defense_centers.append((cx, cy))
    #     else:  # offense
    #         offense_centers.append((cx, cy))

    # if defense_centers and offense_centers:
    #     if len(defense_centers) >= 2:
    #         pt1, pt2 = fit_line(defense_centers, img.shape)
    #         cv2.line(img, pt1, pt2, (0, 255, 255), 2)
    #     if len(offense_centers) >= 2:
    #         pt1, pt2 = fit_line(offense_centers, img.shape)
    #         cv2.line(img, pt1, pt2, (0, 0, 255), 2)

    #     os.makedirs("outputs/line", exist_ok=True)
    #     cv2.imwrite(f"outputs/line/{file}", img)
