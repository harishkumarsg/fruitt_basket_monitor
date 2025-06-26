# scripts/yolo_detect_and_crop.py

import cv2
import os
from ultralytics import YOLO
from datetime import datetime

def detect_and_crop_fruits(img_path, output_crop_dir):
    model = YOLO('yolov8n.pt')  # using pretrained model
    img = cv2.imread(img_path)

    results = model(img, save=False, verbose=False)[0]
    fruit_data = []

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]  # e.g., 'apple', 'banana'
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img[y1:y2, x1:x2]

        filename = f"{label}_{i}.jpg"
        filepath = os.path.join(output_crop_dir, filename)
        cv2.imwrite(filepath, crop)

        fruit_data.append({
            'name': label,              # ✅ changed 'label' ➝ 'name'
            'path': filepath,
            'bbox': (x1, y1, x2, y2)
        })

    # Save annotated image
    annotated_img = img.copy()
    for f in fruit_data:
        x1, y1, x2, y2 = f['bbox']
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, f['name'], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_img_path = f"output/images/detection_{timestamp}.jpg"
    cv2.imwrite(result_img_path, annotated_img)

    return fruit_data, result_img_path