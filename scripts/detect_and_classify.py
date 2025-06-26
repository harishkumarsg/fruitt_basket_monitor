from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model

# --- Load Models ---
yolo_model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 model

# Path to rot classifier model
script_dir = os.path.dirname(__file__)
rot_model_path = os.path.join(script_dir, "../fruit_classifier_model.keras")
rot_model = load_model(rot_model_path)

# Labels
labels = ["Fresh", "Rotten"]

# --- Load Image ---
image_path = os.path.join(script_dir, "fruitbasket1.jpg")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

# --- YOLO Detection ---
results = yolo_model.predict(source=image, show=False, save=False)

# --- Output Dir ---
output_dir = os.path.join(script_dir, "../output/images")
os.makedirs(output_dir, exist_ok=True)

# Annotate image
annotated_img = results[0].plot()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Counters
total = 0
rotten = 0

# --- Process Detected Fruits ---
for r in results:
    for i, box in enumerate(r.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        crop = r.orig_img[y1:y2, x1:x2]

        # Save cropped image
        crop_filename = os.path.join(output_dir, f"fruit_{timestamp}_{i}.jpg")
        cv2.imwrite(crop_filename, crop)

        # Resize to 138x138 → Flatten to (1, 57600)
        resized = cv2.resize(crop, (138, 138))
        normalized = resized.astype("float32") / 255.0
        flattened = normalized.reshape(1, -1)

        # Predict rot status
        pred = rot_model.predict(flattened)[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]

        label_text = f"{labels[predicted_class]} ({confidence*100:.1f}%)"
        print(f"[INFO] Fruit {i}: {label_text} saved as {crop_filename}")

        # Draw box and label
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        total += 1
        if predicted_class == 1:
            rotten += 1

# --- Show Annotated Image ---
cv2.imshow("Detected & Classified Fruits", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Summary ---
if total > 0:
    rot_percent = (rotten / total) * 100
    print(f"\n[SUMMARY] Total Fruits: {total} | Rotten: {rotten} | Rot %: {rot_percent:.2f}%")
else:
    print("[SUMMARY] No fruits detected.")