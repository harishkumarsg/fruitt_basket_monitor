import cv2
import os
from datetime import datetime
from yolo_detect_and_crop import detect_and_crop_fruits
from rot_segmentation import estimate_rot_percent
from report_generator import generate_report

# 📁 Ensure all required folders exist
os.makedirs("output/images", exist_ok=True)
os.makedirs("classifier/cropped_fruits", exist_ok=True)
os.makedirs("output/reports", exist_ok=True)

crop_dir = "classifier/cropped_fruits"
cap = cv2.VideoCapture(0)  # 0 = default camera

print("📷 Press SPACE to capture and analyze, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to access camera")
        break

    cv2.imshow("Live Fruit Monitor", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to quit
        break

    elif key == 32:  # SPACE to capture and analyze
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"output/images/live_capture_{timestamp}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"📸 Captured: {img_path}")

        # 🔍 Run YOLO detection and crop
        try:
            fruit_data, annotated_path = detect_and_crop_fruits(img_path, crop_dir)
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            continue

        results = []
        for fruit in fruit_data:
            rot_percent = estimate_rot_percent(fruit['path'])
            fruit['rot_percent'] = rot_percent  # only rot percent added
            results.append(fruit)

        # 📝 Generate PDF report
        pdf_path = f"output/reports/report_{timestamp}.pdf"
        generate_report(results, annotated_path, pdf_path)

        print(f"✅ Report saved: {pdf_path}")

        # 🖼️ Show annotated image
        try:
            annotated_img = cv2.imread(annotated_path)
            if annotated_img is not None:
                cv2.imshow("📌 Detection Result", annotated_img)
                cv2.waitKey(0)
                cv2.destroyWindow("📌 Detection Result")
            else:
                print("⚠️ Couldn't open annotated result")
        except:
            print("⚠️ Error displaying annotated image")

cap.release()
cv2.destroyAllWindows()