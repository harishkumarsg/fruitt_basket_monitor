# app.py

from flask import Flask, request, jsonify
from datetime import datetime
import os
from scripts.yolo_detect_and_crop import detect_and_crop_fruits
from scripts.rot_classifier import classify_rot

app = Flask(_name_)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['file']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Detect and crop fruits using YOLO
        crops_dir = os.path.join("output", "crops", f"img_{timestamp}")
        os.makedirs(crops_dir, exist_ok=True)
        fruit_data, _ = detect_and_crop_fruits(filepath, crops_dir)

        results = []
        for fruit in fruit_data:
            rot_percent, _ = classify_rot(fruit['path'])
            results.append({
                "fruit": fruit['label'],
                "rot_percent": round(rot_percent, 2)
            })

        return jsonify({
            "timestamp": timestamp,
            "results": results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Main entry point for deployment platforms like Render/Railway
if _name_ == '_main_':
    import os
    port = int(os.environ.get("PORT", 5000))  # Bind to platform's dynamic port
    app.run(host='0.0.0.0', port=port)