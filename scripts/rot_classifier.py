# scripts/rot_classifier.py

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained classifier model
model = tf.keras.models.load_model('fruit_classifier_model.keras')

# Print the model's expected input shape
expected_shape = model.input_shape[1:3]  # e.g., (128, 128)
print(f"📐 Expected input shape for classifier: {expected_shape}")

# Full label list (match your 16-class model)
labels = [
    'FreshApple', 'FreshBanana', 'FreshGrape', 'FreshGuava', 'FreshJujube',
    'FreshOrange', 'FreshStrawberry', 'FreshPomegranate',
    'RottenApple', 'RottenBanana', 'RottenGrape', 'RottenGuava', 'RottenJujube',
    'RottenOrange', 'RottenStrawberry', 'RottenPomegranate'
]

def classify_rot(img_path):
    try:
        # Load and resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(expected_shape)

        # Preprocess image
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        pred = model.predict(img_array)[0]
        class_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))

        print(f"🔎 Prediction vector: {pred}")
        print(f"🔎 Predicted index: {class_idx}")

        if class_idx < len(labels):
            full_label = labels[class_idx]  # e.g., "FreshApple"
            rot_status = 'fresh' if 'Fresh' in full_label else 'rotten'
            return rot_status, confidence
        else:
            return "unknown", confidence

    except Exception as e:
        print(f"❌ Error in classify_rot: {e}")
        return "error", 0.0