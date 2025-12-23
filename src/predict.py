import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import sys

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/binary_mobilenet_model.h5"
IMG_SIZE = (224, 224)

# 🔴 CHANGE THESE LABELS AS PER YOUR DATASET
# Example: ["Cat", "Dog"] OR ["Normal", "Pneumonia"]
CLASS_NAMES = ["Class_0", "Class_1"]

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load & preprocess image (same as training)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Decode prediction
    if prediction > 0.5:
        label = CLASS_NAMES[1]
        confidence = prediction
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - prediction

    return label, confidence

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Usage: python src/predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    print("📸 Reading image from:", os.path.abspath(img_path))

    label, confidence = predict_image(img_path)

    print("\n✅ PREDICTION RESULT")
    print(f"🧠 Predicted Class : {label}")
    print(f"📊 Confidence      : {confidence:.2f}")
