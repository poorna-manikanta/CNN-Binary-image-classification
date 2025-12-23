import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/binary_mobilenet_model.h5"
IMG_SIZE = (224, 224)

# 🔴 CHANGE THESE LABELS AS PER YOUR DATASET
# Example: ["Cat", "Dog"] or ["Normal", "Pneumonia"]
CLASS_NAMES = ["Class_0", "Class_1"]

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Image Classification App", layout="centered")

st.title("🧠 Image Classification App")
st.write("Upload an image to get prediction using **MobileNetV2**")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = CLASS_NAMES[1]
        confidence = prediction
    else:
        label = CLASS_NAMES[0]
        confidence = 1 - prediction

    # Output
    st.success(f"🧠 **Prediction:** {label}")
    st.info(f"📊 **Confidence:** {confidence:.2f}")
