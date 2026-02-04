import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2

from src.brain_mri_filter import brain_mri_quick_check

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Alzheimer’s MRI Classification Dashboard",
    layout="wide"
)

# ==================================================
# CONSTANTS
# ==================================================
OVERALL_ACCURACY = "91.30%"

STAGE_MAP = {
    0: "Non Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Moderate Demented"
}

MODEL_ACCURACY = {
    "Non Demented": "89.37%",
    "Very Mild Demented": "84.39%",
    "Mild Demented": "96.20%",
    "Moderate Demented": "94.72%"
}

# ==================================================
# LOAD MODEL
# ==================================================
MODEL_DIR = "Outputs"
cnn_model = tf.keras.models.load_model(
    f"{MODEL_DIR}/alzheimer_cnn_model.keras",
    compile=False,
    safe_mode=False
)

# ==================================================
# HEADER
# ==================================================
st.title("🧠 Alzheimer’s Early Detection System")
st.caption("AI-assisted MRI-based Alzheimer’s stage classification")

# ==================================================
# ACCURACY SUMMARY
# ==================================================
st.subheader("📊 Model Accuracy (Offline Evaluation)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("ML Accuracy", "88.6%")
col2.metric("ANN Accuracy", "90.1%")
col3.metric("CNN Accuracy", "91.3%")
col4.metric("Fusion Accuracy", "93.4%")

st.divider()

# ==================================================
# STAGE SUMMARY
# ==================================================
st.subheader("🧠 Alzheimer’s Disease Stage Summary")

stage_summary = {
    "Stage": list(STAGE_MAP.values()),
    "Severity": ["None", "Low", "Medium", "High"],
    "Description": [
        "Healthy brain with no significant cognitive impairment",
        "Early cognitive decline, symptoms may not affect daily life",
        "Noticeable memory loss and difficulty with daily activities",
        "Advanced stage with severe cognitive and functional decline"
    ]
}

st.table(pd.DataFrame(stage_summary))

st.info(
    "⚠️ This system supports only axial brain MRI images. "
    "Other medical images are automatically rejected."
)

st.divider()

# ==================================================
# IMAGE UPLOAD
# ==================================================
st.subheader("📂 Upload MRI Image")

img_file = st.file_uploader(
    "Upload MRI Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# MRI → VALIDATION → CNN PREDICTION
# ==================================================
if img_file is not None:
    st.subheader("🖼️ MRI Image Preview")

    try:
        # Load image using PIL
        pil_img = Image.open(img_file).convert("RGB")
        st.image(pil_img, width=260)

        # Convert to NumPy (OpenCV format) for validation
        img_np = np.array(pil_img)

        # -------------------------------
        # 🛑 Brain MRI validation
        # -------------------------------
        if not brain_mri_quick_check(img_np):
            st.error("❌ Invalid input detected")
            st.warning("Please upload a valid **BRAIN MRI** image.")
            st.stop()

        # -------------------------------
        # Preprocess image for CNN
        # -------------------------------
        img_resized = pil_img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------
        # CNN Prediction
        # -------------------------------
        probs = cnn_model.predict(img_array)[0]
        pred_class = int(np.argmax(probs))

        stage = STAGE_MAP[pred_class]
        stage_accuracy = MODEL_ACCURACY[stage]

        # -------------------------------
        # Confidence-based rejection
        # -------------------------------
        max_prob = float(np.max(probs))

        if max_prob < 0.85:
            st.error("⚠️ Prediction rejected due to low confidence")
            st.warning(
                "The uploaded image may not be a valid brain MRI "
                "or is outside the model’s training distribution."
            )
            st.stop()

        # -------------------------------
        # Model certainty label
        # -------------------------------
        if max_prob >= 0.75:
            certainty = "High"
        elif max_prob >= 0.55:
            certainty = "Medium"
        else:
            certainty = "Low"

        # -------------------------------
        # Display result
        # -------------------------------
        st.success(
            f"🧠 **CNN Diagnosis Result**\n\n"
            f"- **Predicted Stage:** {stage}\n"
            f"- **Model Certainty:** {certainty}\n"
            f"- **Stage Accuracy:** {stage_accuracy}\n"
        )

        # -------------------------------
        # Probability table
        # -------------------------------
        st.subheader("📈 Class Probability Distribution")
        prob_df = pd.DataFrame({
            "Stage": list(STAGE_MAP.values()),
            "Probability": [f"{p:.2f}" for p in probs]
        })
        st.table(prob_df)

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "Alzheimer’s Early Detection System | CNN Deployment | "
    "Predictions are probabilistic and not a medical diagnosis"
)
