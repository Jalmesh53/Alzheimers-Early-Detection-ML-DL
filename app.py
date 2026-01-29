import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Alzheimer’s MRI Classification Dashboard",
    layout="wide"
)

# ==================================================
# CONSTANTS (POWER BI – SOURCE OF TRUTH)
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
# LOAD MODEL (ONLY CNN – USED IN APP)
# ==================================================
MODEL_DIR = "Outputs"
cnn_model = tf.keras.models.load_model(f"{MODEL_DIR}/alzheimer_cnn_model.h5")

# ==================================================
# HEADER
# ==================================================
st.title("🧠 Alzheimer’s Early Detection System")
st.caption("Real-time MRI-based Alzheimer’s stage classification")

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
# ALZHEIMER'S STAGE SUMMARY
# ==================================================
st.subheader("🧠 Alzheimer’s Disease Stage Summary")

stage_summary = {
    "Stage": [
        "Non Demented",
        "Very Mild Demented",
        "Mild Demented",
        "Moderate Demented"
    ],
    "Severity": [
        "None",
        "Low",
        "Medium",
        "High"
    ],
    "Description": [
        "Healthy brain with no significant cognitive impairment",
        "Early cognitive decline; symptoms may not affect daily life",
        "Noticeable memory loss and difficulty with daily activities",
        "Advanced stage with severe cognitive and functional decline"
    ]
}

st.table(pd.DataFrame(stage_summary))

# ==================================================
# IMPORTANT NOTE
# ==================================================
st.info(
    "⚠️ Clinical ML, ANN, and Fusion models were evaluated offline due to "
    "high-dimensional preprocessing requirements. "
    "This live application focuses on MRI-based CNN prediction."
)

st.divider()

# ==================================================
# MRI IMAGE UPLOAD
# ==================================================
st.subheader("📂 Upload MRI Image")

img_file = st.file_uploader(
    "Upload MRI Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# MRI IMAGE → CNN
# ==================================================
if img_file:
    st.subheader("🖼️ MRI Image Preview")

    try:
        img = Image.open(img_file).convert("RGB")
        st.image(img, width=260)

        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        probs = cnn_model.predict(img)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100

        stage = STAGE_MAP[pred_class]
        stage_accuracy = MODEL_ACCURACY[stage]

        st.success(
            f"🧠 **CNN Diagnosis Result**\n\n"
            f"- **Predicted Stage:** {stage}\n"
            f"- **Prediction Confidence:** {confidence:.2f}%\n"
            f"- **Stage Accuracy:** {stage_accuracy}\n"
        )

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "Alzheimer’s Early Detection System | CNN Deployment | "
    "ML/ANN/Fusion Evaluated Offline (Power BI)"
)
