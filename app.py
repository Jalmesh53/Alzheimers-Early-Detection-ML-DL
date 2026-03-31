import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2

# 👉 Brain MRI validation
from src.brain_mri_filter import brain_mri_quick_check

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

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(
        f"{MODEL_DIR}/alzheimer_cnn_model.h5",
        compile=False
    )

cnn_model = load_cnn_model()

# ==================================================
# CLINICAL DECISION SUPPORT LOGIC
# ==================================================
def get_clinical_advice(patient_age, stage_label):
    """Calculates the seriousness based on age and AI prediction."""
    stage = stage_label.lower()
    
    if "non" in stage:
        return "🟢 **Seriousness: LOW**\n\n*Analysis:* Patient shows healthy cognitive structures. Continue routine annual screenings."
        
    elif "very mild" in stage:
        if patient_age < 65:
            return "🟠 **Seriousness: HIGH ALERT (Early-Onset Risk)**\n\n*Analysis:* Mild tissue loss is highly unusual for a patient under 65. Immediate neurological evaluation required to rule out aggressive early-onset factors."
        else:
            return "🟡 **Seriousness: MODERATE**\n\n*Analysis:* Typical signs of Mild Cognitive Impairment (MCI) for this age. Monitor closely and consider lifestyle/dietary interventions."
            
    elif "mild" in stage:
        if patient_age < 65:
            return "🔴 **Seriousness: SEVERE (Early-Onset Risk)**\n\n*Analysis:* Significant early-onset progression suspected. Immediate medical intervention and treatment plan needed."
        else:
            return "🟠 **Seriousness: HIGH**\n\n*Analysis:* Clinical Alzheimer's indicators are present. Begin symptom management and medication evaluation."
            
    elif "moderate" in stage:
        return "🔴 **Seriousness: CRITICAL**\n\n*Analysis:* Advanced stage neurodegeneration detected. Comprehensive care plan, specialist intervention, and daily support required."
        
    return "⚠️ Assessment unavailable."

# ==================================================
# HEADER
# ==================================================
st.title("🧠 Alzheimer’s Early Detection System")
st.caption("Real-time MRI-based Alzheimer’s stage classification with Clinical Support")

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

st.info(
    "⚠️ This system supports only **axial brain MRI images**. "
    "Other medical images are automatically rejected."
)

st.info(
    "⚠️ Clinical ML, ANN, and Fusion models were evaluated offline due to "
    "high-dimensional preprocessing requirements. "
    "This live application focuses on MRI-based CNN prediction."
)

st.divider()

# ==================================================
# PATIENT INFO & MRI UPLOAD
# ==================================================
st.subheader("👨‍⚕️ Patient Details & MRI Upload")

# Split into two columns for Age and Image Upload
col_age, col_upload = st.columns(2)

with col_age:
    patient_age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=65, step=1)

with col_upload:
    img_file = st.file_uploader(
        "Upload MRI Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

# ==================================================
# MRI → VALIDATION → CNN
# ==================================================
if img_file is not None:
    st.subheader("🖼️ MRI Image Preview")

    try:
        # Load image
        pil_img = Image.open(img_file).convert("RGB")
        st.image(pil_img, width=260)

        # Convert to NumPy for validation
        img_np = np.array(pil_img)

        # -------------------------------
        # 🛑 Brain MRI validation
        # -------------------------------
        if not brain_mri_quick_check(img_np):
            st.error("❌ Invalid input detected")
            st.warning("Please upload a valid **BRAIN MRI** image.")
            st.stop()

        # -------------------------------
        # Preprocess for CNN
        # -------------------------------
        img_resized = pil_img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------
        # CNN Prediction
        # -------------------------------
        probs = cnn_model.predict(img_array, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        max_prob = float(np.max(probs))

        # -------------------------------
        # Confidence-based rejection
        # -------------------------------
        if max_prob < 0.85:
            st.error("⚠️ Prediction rejected due to low confidence")
            st.warning(
                "The uploaded image may not be a valid brain MRI "
                "or is outside the model’s training distribution."
            )
            st.stop()

        stage = STAGE_MAP[pred_class]
        stage_accuracy = MODEL_ACCURACY[stage]
        confidence = max_prob * 100

        # -------------------------------
        # Display Result & Clinical Advice
        # -------------------------------
        st.success(
            f"🧠 **CNN Diagnosis Result**\n\n"
            f"- **Predicted Stage:** {stage}\n"
            f"- **Prediction Confidence:** {confidence:.2f}%\n"
            f"- **Stage Accuracy:** {stage_accuracy}\n"
        )
        
        # Add the Decision Support output
        st.subheader("👨‍⚕️ Clinical Decision Support")
        advice = get_clinical_advice(patient_age, stage)
        st.info(advice)

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