import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# 1. Load the model and scaler (Ensure filenames match exactly)
@st.cache_resource
def load_assets():
    # Adding 'Outputs/' to the path so the app can find the files
    model = tf.keras.models.load_model('final_fusion_model.keras')
    scaler = joblib.load('clinical_scaler.pkl')
    return model, scaler

model, scaler = load_assets()
labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# 2. UI Header
st.title("Alzheimer's Diagnostic Assistant")
st.markdown("### Multimodal Fusion Model | Accuracy: 91.30%")

# 3. Inputs: Image + Clinical Data
col1, col2 = st.columns(2)

with col1:
    st.header("Clinical Info")
    age = st.number_input("Age", 50, 100, 75)
    mmse = st.slider("MMSE Score", 0, 30, 24)
    etiv = st.number_input("eTIV", 1000, 2000, 1500)

with col2:
    st.header("MRI Scan")
    file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

# 4. Prediction Logic
if file and st.button("Predict"):
    # Process Image
    img = Image.open(file).convert('RGB').resize((128, 128))
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Process Clinical
    clinical_arr = scaler.transform([[age, mmse, etiv]])

    # Final Prediction
    pred = model.predict([img_arr, clinical_arr])
    res = labels[np.argmax(pred)]
    st.success(f"Result: **{res}** (Confidence: {np.max(pred)*100:.2f}%)")