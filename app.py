import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import os
from PIL import Image

# 1. Load the model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('final_fusion_model.keras')
    scaler = joblib.load('clinical_scaler.pkl')
    return model, scaler

model, scaler = load_assets()
labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# 2. UI Header
st.title("🧠 Alzheimer's Diagnostic Assistant")
st.markdown("---")

# 3. Inputs: Image + Clinical Data
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Clinical Data")
    age = st.number_input("Age", 50, 100, 75)
    mmse = st.slider("MMSE Score (Cognitive Test)", 0, 30, 24)
    etiv = st.number_input("eTIV (Total Intracranial Volume)", 1000, 2000, 1500)
    
    # Check if Scaler needs more features than the 3 we have
    expected_features = scaler.n_features_in_
    if expected_features > 3:
        st.info(f"💡 Scaler expects {expected_features} features. Filling missing ones with 0.")

with col2:
    st.subheader("🖼️ MRI Scan")
    file = st.file_uploader("Upload Axial MRI Scan", type=["jpg", "png", "jpeg"])
    if file:
        st.image(file, caption="Uploaded MRI", use_container_width=True)

# 4. Prediction Logic
if st.button("Generate Diagnostic Prediction", type="primary"):
    if not file:
        st.error("Please upload an MRI image first.")
    else:
        try:
            # A. Process Image
            img = Image.open(file).convert('RGB').resize((128, 128))
            img_arr = np.array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)

            # B. Process Clinical Data (Dynamic Feature Padding)
            # We provide the 3 we have, and fill the rest with 0.0 to avoid ValueError
            clinical_input = [age, mmse, etiv]
            while len(clinical_input) < expected_features:
                clinical_input.append(0.0)
            
            # Scaler expects a 2D array: [[f1, f2, f3...]]
            clinical_arr = scaler.transform([clinical_input])

            # C. Final Prediction
            pred = model.predict([img_arr, clinical_arr], verbose=0)
            idx = np.argmax(pred)
            res = labels[idx]
            conf = np.max(pred) * 100

            # D. Display Results
            st.markdown("---")
            if "Non Demented" in res:
                st.balloons()
                st.success(f"### Result: {res}")
            else:
                st.warning(f"### Result: {res}")
            
            st.metric("Confidence Level", f"{conf:.2f}%")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.write("Ensure your scaler matches your input features.")
