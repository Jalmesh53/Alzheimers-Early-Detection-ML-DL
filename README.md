---
title: Alzheimer’s MRI Detection
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.31.1"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Alzheimer’s Early Detection using Multimodal AI

## 📌 Project Overview

This project presents a **comprehensive Alzheimer’s disease detection pipeline** using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques.  
Multiple models are developed and evaluated offline, while a **CNN-based MRI model** is deployed for real-time prediction in the application.

The goal is to compare different approaches and demonstrate how multimodal AI can assist in early-stage Alzheimer’s detection.

---

## 📂 Project Structure & Workflow

The complete experimental pipeline is implemented across the following notebooks (run in order):

1. **01_data_preprocessing.ipynb**
   - Image preprocessing (resizing, normalization)
   - Class mapping and dataset preparation

2. **02_ml_models.ipynb**
   - Traditional Machine Learning models (Random Forest)
   - Baseline performance comparison

3. **03_ann_model.ipynb**
   - Artificial Neural Network (ANN) for tabular/encoded features

4. **04_cnn_model.ipynb**
   - Convolutional Neural Network (CNN) for MRI image classification
   - Core model used for deployment

5. **05_fusion_model.ipynb**
   - Multimodal Fusion Model combining CNN image features and clinical data
   - Evaluated offline for performance comparison

---

## 🤖 Models Implemented

- **Machine Learning:** Random Forest (baseline)
- **Deep Learning:** ANN and CNN (TensorFlow/Keras)
- **Multimodal Fusion:** Functional API model combining image and clinical features

> **Note:**  
> ML, ANN, and Fusion models are evaluated **offline** due to their dependence on high-dimensional preprocessed clinical features.  
> The **CNN model** is deployed for real-time inference.

---

## 📊 Model Performance

Model evaluation metrics are computed on test data and visualized using **Power BI**.

- CNN achieved an overall accuracy of **~91.3%**
- Fusion model achieved the highest accuracy during offline evaluation

---

## 📊 Outputs Generated

All generated artifacts are stored in the `/Outputs` directory:

- **Trained Models:**
  - `alzheimer_cnn_model.h5`
  - `random_forest_model.pkl` (offline evaluation)

- **Evaluation Artifacts:**
  - Confusion matrices
  - Accuracy/Loss plots
  - `power_bi_results.csv` for dashboard visualization

---

## 🖥️ Deployed Application

The deployed Streamlit application focuses on:

- **MRI image upload**
- **CNN-based Alzheimer’s stage prediction**
- Clear stage interpretation and confidence display

Clinical ML/ANN/Fusion models are documented as part of experimentation and evaluation, not live inference.

---

## 🛠️ Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```
