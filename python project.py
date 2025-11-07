# app.py

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
import pickle

# ----------------- Load Models -----------------
st.title("Breast Cancer Detection with Grad-CAM")

# CNN Model
CNN_MODEL_PATH = "breast_cancer_model_streamlit.h5"
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

# TML Models
TML_MODELS_PATH = "tml_models.pkl"  # Optional: save all TML models + scaler in pickle
with open(TML_MODELS_PATH, "rb") as f:
    tml_data = pickle.load(f)
tml_models = tml_data["models"]
scaler = tml_data["scaler"]

# Class mapping
class_mapping = {0: "benign", 1: "malignant", 2: "normal"}

# ----------------- Upload Image -----------------
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ----------------- TML Predictions -----------------
    img_flat = img_array.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    st.subheader("Traditional ML Predictions")
    for name, model in tml_models.items():
        pred = model.predict(img_scaled)[0]
        st.write(f"{name}: {class_mapping[pred]}")

    # ----------------- CNN Prediction -----------------
    img_input = np.expand_dims(img_array, axis=0)
    cnn_probs = cnn_model.predict(img_input)
    cnn_class = np.argmax(cnn_probs)

    st.subheader("CNN Prediction")
    st.write(f"Class: {class_mapping[cnn_class]}")
    st.write(f"Confidence: {cnn_probs[0][cnn_class]*100:.2f}%")

    # ----------------- Grad-CAM -----------------
    last_conv_layer_name = "conv2d_2"  # Update based on your CNN model

    # Build functional model for Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=cnn_model.input,
        outputs=[cnn_model.get_layer(last_conv_layer_name).output, cnn_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions_grad = grad_model(img_input)
        loss = predictions_grad[:, cnn_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap.numpy()

    # Superimpose heatmap
    heatmap_resized = cv2.resize(heatmap, (img.width, img.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap_color, 0.4, 0)

    st.subheader("Grad-CAM Visualization")
    st.image(superimposed_img, use_column_width=True)
