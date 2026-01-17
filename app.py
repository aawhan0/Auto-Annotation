import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.title("🏗️ Construction Safety Audit Tool")
st.write("Upload an image to run the AI Auto-Annotation and Quality Audit.")

# Load Model
model = YOLO('models/yolov8n.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Layer 1: Quality Audit (Blur)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    st.subheader("Layer 1: Quality Audit")
    if variance < 100:
        st.error(f"🚩 Image Rejected: Too Blurry (Variance: {variance:.2f})")
    else:
        st.success(f"✅ Image Passed Quality Check (Variance: {variance:.2f})")

        # Layer 2: AI Audit
        results = model(img_cv)[0]
        
        st.subheader("Layer 2: AI Annotation Audit")
        
        # Display the prediction image
        res_plotted = results.plot()
        st.image(res_plotted, caption='AI Predictions', channels="BGR")

        # Decision Logic
        is_low_conf = any(box.conf < 0.85 for box in results.boxes)
        if is_low_conf or len(results.boxes) == 0:
            st.warning("⚠️ Result: Flagged for Human Review (Low Confidence)")
        else:
            st.success("✨ Result: Auto-Accepted (Gold Standard Label)")