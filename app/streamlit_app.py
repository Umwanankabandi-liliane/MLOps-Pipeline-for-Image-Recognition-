import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

API_URL = "https://your-render-api-url.onrender.com"  # <-- Update after deployment

st.set_page_config(page_title="CIFAR-10 MLOps UI", layout="wide")

st.title("🖼️ CIFAR-10 Image Classifier – MLOps UI")
st.write("Upload an image to get a prediction, or upload a ZIP to retrain the model.")

# ----------------------- PREDICTION TAB -----------------------
tab1, tab2, tab3 = st.tabs(["🔮 Predict Image", "🔁 Retrain Model", "📡 API Health"])

with tab1:
    st.header("🔮 Predict Image")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width=200)

        img_bytes = uploaded_image.read()
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

        with st.spinner("Predicting..."):
            response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            pred = response.json()
            st.success(f"Prediction: **{pred['prediction']}**")
            st.info(f"Confidence: **{pred['confidence']:.4f}**")
        else:
            st.error("Prediction failed. Check API.")

# ----------------------- RETRAIN TAB -----------------------
with tab2:
    st.header("🔁 Retrain Model")
    st.write("Upload a ZIP file containing folders named after CIFAR-10 classes:")

    st.code("""
airplane/
automobile/
bird/
cat/
deer/
dog/
frog/
horse/
ship/
truck/
""")

    uploaded_zip = st.file_uploader("Upload ZIP", type=["zip"])

    if uploaded_zip is not None:
        zip_bytes = uploaded_zip.read()
        files = {"file": ("new_data.zip", zip_bytes, "application/zip")}

        with st.spinner("Retraining model..."):
            response = requests.post(f"{API_URL}/retrain", files=files)

        if response.status_code == 200:
            st.success("🎉 Model retrained successfully!")
        else:
            st.error("Retraining failed.")

# ----------------------- HEALTH TAB -----------------------
with tab3:
    st.header("📡 API Health Check")

    if st.button("Check API Status"):
        try:
            response = requests.get(f"{API_URL}/health")
            st.success(response.json())
        except:
            st.error("API is not reachable.")
