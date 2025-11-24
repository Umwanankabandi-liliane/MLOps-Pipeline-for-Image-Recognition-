import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

# ======================== CONFIG ========================
API_URL = "https://mlops-pipeline-for-image-recognition.onrender.com"

st.set_page_config(
    page_title="CIFAR-10 MLOps Dashboard",
    layout="wide",
    page_icon="🧠"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
    <style>
        .big-title {
            font-size: 42px !important;
            font-weight: 900 !important;
            color: #2C3E50;
            margin-bottom: 10px;
        }
        .sub-text {
            font-size: 18px;
            color: #5D6D7E;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .section-header {
            font-size: 26px;
            font-weight: 700;
            color: #2C3E50;
            margin-bottom: 15px;
        }
        .success-box {
            background-color: #D4EFDF;
            padding: 15px;
            border-radius: 10px;
        }
        .error-box {
            background-color: #FADBD8;
            padding: 15px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ======================== HEADER ========================
st.markdown("<div class='big-title'>🖼️ CIFAR-10 MLOps Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Deploy • Predict • Retrain • Monitor</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict Image", "🔁 Retrain Model", "📡 API Health"])

# =========================================================
#                      PREDICTION TAB
# =========================================================
with tab1:
    st.markdown("<div class='section-header'>🔮 Predict Image</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Choose any image to classify"
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", width=250)

            img_bytes = uploaded_image.read()
            files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

            with st.spinner("Running prediction..."):
                response = requests.post(f"{API_URL}/predict", files=files)

            if response.status_code == 200:
                pred = response.json()
                st.success(f"🎉 **Prediction:** {pred['prediction'].upper()}")

                st.progress(float(pred["confidence"]))
                st.write(f"Confidence: **{pred['confidence']:.3f}**")

            else:
                st.error("❌ Unable to get prediction. Check API.")

# =========================================================
#                      RETRAIN TAB
# =========================================================
with tab2:
    st.markdown("<div class='section-header'>🔁 Retrain Model</div>", unsafe_allow_html=True)
    st.write("Upload a ZIP file containing CIFAR-10 structured folders:")

    st.code("""
data/
 ├─ airplane/
 ├─ automobile/
 ├─ bird/
 ├─ cat/
 ├─ deer/
 ├─ dog/
 ├─ frog/
 ├─ horse/
 ├─ ship/
 └─ truck/
    """)

    uploaded_zip = st.file_uploader(
        "Upload dataset ZIP",
        type="zip",
        help="Make sure it contains folders named after CIFAR-10 classes"
    )

    if uploaded_zip:
        zip_bytes = uploaded_zip.read()
        files = {"file": ("dataset.zip", zip_bytes, "application/zip")}

        with st.spinner("Retraining model... This may take up to 2 minutes."):
            response = requests.post(f"{API_URL}/retrain", files=files)

        if response.status_code == 200:
            st.markdown("<div class='success-box'>🎉 Model retrained successfully!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='error-box'>❌ Retraining failed. Check ZIP structure.</div>", unsafe_allow_html=True)

# =========================================================
#                      HEALTH CHECK TAB
# =========================================================
with tab3:
    st.markdown("<div class='section-header'>📡 API Health Check</div>", unsafe_allow_html=True)

    if st.button("Check API Status"):
        try:
            response = requests.get(f"{API_URL}/health")

            if response.status_code == 200:
                st.success(f"🟢 API is Live → {response.json()}")
            else:
                st.error("🔴 API is unreachable.")
        except:
            st.error("❌ API is not reachable. Service may be down.")
