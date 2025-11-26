import streamlit as st
import requests
from PIL import Image
import io
import numpy as np

API_URL = "https://mlops-pipeline-for-image-recognition.onrender.com"

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

        uploaded_image.seek(0)
        files = {"file": ("image.jpg", uploaded_image, "image/jpeg")}

        with st.spinner("Predicting..."):
            try:
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                if response.status_code == 200:
                    pred = response.json()
                    st.success(f"Prediction: **{pred['prediction']}**")
                    st.info(f"Confidence: **{pred['confidence']:.4f}**")
                else:
                    st.error(f"Prediction failed. Status: {response.status_code}")
                    st.write(response.text)
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {API_URL}")
                st.info("Please check if the API is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

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
        uploaded_zip.seek(0)
        files = {"file": ("retrain_data.zip", uploaded_zip, "application/zip")}

        with st.spinner("Retraining model... This may take several minutes."):
            try:
                response = requests.post(f"{API_URL}/retrain", files=files, timeout=300)
                if response.status_code == 200:
                    result = response.json()
                    st.success("🎉 Model retrained successfully!")
                    st.json(result)
                else:
                    st.error(f"Retraining failed. Status: {response.status_code}")
                    st.write(response.text)
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to API at {API_URL}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. Retraining is taking too long.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ----------------------- HEALTH TAB -----------------------
with tab3:
    st.header("📡 API Health Check")
    st.write(f"**API Endpoint:** `{API_URL}`")
    
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            if response.status_code == 200:
                st.success("✅ API is reachable!")
                st.json(response.json())
            else:
                st.warning(f"⚠️ API responded with status: {response.status_code}")
                st.write(response.text)
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API at {API_URL}")
            st.write("**Possible issues:**")
            st.write("- API is not deployed or not running")
            st.write("- Wrong API URL")
            st.write("- Network/firewall issues")
        except requests.exceptions.Timeout:
            st.error("⏱️ Connection timeout")
        except Exception as e:
            st.error(f"Error: {str(e)}")
