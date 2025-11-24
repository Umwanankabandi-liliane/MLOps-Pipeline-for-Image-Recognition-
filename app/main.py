# app/main.py

# ===================== IMPORTANT FIX FOR RETRAINING =====================
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()
# ========================================================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.prediction import predict_image
import shutil
import os
import zipfile
from src.model import load_model, load_class_names
import numpy as np
import cv2

app = FastAPI(
    title="CIFAR-10 MLOps API",
    description="Prediction + Retraining API using FastAPI",
    version="1.0"
)


# ====================== HEALTH CHECK ======================
@app.get("/health")
def check_health():
    return {"status": "API is running ✔"}


# ====================== PREDICTION ENDPOINT ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    result = predict_image(content)
    return result


# ====================== RETRAIN ENDPOINT ======================
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):

    # --- Save uploaded ZIP ---
    zip_path = "temp_data.zip"
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    # --- Extract ZIP ---
    extract_dir = "new_data"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    except:
        return JSONResponse(
            status_code=400,
            content={"error": "Uploaded file is not a valid ZIP archive."}
        )

    os.remove(zip_path)

    # --- Load model + class names ---
    model = load_model()
    class_names = load_class_names()

    images = []
    labels = []

    # --- Loop through CIFAR-10 class folders ---
    for class_index, class_name in enumerate(class_names):

        class_folder = os.path.join(extract_dir, class_name)

        if not os.path.exists(class_folder):
            continue

        files = os.listdir(class_folder)
        if len(files) == 0:
            continue

        for img_file in files:
            img_path = os.path.join(class_folder, img_file)

            img = cv2.imread(img_path)

            if img is None:
                try:
                    img = cv2.imdecode(
                        np.fromfile(img_path, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                except:
                    img = None

            if img is None:
                continue

            try:
                img = cv2.resize(img, (32, 32))
            except:
                continue

            img = img.astype("float32") / 255.0

            images.append(img)
            labels.append(class_index)

    # --- Validate loaded images ---
    if len(images) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid images found in ZIP. Ensure structure: class_name/image.jpg"}
        )

    images = np.array(images)
    labels = np.array(labels)

    # --- Retrain model ---
    model.fit(images, labels, epochs=3, verbose=1)

    # --- Save updated model ---
    model.save("models/cifar10_model.h5")

    return {"status": "Retrained and model updated successfully!"}
