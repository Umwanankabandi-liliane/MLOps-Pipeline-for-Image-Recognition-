# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from src.prediction import predict_image
import shutil
import os
import zipfile
from src.model import load_model, load_class_names
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI(
    title="CIFAR-10 MLOps API",
    description="Prediction + Retraining API using FastAPI",
    version="1.0"
)

# HEALTH CHECK
@app.get("/health")
def check_health():
    return {"status": "API is running ✔"}


# PREDICTION ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    result = predict_image(content)
    return result


# RETRAINING ENDPOINT (FULLY FIXED)
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    # Save ZIP temporarily
    zip_path = "temp_data.zip"
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    # Extract folder
    extract_dir = "new_data"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    # Unzip safely
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)

    # Load model + class names
    model = load_model()
    class_names = load_class_names()

    images = []
    labels = []

    # Loop through each class folder
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(extract_dir, class_name)

        # Skip missing folders
        if not os.path.exists(class_folder):
            continue

        files = os.listdir(class_folder)

        # Skip empty folders
        if len(files) == 0:
            continue

        for img_file in files:
            img_path = os.path.join(class_folder, img_file)

            # Read image safely
            img = cv2.imread(img_path)

            # Skip unreadable files
            if img is None:
                continue

            try:
                img = cv2.resize(img, (32, 32))
            except:
                # skip images cv2 cannot resize
                continue

            img = img.astype("float32") / 255.0

            images.append(img)
            labels.append(class_index)

    # Convert to numpy
    if len(images) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid images found in uploaded ZIP. Ensure correct folder structure."}
        )

    images = np.array(images)
    labels = np.array(labels)

    # Retrain the model
    model.fit(images, labels, epochs=3, verbose=1)

    # Save updated model
    model.save("models/cifar10_model.h5")

    return {"status": "Retrained and model updated successfully!"}
