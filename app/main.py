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

# RETRAINING ENDPOINT
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):

    # Save ZIP temporarily
    zip_path = "temp_data.zip"
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    # Extract
    extract_dir = "new_data/"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)

    # Load model + class names
    model = load_model()
    class_names = load_class_names()

    images = []
    labels = []

    # Load uploaded images for retraining
    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(extract_dir, class_name)
        if os.path.exists(class_folder):
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (32,32))
                images.append(img / 255.0)
                labels.append(class_index)

    images = np.array(images)
    labels = np.array(labels)

    # Retrain for 3 epochs
    model.fit(images, labels, epochs=3)

    # Save model
    model.save("models/cifar10_model.h5")

    return {"status": "Retrained and model updated successfully!"}
