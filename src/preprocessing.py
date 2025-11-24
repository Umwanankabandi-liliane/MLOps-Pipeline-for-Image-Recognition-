# src/preprocessing.py
import numpy as np
import cv2

def preprocess_image(file):
    """Convert uploaded image into model-ready tensor."""
    img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
