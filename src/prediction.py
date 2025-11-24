# src/prediction.py
import numpy as np
from src.model import load_model, load_class_names
from src.preprocessing import preprocess_image

model = load_model()
class_names = load_class_names()

def predict_image(file):
    img = preprocess_image(file)
    preds = model.predict(img)[0]
    top_index = np.argmax(preds)
    return {
        "prediction": class_names[top_index],
        "confidence": float(preds[top_index])
    }
