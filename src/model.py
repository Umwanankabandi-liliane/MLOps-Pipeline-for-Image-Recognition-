# src/model.py
import tensorflow as tf
import json

def load_model():
    model = tf.keras.models.load_model("models/cifar10_model.h5")
    return model

def load_class_names():
    with open("models/class_names.json") as f:
        class_names = json.load(f)
    return class_names
