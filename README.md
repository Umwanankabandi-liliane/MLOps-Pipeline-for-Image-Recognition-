# MLOps Pipeline for Image Recognition

## 1. Project Overview

This project implements a complete MLOps pipeline for an image classification model trained on the CIFAR-10 dataset.
The solution demonstrates the full lifecycle of a machine learning system, including preprocessing, model training, deployment, monitoring, retraining, and load testing.

The final system consists of:

* A TensorFlow/Keras CNN model trained on CIFAR-10.
* A FastAPI backend providing prediction and retraining endpoints.
* A Streamlit dashboard for interacting with the model.
* A retraining pipeline allowing model updates using new data.
* A Locust load-test to measure the API’s performance under stress.
* A fully documented Jupyter notebook demonstrating all preprocessing and training steps.




  ## 2. Live Application URLs

### FastAPI Backend

Base URL:
https://mlops-pipeline-for-image-recognition.onrender.com



Available endpoints:

* `/predict` — Image prediction
* `/retrain` — Model retraining
* `/health` — Health check

### Streamlit Dashboard

User interface for prediction and retraining:
https://kvbxjbosjedmcytvxsf5tu.streamlit.app/


## 3. How to Set Up Locally

### Install dependencies


pip install -r requirements.txt


### Run the API


uvicorn app.main:app --reload
```

### Run the Streamlit UI


streamlit run app/streamlit_app.py



## 4. API Endpoints

### 4.1 Prediction Endpoint


POST /predict


Input:

* Image file (`.jpg`, `.jpeg`, `.png`)

Output:

* Predicted class
* Confidence score

### 4.2 Retraining Endpoint


POST /retrain


Input:
A ZIP file structured as:


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


Output:

* Confirmation of successful retraining

### 4.3 Health Check


GET /health




## 5. Locust Load Test Results

Locust was used to simulate concurrent API requests to evaluate performance during high-traffic conditions.

### Test Configuration

* 5 users
* Spawn rate: 2 users/second
* Endpoint tested: `/predict`
* File used: `test_image.jpg`

### Results Summary

* All requests reached the FastAPI service successfully.
* Average response latency remained stable under load.
* The API handled multiple concurrent requests without timeout.

A screenshot and logs from the Locust test are included in the `/locust_results` directory (or attached in the report submission).



## 6. Model File

The trained model is saved in:


models/cifar10_model.h5


This file is used by the FastAPI backend for inference and retraining.



## 7. Video Demo 

A full walkthrough demonstrating:

* Notebook preprocessing
* Model training
* API deployment
* Streamlit usage
* Retraining workflow
* Locust performance testing

YouTube link:
(Add your link after uploading the video)



## 8. How to Retrain the Model with New Data

1. Prepare a folder with CIFAR-10-style subfolders.
2. Zip the folder:


new_data.zip


3. Upload zip file through:

* Streamlit UI
  or
* POST request to `/retrain`

The backend retrains the model for 3 epochs and updates the saved weights.





