# CIFAR10-Image-MLOps-Pipeline

This project implements a full MLOps pipeline using the CIFAR-10 image dataset.  
It demonstrates:

✔ Model training  
✔ Evaluation  
✔ Prediction  
✔ Retraining using new data  
✔ FastAPI backend  
✔ Streamlit User Interface  
✔ Cloud deployment with Docker + Render  
✔ Load testing with Locust  

---

## 1. Project Structure

project/
│── app/
│     ├── main.py
│     ├── streamlit_app.py
│
│── src/
│     ├── preprocessing.py
│     ├── model.py
│     ├── prediction.py
│
│── notebook/
│     ├── train_model.ipynb
│     ├── retrain_model.ipynb
│
│── models/
│     ├── cifar10_model.h5
│     ├── class_names.json
│
│── Dockerfile
│── requirements.txt
│── locustfile.py
│── README.md

---

## 2. How to Run Locally

### Install dependencies:
pip install -r requirements.txt

### Start API:
uvicorn app.main:app --reload

### Start Streamlit UI:
streamlit run app/streamlit_app.py

---

## 3. Deployment

The application is containerized using Docker and deployed on Render.

---

## 4. Video Demo
🎥 YouTube Link: (paste)

---

## 5. Team Features
- Upload new data
- Trigger retraining
- Predict images using the model
