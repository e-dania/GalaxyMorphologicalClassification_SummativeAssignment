# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
import io
import time

# ------------------ Page Config ------------------
st.set_page_config(page_title="Galaxy Morphology Dashboard", layout="wide")

# ------------------ Sidebar ------------------
st.sidebar.title("Galaxy Morphology Control Panel")
app_mode = st.sidebar.radio(
    "Choose Action",
    ["Check Model Uptime", "Visualize Data", "Predict Galaxy", "Retrain Model", "Upload New Data"]
)

API_URL = "https://galaxymorphologicalclassification.onrender.com"

# ------------------ Helper Functions ------------------
def check_model_uptime():
    try:
        resp = requests.get(f"{API_URL}/predict")
        return True
    except:
        return False

def visualize_predictions(csv_path="data/processed/predictions_random_subset.csv"):
    try:
        df = pd.read_csv(csv_path)
        st.write("Sample Predictions")
        st.dataframe(df.head(10))

        st.write("Class Distribution")
        st.bar_chart(df[df.columns[1:]].mean())
    except FileNotFoundError:
        st.warning("Prediction CSV not found. Run predictions first.")

def predict_image(uploaded_file):
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict", files=files)
        if response.status_code == 200:
            st.success("Prediction Complete")
            st.json(response.json())
        else:
            st.error(f"Prediction failed: {response.text}")

def retrain_model():
    response = requests.post(f"{API_URL}/retrain")
    if response.status_code == 200:
        st.success(response.json()["message"])
    else:
        st.error(f"Retraining failed: {response.text}")

def upload_new_data(uploaded_file, class_name):
    if uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
        response = requests.post(f"{API_URL}/upload_new_data", files=files, data={"class_name": class_name})
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Upload failed: {response.text}")

# ------------------ App Modes ------------------
if app_mode == "Check Model Uptime":
    st.header("Model Uptime")
    st.write("Checking if the model API is live...")
    status = check_model_uptime()
    if status:
        st.success("✅ Model API is up and running!")
    else:
        st.error("❌ Model API is not reachable.")

elif app_mode == "Visualize Data":
    st.header("Data Visualization")
    st.write("Visualizing predictions from a CSV file")
    visualize_predictions()

elif app_mode == "Predict Galaxy":
    st.header("Predict a Galaxy Morphology")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if st.button("Predict"):
        predict_image(uploaded_file)

elif app_mode == "Retrain Model":
    st.header("Retrain the Model")
    st.write("Click the button to retrain the model with new data")
    if st.button("Retrain Now"):
        with st.spinner("Retraining model..."):
            retrain_model()

elif app_mode == "Upload New Data":
    st.header("Upload New Data for Retraining")
    class_name = st.selectbox("Select Class", ["class_0", "class_1", "class_2"])
    uploaded_file = st.file_uploader("Upload an image for new data", type=["jpg", "jpeg", "png"])
    if st.button("Upload"):
        upload_new_data(uploaded_file, class_name)
