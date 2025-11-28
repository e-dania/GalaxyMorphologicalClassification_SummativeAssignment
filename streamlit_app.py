import streamlit as st
import requests
import pandas as pd

API_URL = "https://galaxymorphologicalclassification.onrender.com"

st.set_page_config(page_title="Galaxy Dashboard", layout="wide")

menu = st.sidebar.radio("Select", ["Uptime", "Visualize Data", "Predict", "Upload Data", "Retrain"])

# -------------------- UPTIME -----------------------
if menu == "Uptime":
    st.title("Model Uptime")
    try:
        r = requests.get(f"{API_URL}/predict")
        st.success("API is live!")
    except:
        st.error("API is offline.")

# --------------------- VISUALIZE --------------------
elif menu == "Visualize Data":
    st.title("Sample Predictions")
    try:
        df = pd.read_csv("data/processed/predictions_random_subset.csv")
        st.dataframe(df.head())
        st.bar_chart(df[df.columns[1:]].mean())
    except:
        st.warning("CSV not found.")

# --------------------- PREDICT ----------------------
elif menu == "Predict":
    st.title("Predict Galaxy Class")
    img = st.file_uploader("Upload galaxy image", type=["jpg","png","jpeg"])

    if st.button("Predict"):
        if img:
            files = {"file": (img.name, img, "image/jpeg")}
            res = requests.post(f"{API_URL}/predict", files=files)
            st.json(res.json())
        else:
            st.error("No image selected.")

# ------------------ UPLOAD NEW DATA -----------------
elif menu == "Upload Data":
    st.title("Upload New Training Data")
    class_name = st.selectbox("Select Class", ["class_0","class_1","class_2"])
    imgs = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if st.button("Upload Images"):
        if imgs:
            files_list = []
            for img in imgs:
                files_list.append(("files", (img.name, img.getvalue(), "image/jpeg")))

            res = requests.post(
                f"{API_URL}/upload_new_data",
                files=files_list,
                data={"class_name": class_name}
            )

            st.success(res.json()["message"])
        else:
            st.error("No files uploaded.")

# --------------------- RETRAIN ----------------------
elif menu == "Retrain":
    st.title("Retrain Model")

    if st.button("Start Retraining"):
        st.info("Retraining... This may take a while.")
        res = requests.post(f"{API_URL}/retrain")
        st.success(res.json()["message"])
