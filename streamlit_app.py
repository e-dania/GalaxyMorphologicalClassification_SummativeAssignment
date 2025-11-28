# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import time
import json

API_URL = "https://galaxymorphologicalclassification.onrender.com"

st.set_page_config(page_title="Galaxy Morphology Dashboard", layout="wide")

# -------------------- SIDEBAR MENU -----------------------
menu = st.sidebar.radio(
    "Select",
    ["Uptime", "Visualize Data", "Predict", "Upload Data", "Retrain"],
)

# -------------------- CSS FOR ANIMATIONS -----------------
st.markdown(
    """
<style>
@keyframes fadeIn {
  from {opacity: 0;} to {opacity: 1;}
}
.fade-in {
  animation: fadeIn 1s ease-in-out;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- CONFETTI ---------------------------
def show_confetti():
    st.snow()

# -------------------- UPTIME -----------------------------
if menu == "Uptime":
    st.title("Model Uptime")
    try:
        requests.get(f"{API_URL}/predict")
        st.success("✅ API is live!")
    except:
        st.error("❌ API is offline.")

# --------------------- VISUALIZE -------------------------
elif menu == "Visualize Data":
    st.title("Sample Predictions")
    try:
        df = pd.read_csv("data/processed/predictions_random_subset.csv")
        st.dataframe(df.head())
        st.bar_chart(df[df.columns[1:]].mean())
    except:
        st.warning("CSV not found. Run predictions first.")

# --------------------- PREDICT ---------------------------
elif menu == "Predict":
    st.subheader("🔭 Galaxy Classification")
    img = st.file_uploader("Upload galaxy image", type=["jpg", "png", "jpeg"])

    # ---------- IMAGE PREVIEW ----------
    if img:
        st.markdown("### 🖼 Preview")
        st.image(img, use_column_width=True, caption="Uploaded Image")

    # ---------- PREDICT BUTTON ----------
    if st.button("Run Prediction"):
        if img:
            with st.spinner("🔄 Sending image to model… please wait"):
                try:
                    files = {"file": (img.name, img, "image/jpeg")}
                    res = requests.post(f"{API_URL}/predict", files=files)
                    res_json = res.json()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    res_json = None

            if res_json:
                st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                st.success("Prediction Complete! 🎉")
                st.json(res_json)
                st.markdown('</div>', unsafe_allow_html=True)
                show_confetti()
        else:
            st.error("Please upload an image first.")

# ------------------ UPLOAD NEW DATA ----------------------
elif menu == "Upload Data":
    st.title("Upload New Training Data")
    class_name = st.selectbox("Select Class", ["class_0", "class_1", "class_2"])
    imgs = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if st.button("Upload Images"):
        if imgs:
            multipart_files = [
                ("files", (img.name, img.getvalue(), "image/jpeg")) for img in imgs
            ]
            payload = {"class_name": class_name}
            st.markdown("### 🖼 Previews")
            for img in imgs:
                st.image(img, use_column_width=False, width=150)

            try:
                res = requests.post(
                    f"{API_URL}/upload_new_data", files=multipart_files, data=payload
                )
                res_json = res.json()
                st.success(res_json.get("message", "Upload complete!"))
            except requests.exceptions.JSONDecodeError:
                st.error("Upload failed: Could not decode server response.")
            except Exception as e:
                st.error(f"Upload failed: {e}")
        else:
            st.error("No files uploaded.")

# --------------------- RETRAIN MODEL ---------------------
elif menu == "Retrain":
    st.title("Retrain Model")
    st.write("Click below to retrain the model. This may take some time.")

    if st.button("Start Retraining"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Start retraining (non-blocking on the server)
            requests.post(f"{API_URL}/retrain", timeout=5)

            st.success("Retraining started! 🚀")
            status_text.text("Monitoring progress...")

            # Poll the server for progress
            while True:
                try:
                    res = requests.get(f"{API_URL}/progress", timeout=5)
                    data = res.json()
                    epoch = data.get("epoch", 0)
                    val_acc = data.get("val_accuracy", 0.0)

                    progress_percent = int((epoch / data.get("total_epochs", 3)) * 100)
                    progress_bar.progress(progress_percent)
                    status_text.text(
                        f"Epoch {epoch}/{data.get('total_epochs',3)} — Validation Accuracy: {val_acc:.2%}"
                    )

                    if data.get("done", False):
                        progress_bar.progress(100)
                        status_text.text("Retraining complete! 🎉")
                        st.balloons()
                        break

                except requests.exceptions.RequestException:
                    status_text.text("Waiting for progress...")
                
                time.sleep(2)  # Poll every 2 seconds

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to start retraining: {e}")
