import streamlit as st
import os
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Configuration ------------------
MODEL_PATH = "models/galaxy_mobilenet.h5"
NEW_DATA_DIR = "data/new"
RAW_DATA_DIR = "data/raw/images_train"  # For visualization
IMG_SIZE = (224, 224)
CLASS_COLUMNS = ["class_0", "class_1", "class_2"]

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Galaxy Morphology Dashboard", layout="wide")
st.title("🌌 Galaxy Morphology Classification Dashboard")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

try:
    model = load_model()
    model_status = "✅ Model Loaded"
except:
    model = None
    model_status = "❌ Model Not Available"

# ------------------ Utility Functions ------------------
def preprocess_image(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def retrain_model():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    if not os.path.exists(NEW_DATA_DIR):
        st.warning("No new data for retraining.")
        return "Retraining skipped."

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    train_gen = datagen.flow_from_directory(
        NEW_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        NEW_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(CLASS_COLUMNS), activation='softmax')(x)
    retrain_model = Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False

    retrain_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    retrain_model.fit(train_gen, validation_data=val_gen, epochs=3)
    retrain_model.save(MODEL_PATH)
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    return "✅ Model retrained successfully!"

def visualize_data():
    # Visualize number of images per class
    class_counts = {}
    for cls in CLASS_COLUMNS:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if os.path.exists(cls_dir):
            class_counts[cls] = len(os.listdir(cls_dir))
        else:
            class_counts[cls] = 0
    df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])

    st.subheader("📊 Dataset Class Distribution")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Class", y="Count", palette="viridis", ax=ax)
    st.pyplot(fig)

    # Sample images
    st.subheader("🖼 Sample Images")
    cols = st.columns(len(CLASS_COLUMNS))
    for i, cls in enumerate(CLASS_COLUMNS):
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        if os.path.exists(cls_dir) and len(os.listdir(cls_dir)) > 0:
            sample_img = Image.open(os.path.join(cls_dir, os.listdir(cls_dir)[0]))
            cols[i].image(sample_img, caption=cls, use_column_width=True)

# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["Model Uptime", "Data Visualizations", "Train / Retrain"])

# ---- Model Uptime ----
with tab1:
    st.subheader("⚡ Model Status")
    st.write(model_status)
    if model is not None:
        st.write(f"Model last loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.write("Upload an image for prediction:")
    uploaded_file = st.file_uploader("", type=["png","jpg","jpeg"])
    if uploaded_file and st.button("Predict"):
        image = Image.open(uploaded_file)
        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        st.subheader("Prediction Probabilities:")
        for i, cls in enumerate(CLASS_COLUMNS):
            st.write(f"{cls}: {preds[0][i]:.3f}")

# ---- Data Visualizations ----
with tab2:
    visualize_data()

# ---- Train / Retrain ----
with tab3:
    st.subheader("Upload New Training Data")
    new_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    class_name = st.selectbox("Select class", CLASS_COLUMNS)
    if new_file and st.button("Upload"):
        class_dir = os.path.join(NEW_DATA_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        file_path = os.path.join(class_dir, new_file.name)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(new_file, f)
        st.success(f"File {new_file.name} uploaded to {class_name}")

    st.subheader("Retrain Model")
    if st.button("Retrain"):
        status = retrain_model()
        st.success(status)
