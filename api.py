from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import numpy as np
import tensorflow as tf
import os, shutil, json, threading

MODEL_PATH = "models/galaxy_mobilenet.h5"
NEW_DATA_DIR = "data/new"
IMG_SIZE = (224, 224)
CLASS_COLUMNS = ["class_0", "class_1", "class_2"]

app = FastAPI()

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# Progress tracking
progress_file = "progress.json"
progress_lock = threading.Lock()

def update_progress(data):
    with progress_lock:
        with open(progress_file, "w") as f:
            json.dump(data, f)

# ---------------- Image preprocessing ----------------
def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)

# ---------------- Prediction ------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = preprocess_image(file.file)
    preds = model.predict(img)
    return {CLASS_COLUMNS[i]: float(preds[0][i]) for i in range(len(CLASS_COLUMNS))}

# ---------------- Upload new data -------------------
@app.post("/upload_new_data")
async def upload_new_data(files: List[UploadFile] = File(...), class_name: str = "class_0"):
    if class_name not in CLASS_COLUMNS:
        raise HTTPException(status_code=400, detail="Invalid class name.")

    class_dir = os.path.join(NEW_DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    uploaded_files = []

    for f in files:
        if not f.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        save_path = os.path.join(class_dir, f.filename)
        if os.path.exists(save_path):
            base, ext = os.path.splitext(f.filename)
            save_path = os.path.join(class_dir, f"{base}_copy{ext}")

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        uploaded_files.append(f.filename)

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No valid images uploaded.")

    return {"message": f"Uploaded {len(uploaded_files)} file(s) to {class_name}.", "files": uploaded_files}

# ---------------- Retrain --------------------------
def retrain_model():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    total_epochs = 3

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            data = {
                "epoch": epoch + 1,
                "accuracy": float(logs.get("accuracy", 0)),
                "val_accuracy": float(logs.get("val_accuracy", 0)),
                "total_epochs": total_epochs,
                "done": False
            }
            update_progress(data)

        def on_train_end(self, logs=None):
            data = {
                "epoch": total_epochs,
                "accuracy": float(logs.get("accuracy", 0)) if logs else 0,
                "val_accuracy": float(logs.get("val_accuracy", 0)) if logs else 0,
                "total_epochs": total_epochs,
                "done": True
            }
            update_progress(data)

    if not os.listdir(NEW_DATA_DIR):
        update_progress({"done": True})
        return

    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.1)
    train_gen = datagen.flow_from_directory(
        NEW_DATA_DIR, target_size=IMG_SIZE, class_mode="categorical", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        NEW_DATA_DIR, target_size=IMG_SIZE, class_mode="categorical", subset="validation"
    )

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(len(CLASS_COLUMNS), activation="softmax")(x)
    new_model = Model(inputs=base.input, outputs=out)

    for layer in base.layers:
        layer.trainable = False

    new_model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    new_model.fit(train_gen, validation_data=val_gen, epochs=total_epochs, callbacks=[ProgressCallback()])
    new_model.save(MODEL_PATH)

    global model
    model = tf.keras.models.load_model(MODEL_PATH)

    shutil.rmtree(NEW_DATA_DIR)
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

# Retrain endpoint (non-blocking)
@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    # Initialize progress file
    update_progress({"epoch": 0, "val_accuracy": 0, "total_epochs": 3, "done": False})
    background_tasks.add_task(retrain_model)
    return {"message": "Retraining started. Check /progress to monitor progress."}

# Progress endpoint
@app.get("/progress")
async def get_progress():
    if not os.path.exists(progress_file):
        return {"done": True}

    with progress_lock:
        with open(progress_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return {"done": False}

    return data
