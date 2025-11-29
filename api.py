import os
import shutil
import json
import threading
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# ---------------- config ----------------
BASE_DIR = "data"
NEW_DATA_DIR = os.path.join(BASE_DIR, "new")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "galaxy_mobilenet.h5")
IMG_SIZE = (224, 224)
CLASS_COLUMNS = ["class_0", "class_1", "class_2"]
NUM_CLASSES = len(CLASS_COLUMNS)
MIN_REQUIRED_TRAINING_IMAGES = 8   # fail retrain if too few training images

progress_file = "progress.json"
progress_lock = threading.Lock()

# ensure directories exist
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="Galaxy Morphology API")

# Load model (if present) or set model to None and create lazily
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print("Warning loading model:", e)
        model = None

# ---------------- helper utils ----------------
def write_progress(d: dict):
    with progress_lock:
        with open(progress_file, "w") as f:
            json.dump(d, f)

def clear_progress():
    write_progress({"done": True, "epoch": 0, "total_epochs": 0})

def preprocess_image_fileobj(fileobj):
    img = Image.open(fileobj).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- prediction endpoints ----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Single image prediction. POST multipart/form-data with key 'file'."""
    global model
    if model is None:
        raise HTTPException(503, detail="Model not loaded")

    try:
        img_arr = preprocess_image_fileobj(file.file)
        preds = model.predict(img_arr)
        return {CLASS_COLUMNS[i]: float(preds[0][i]) for i in range(NUM_CLASSES)}
    except Exception as e:
        raise HTTPException(400, detail=str(e))

@app.post("/predict_bulk")
async def predict_bulk(files: List[UploadFile] = File(...)):
    """Bulk predict: send multiple files with key 'files'."""
    global model
    if model is None:
        raise HTTPException(503, detail="Model not loaded")

    results = []
    for f in files:
        try:
            arr = preprocess_image_fileobj(f.file)
            preds = model.predict(arr)
            results.append({CLASS_COLUMNS[i]: float(preds[0][i]) for i in range(NUM_CLASSES)})
        except Exception as e:
            results.append({"error": str(e)})
    return results

# ---------------- upload new data ----------------
@app.post("/upload_new_data")
async def upload_new_data(
    files: List[UploadFile] = File(...),
    class_name: str = Form(...)
):
    """Upload one or many images for a given class_name (form field)."""
    if class_name not in CLASS_COLUMNS:
        raise HTTPException(400, "Invalid class name")

    if not files:
        raise HTTPException(400, "No files provided")

    class_dir = os.path.join(NEW_DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    saved = []
    for f in files:
        fname = f.filename
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        dst = os.path.join(class_dir, fname)
        # avoid overwrite
        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            dst = os.path.join(class_dir, f"{base}_copy{ext}")
        with open(dst, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved.append(os.path.basename(dst))
    if not saved:
        raise HTTPException(400, "No valid images uploaded")
    return {"message": f"Uploaded {len(saved)} file(s) to {class_name}", "files": saved}

# ---------------- retrain logic (background) ----------------
class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        write_progress({
            "epoch": epoch + 1,
            "accuracy": float(logs.get("accuracy", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0)),
            "total_epochs": self.total_epochs,
            "done": False
        })

    def on_train_end(self, logs=None):
        write_progress({
            "epoch": self.total_epochs,
            "accuracy": float(logs.get("accuracy", 0)) if logs else 0,
            "val_accuracy": float(logs.get("val_accuracy", 0)) if logs else 0,
            "total_epochs": self.total_epochs,
            "done": True
        })

def collect_new_data_count():
    total = 0
    if not os.path.exists(NEW_DATA_DIR):
        return 0
    for c in os.listdir(NEW_DATA_DIR):
        path = os.path.join(NEW_DATA_DIR, c)
        if os.path.isdir(path):
            total += len([f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return total

def run_retraining(total_epochs: int = 3):
    """Background retraining job. Moves new->train/val, trains, saves model."""
    global model

    try:
        write_progress({"epoch": 0, "total_epochs": total_epochs, "done": False})

        # check new data exists
        total_new = collect_new_data_count()
        if total_new == 0:
            write_progress({"done": True, "error": "No new data found"})
            return

        # move new data to train/val with 80/20 split per-class
        for class_name in os.listdir(NEW_DATA_DIR):
            class_new_dir = os.path.join(NEW_DATA_DIR, class_name)
            if not os.path.isdir(class_new_dir):
                continue
            train_class_dir = os.path.join(TRAIN_DIR, class_name)
            val_class_dir = os.path.join(VAL_DIR, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)

            files = [f for f in os.listdir(class_new_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            for i, fname in enumerate(files):
                src = os.path.join(class_new_dir, fname)
                dst = train_class_dir if (i % 5) != 0 else val_class_dir  # 80/20
                shutil.move(src, dst)
            shutil.rmtree(class_new_dir, ignore_errors=True)

        # make sure we have enough data to actually train
        # count train samples
        train_gen_check = ImageDataGenerator(rescale=1./255).flow_from_directory(
            TRAIN_DIR, target_size=IMG_SIZE, class_mode="categorical", shuffle=False
        )
        if train_gen_check.samples < MIN_REQUIRED_TRAINING_IMAGES:
            write_progress({"done": True, "error": f"Not enough training images ({train_gen_check.samples})"})
            return

        # Create generators
        datagen = ImageDataGenerator(rescale=1./255)
        train_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, class_mode="categorical", batch_size=32)
        val_gen = datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, class_mode="categorical", batch_size=32)

        # Build or load model
        if os.path.exists(MODEL_PATH):
            m = load_model(MODEL_PATH)
        else:
            base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
            x = GlobalAveragePooling2D()(base.output)
            x = Dense(128, activation="relu")(x)
            x = Dropout(0.2)(x)
            preds = Dense(NUM_CLASSES, activation="softmax")(x)
            m = Model(inputs=base.input, outputs=preds)
            # compile initial
            m.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

        # Unfreeze some layers for fine-tuning (optional)
        for layer in m.layers:
            layer.trainable = True

        m.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

        # Train
        m.fit(train_gen, validation_data=val_gen, epochs=total_epochs, callbacks=[ProgressCallback(total_epochs)])

        # Save
        m.save(MODEL_PATH)

        # update global model reference
        model = tf.keras.models.load_model(MODEL_PATH)

        # optional cleanup: keep train/val, just clear NEW_DATA_DIR
        # ensure NEW_DATA_DIR exists and empty
        shutil.rmtree(NEW_DATA_DIR, ignore_errors=True)
        os.makedirs(NEW_DATA_DIR, exist_ok=True)

    except Exception as e:
        write_progress({"done": True, "error": str(e)})
        raise
    finally:
        # make sure progress done flag true if not already
        try:
            with progress_lock:
                if os.path.exists(progress_file):
                    data = {}
                    with open(progress_file, "r") as f:
                        try:
                            data = json.load(f)
                        except Exception:
                            data = {}
                    if not data.get("done"):
                        data["done"] = True
                        with open(progress_file, "w") as fw:
                            json.dump(data, fw)
        except Exception:
            pass

@app.post("/retrain")
async def retrain(total_epochs: int = Form(3), background: BackgroundTasks = None):
    """
    Start retraining in background. Provide form field 'total_epochs' (int).
    Response returns immediately and background thread continues training.
    """
    # Check new data exists first (quick check)
    if collect_new_data_count() == 0:
        return JSONResponse({"message": "No new data uploaded; retrain cancelled."}, status_code=400)

    # start background thread
    t = threading.Thread(target=run_retraining_wrapper, args=(total_epochs,), daemon=True)
    t.start()
    return {"message": "Retraining started", "total_epochs": total_epochs}

def run_retraining_wrapper(total_epochs):
    try:
        run_retraining(total_epochs)
    except Exception as e:
        write_progress({"done": True, "error": str(e)})

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
