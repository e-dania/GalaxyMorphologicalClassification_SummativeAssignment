import os, shutil, json, threading
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import Callback

app = FastAPI()

BASE_DIR = "data"
NEW_DATA_DIR = os.path.join(BASE_DIR, "new")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = "model.h5"
NUM_CLASSES = 3

# Progress tracking
progress_file = "progress.json"
progress_lock = threading.Lock()

def update_progress(data):
    with progress_lock:
        with open(progress_file, "w") as f:
            json.dump(data, f)

class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        update_progress({
            "epoch": epoch+1,
            "accuracy": float(logs.get("accuracy", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0)),
            "total_epochs": self.total_epochs,
            "done": False
        })

    def on_train_end(self, logs=None):
        update_progress({
            "epoch": self.total_epochs,
            "accuracy": float(logs.get("accuracy", 0)) if logs else 0,
            "val_accuracy": float(logs.get("val_accuracy", 0)) if logs else 0,
            "total_epochs": self.total_epochs,
            "done": True
        })

# Ensure directories exist
os.makedirs(NEW_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

@app.post("/upload_new_data")
async def upload_new_data(file: UploadFile = File(...), class_name: str = Form(...)):
    class_dir = os.path.join(NEW_DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    save_path = os.path.join(class_dir, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"message": f"Uploaded {file.filename} to {class_name}"}

@app.post("/retrain")
async def retrain_model(total_epochs: int = Form(...)):
    # Move new data to train/val with 80/20 split
    for class_name in os.listdir(NEW_DATA_DIR):
        class_new_dir = os.path.join(NEW_DATA_DIR, class_name)
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        files = os.listdir(class_new_dir)
        for i, f in enumerate(files):
            src = os.path.join(class_new_dir, f)
            dst = train_class_dir if i % 5 != 0 else val_class_dir
            shutil.move(src, dst)
        shutil.rmtree(class_new_dir)

    # Generators
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        TRAIN_DIR, target_size=(224,224), batch_size=32, class_mode="categorical"
    )
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR, target_size=(224,224), batch_size=32, class_mode="categorical"
    )

    # Load or create model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation="relu")(x)
        preds = Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs=base.input, outputs=preds)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Retrain
    model.fit(train_gen, validation_data=val_gen, epochs=total_epochs, callbacks=[ProgressCallback(total_epochs)])
    model.save(MODEL_PATH)

    # Cleanup
    shutil.rmtree(NEW_DATA_DIR)
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

    return JSONResponse({"message": "Retraining complete"})

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
