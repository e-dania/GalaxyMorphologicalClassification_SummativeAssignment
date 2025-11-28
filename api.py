from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import numpy as np
import tensorflow as tf
import os, shutil

MODEL_PATH = "models/galaxy_mobilenet.h5"
NEW_DATA_DIR = "data/new"
IMG_SIZE = (224, 224)
CLASS_COLUMNS = ["class_0", "class_1", "class_2"]

app = FastAPI()

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = preprocess_image(file.file)
    preds = model.predict(img)
    return {CLASS_COLUMNS[i]: float(preds[0][i]) for i in range(len(CLASS_COLUMNS))}

@app.post("/upload_new_data")
async def upload_new_data(
    files: List[UploadFile] = File(...),
    class_name: str = "class_0"
):
    if class_name not in CLASS_COLUMNS:
        raise HTTPException(status_code=400, detail="Invalid class name")

    class_dir = os.path.join(NEW_DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    saved = 0

    for file in files:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        save_path = os.path.join(class_dir, file.filename)

        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        saved += 1

    return {"message": f"Uploaded {saved} files to {class_name}"}

@app.post("/retrain")
async def retrain():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    if not os.listdir(NEW_DATA_DIR):
        raise HTTPException(status_code=400, detail="No new data uploaded.")

    datagen = ImageDataGenerator(rescale=1/255, validation_split=0.1)

    train_gen = datagen.flow_from_directory(
        NEW_DATA_DIR, target_size=IMG_SIZE,
        class_mode="categorical", subset="training"
    )

    val_gen = datagen.flow_from_directory(
        NEW_DATA_DIR, target_size=IMG_SIZE,
        class_mode="categorical", subset="validation"
    )

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(3, activation="softmax")(x)
    new_model = Model(inputs=base.input, outputs=out)

    for layer in base.layers:
        layer.trainable = False

    new_model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    history = new_model.fit(train_gen, validation_data=val_gen, epochs=3)

    new_model.save(MODEL_PATH)

    global model
    model = tf.keras.models.load_model(MODEL_PATH)

    shutil.rmtree(NEW_DATA_DIR)
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

    return {"message": "Retraining complete"}
