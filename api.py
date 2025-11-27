from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import shutil

# ------------------ Configuration ------------------
MODEL_PATH = "models/galaxy_mobilenet.h5"
NEW_DATA_DIR = "data/new"
PROCESSED_DIR = "data/processed"
IMG_SIZE = (224, 224)
CLASS_COLUMNS = ["class_0", "class_1", "class_2"]

# ------------------ Initialize API ------------------
app = FastAPI(title="Galaxy Morphology API", version="1.0")

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------ Utility Functions ------------------
def preprocess_image(file):
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def retrain_model():
    """
    Retrains the model using images in NEW_DATA_DIR.
    Expects subfolders for each class: NEW_DATA_DIR/class_0, class_1, class_2
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    if not os.path.exists(NEW_DATA_DIR):
        raise HTTPException(status_code=400, detail="No new data directory found for retraining.")

    # Image generator
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

    # Load base model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(CLASS_COLUMNS), activation='softmax')(x)
    retrain_model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    retrain_model.compile(optimizer=Adam(learning_rate=1e-4),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    # Train
    history = retrain_model.fit(train_gen, validation_data=val_gen, epochs=3)

    # Save new model
    retrain_model.save(MODEL_PATH)

    # Reload global model
    global model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Optional: clear new data directory
    shutil.rmtree(NEW_DATA_DIR)
    os.makedirs(NEW_DATA_DIR, exist_ok=True)

    val_acc = history.history['val_accuracy'][-1]
    return f"Model retrained successfully! Validation accuracy: {val_acc:.2f}"

# ------------------ API Endpoints ------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_array = preprocess_image(file.file)
    preds = model.predict(img_array)
    return {cls: float(preds[0][i]) for i, cls in enumerate(CLASS_COLUMNS)}

@app.post("/predict_bulk")
async def predict_bulk(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        img_array = preprocess_image(file.file)
        preds = model.predict(img_array)
        results.append({cls: float(preds[0][i]) for i, cls in enumerate(CLASS_COLUMNS)})
    return results

@app.post("/upload_new_data")
async def upload_new_data(file: UploadFile = File(...), class_name: str = "class_0"):
    if class_name not in CLASS_COLUMNS:
        raise HTTPException(status_code=400, detail="Invalid class name.")

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files allowed.")

    class_dir = os.path.join(NEW_DATA_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    file_path = os.path.join(class_dir, file.filename)

    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File already exists.")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"File {file.filename} uploaded to {class_name}."}

@app.post("/retrain")
async def retrain():
    message = retrain_model()
    return JSONResponse({"message": message})
