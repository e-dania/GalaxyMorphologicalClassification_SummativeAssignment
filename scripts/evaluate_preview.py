import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import random

# Paths
TEST_DIR = "data/raw/images_test_rev1"
MODEL_PATH = "models/galaxy_mobilenet.h5"
PROCESSED_DIR = "data/processed"
IMG_SIZE = (224, 224)

CLASS_COLUMNS = ["class_0", "class_1", "class_2"]
NUM_TEST = 50  # random subset for preview

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

# Image loader & preprocessor
def load_and_preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return img

# Run predictions on a subset
def preview_predictions():
    all_files = os.listdir(TEST_DIR)
    test_files = random.sample(all_files, min(NUM_TEST, len(all_files)))
    galaxy_ids = [os.path.splitext(f)[0] for f in test_files]

    batch_images = np.array([load_and_preprocess_image(os.path.join(TEST_DIR, f))
                             for f in test_files])

    preds = model.predict(batch_images)

    # Build results DataFrame
    df = pd.DataFrame(preds, columns=CLASS_COLUMNS)
    df.insert(0, "GalaxyID", galaxy_ids)
    df["TopClass"] = df[CLASS_COLUMNS].idxmax(axis=1)
    df["TopProbability"] = df[CLASS_COLUMNS].max(axis=1)

    # Save CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, "predictions_preview.csv")
    df.to_csv(save_path, index=False)
    print(f"✅ Preview predictions saved to {save_path}")

if __name__ == "__main__":
    preview_predictions()
