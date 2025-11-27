import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import random

TEST_DIR = "data/raw/images_test_rev1"
MODEL_PATH = "models/galaxy_mobilenet.h5"
PROCESSED_DIR = "data/processed"
IMG_SIZE = (224, 224)

CLASS_COLUMNS = ["class_0", "class_1", "class_2"]

# Number of images to test (small random subset)
NUM_TEST = 50  

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

def load_and_preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return img

def predict_test_images():
    all_files = os.listdir(TEST_DIR)
    test_files = random.sample(all_files, min(NUM_TEST, len(all_files)))  # random subset
    galaxy_ids = [os.path.splitext(f)[0] for f in test_files]
    
    batch_images = np.array([load_and_preprocess_image(os.path.join(TEST_DIR, f))
                             for f in test_files])
    
    preds = model.predict(batch_images)

    # Predicted class index
    predicted_classes = np.argmax(preds, axis=1)

    # Build DataFrame
    df = pd.DataFrame(preds, columns=CLASS_COLUMNS)
    df.insert(0, "GalaxyID", galaxy_ids)
    df["PredictedClass"] = predicted_classes

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, "predictions_random_subset.csv"), index=False)
    print("✅ Predictions saved to", os.path.join(PROCESSED_DIR, "predictions_random_subset.csv"))

if __name__ == "__main__":
    predict_test_images()
