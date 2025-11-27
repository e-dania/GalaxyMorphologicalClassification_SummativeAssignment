import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_TRAIN_DIR = "data/raw/images_training_rev1"
LABELS_PATH = "data/raw/training_solutions_rev1.csv"
PROCESSED_DIR = "data/processed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 5000  # Number of images per batch

# Map labels to 3-class target
def map_labels(df):
    df["class"] = df.apply(lambda row:
        0 if row["Class1.1"] > 0.5 else
        1 if row["Class4.1"] > 0.5 else
        2,
        axis=1)
    return df

# Load and preprocess a single image
def load_and_preprocess_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
        return img
    except Exception as e:
        print(f"⚠️ Error loading {path}: {e}")
        return None

# Find the correct image file for a given GalaxyID
def find_image(img_id):
    possible_extensions = [".jpg", ".jpeg", ".png"]
    for ext in possible_extensions:
        img_path = os.path.join(RAW_TRAIN_DIR, f"{img_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None

def preprocess_dataset():
    print("📥 Loading labels...")
    df = pd.read_csv(LABELS_PATH)
    df = map_labels(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    images, labels = [], []
    batch_count = 0

    print("🔄 Processing images in batches...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = str(int(row["GalaxyID"]))
        img_path = find_image(img_id)
        if img_path is None:
            continue

        img = load_and_preprocess_image(img_path)
        if img is None:
            continue

        images.append(img)
        labels.append(row["class"])

        # Save a batch
        if len(images) == BATCH_SIZE or idx == len(df) - 1:
            batch_count += 1
            np.save(os.path.join(PROCESSED_DIR, f"X_batch{batch_count}.npy"), np.array(images, dtype=np.float32))
            np.save(os.path.join(PROCESSED_DIR, f"y_batch{batch_count}.npy"), np.array(labels, dtype=np.int32))
            print(f"💾 Saved batch {batch_count} with {len(images)} images")
            images, labels = [], []

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    preprocess_dataset()