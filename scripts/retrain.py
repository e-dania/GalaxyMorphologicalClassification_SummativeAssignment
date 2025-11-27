import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Paths
MODEL_PATH = "models/galaxy_mobilenet.h5"
NEW_DATA_DIR = "data/new"
IMG_SIZE = (224, 224)
CLASS_COLUMNS = ["class_0", "class_1", "class_2"]

# Helper to get label from filename (adjust this to your naming/CSV)
def get_label_from_filename(filename):
    # Example: filenames like "class0_123.png"
    if "class0" in filename.lower():
        return [1, 0, 0]
    elif "class1" in filename.lower():
        return [0, 1, 0]
    elif "class2" in filename.lower():
        return [0, 0, 1]
    else:
        raise ValueError(f"Unknown class in filename: {filename}")

# Load new data for retraining
def load_new_data():
    if not os.path.exists(NEW_DATA_DIR):
        print("⚠️  No new data folder found. Skipping retraining.")
        return None, None, None

    files = [f for f in os.listdir(NEW_DATA_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        print("⚠️  No new images found in 'data/new'. Skipping retraining.")
        return None, None, None

    X_new = np.array([np.array(Image.open(os.path.join(NEW_DATA_DIR, f)).resize(IMG_SIZE))/255.0 for f in files])
    y_new = np.array([get_label_from_filename(f) for f in files])
    return X_new, y_new, files

# Main retraining function
def retrain():
    X_new, y_new, files = load_new_data()
    if X_new is None:
        return

    # Load existing model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Loaded model for retraining with {len(files)} new images.")

    # Fine-tune last layers only (optional)
    # for layer in model.layers[:-2]:  # freeze all but last 2 layers
    #     layer.trainable = False

    model.fit(X_new, y_new, epochs=5, batch_size=16, verbose=1)

    # Save updated model
    backup_path = MODEL_PATH.replace(".h5", "_backup.h5")
    model.save(backup_path)  # save a backup
    model.save(MODEL_PATH)
    print(f"✅ Model retrained and saved. Backup saved to {backup_path}")

    # Optional: clear the 'new' folder
    for f in files:
        os.remove(os.path.join(NEW_DATA_DIR, f))
    print("✅ Cleared retraining data from 'data/new'.")

if __name__ == "__main__":
    retrain()
