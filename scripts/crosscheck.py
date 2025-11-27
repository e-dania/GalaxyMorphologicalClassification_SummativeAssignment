import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Paths
PROCESSED_DIR = "data/processed"
TEST_DIR = "data/raw/images_test_rev1"
PREDICTIONS_CSV = os.path.join(PROCESSED_DIR, "predictions_random_subset.csv")

# Load predictions
df = pd.read_csv(PREDICTIONS_CSV)

# Convert probabilities to class labels (argmax)
class_cols = ["class_0", "class_1", "class_2"]
df["predicted_class"] = df[class_cols].idxmax(axis=1)

# Count of each predicted class
counts = df["predicted_class"].value_counts()
print("Predicted class distribution:")
print(counts)

# Optional: Display a few sample images with predicted labels
NUM_SAMPLES = 5
sample_df = df.sample(NUM_SAMPLES, random_state=42)

plt.figure(figsize=(15, 3))
for i, row in enumerate(sample_df.itertuples(), 1):
    img_path = os.path.join(TEST_DIR, f"{row.GalaxyID}.jpg")
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    plt.subplot(1, NUM_SAMPLES, i)
    plt.imshow(img)
    plt.title(row.predicted_class)
    plt.axis("off")

plt.show()
