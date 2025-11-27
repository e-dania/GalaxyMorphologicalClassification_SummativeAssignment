import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Paths
PROCESSED_DIR = "data/processed"
PREDICTIONS_FILE = os.path.join(PROCESSED_DIR, "predictions_random_subset.csv")
TEST_LABELS_FILE = "data/raw/training_solutions_rev1.csv"  # CSV with GalaxyID and true class (must exist)

# Load predictions
pred_df = pd.read_csv(PREDICTIONS_FILE)
print("✅ Predictions loaded:", pred_df.shape)

# Load true labels
true_df = pd.read_csv(TEST_LABELS_FILE)
print("✅ True labels loaded:", true_df.shape)

# Merge predictions with true labels
df = pd.merge(pred_df, true_df, on="GalaxyID")
print("✅ Merged predictions with true labels:", df.shape)

# Predicted class = argmax of probabilities
prob_cols = ["class_0", "class_1", "class_2"]
df["pred_class"] = df[prob_cols].idxmax(axis=1).apply(lambda x: int(x.split("_")[1]))

# True labels (assuming 'true_class' column exists)
y_true = df["true_class"].values
y_pred = df["pred_class"].values

# Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"Accuracy: {acc:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature visualizations
# 1. Distribution of predicted classes
plt.figure(figsize=(6,4))
sns.countplot(x="pred_class", data=df)
plt.title("Predicted Class Distribution")
plt.show()

# 2. Distribution of true classes
plt.figure(figsize=(6,4))
sns.countplot(x="true_class", data=df)
plt.title("True Class Distribution")
plt.show()

# 3. Sample images with predictions
from PIL import Image
TEST_DIR = "data/raw/images_test_rev1"
sample = df.sample(9)
plt.figure(figsize=(10,10))
for i, row in enumerate(sample.itertuples()):
    img_path = os.path.join(TEST_DIR, f"{row.GalaxyID}.jpg")
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB").resize((128,128))
        plt.subplot(3,3,i+1)
        plt.imshow(img)
        plt.title(f"T:{row.true_class} / P:{row.pred_class}")
        plt.axis("off")
plt.tight_layout()
plt.show()
