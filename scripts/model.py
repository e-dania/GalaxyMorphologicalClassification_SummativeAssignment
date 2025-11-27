import os
import numpy as np
import tensorflow as tf

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 3
EPOCHS = 5
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

# Load MobileNetV2 base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze the base

# Add custom head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Get batch files
X_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("X_batch")])
y_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("y_batch")])

# Training loop over batches
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
    for X_file, y_file in zip(X_files, y_files):
        X_path = os.path.join(PROCESSED_DIR, X_file)
        y_path = os.path.join(PROCESSED_DIR, y_file)

        print(f"Loading batch: {X_file}")
        X = np.load(X_path)
        y = np.load(y_path)

        model.fit(X, y, epochs=1, batch_size=BATCH_SIZE, shuffle=True)

# Save the trained model
model_path = os.path.join(MODEL_DIR, "galaxy_mobilenet.h5")
model.save(model_path)
print(f"\n✅ Model saved to {model_path}")
