from tensorflow.keras.models import load_model
import json
import os

# Path to your model
MODEL_PATH = "./models/DenseNet121_Augmented_FINAL.keras"
CLASS_FILE = "./models/class_indices.json"  # optional file

# Load the model
model = load_model(MODEL_PATH)

# Get number of output classes
output_shape = model.output_shape
num_classes = output_shape[-1]
print(f"\n🔹 Number of classes detected: {num_classes}")

# Try to load class labels if available
if os.path.exists(CLASS_FILE):
    with open(CLASS_FILE, "r") as f:
        class_indices = json.load(f)
    print("\n🔹 Class labels (from file):")
    # Sort by value to preserve correct order
    for label, idx in sorted(class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
else:
    print("\n⚠️ No class index file found.")
    print("You can create one using your ImageDataGenerator's `class_indices` when retraining.")
