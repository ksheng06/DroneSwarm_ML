from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# Configuration
# ============================================================

TEST_DIR = Path("dataset/test")
MODEL_PATH = Path("model/best_fire_smoke_model.keras")
RESULTS_DIR = Path("results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Load unseen test dataset
# ============================================================

print("=" * 60)
print("AUTONOMOUS DRONE ML MODEL - FINAL TEST EVALUATION")
print("=" * 60)

print("\nLoading unseen test dataset...")

test_dataset = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# Load trained model
# ============================================================

print("\nLoading trained model...")

model = keras.models.load_model(MODEL_PATH)

print(f"Loaded: {MODEL_PATH}")


# ============================================================
# Obtain predictions
# ============================================================

print("\nRunning predictions on unseen test images...")

true_labels = []
predicted_labels = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    batch_predictions = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(labels.numpy())
    predicted_labels.extend(batch_predictions)


true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# ============================================================
# Accuracy
# ============================================================

test_accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:")
print(report)


# Save report
report_path = RESULTS_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AUTONOMOUS DRONE FIRE AND SMOKE DETECTION\n"
    )

    file.write("=" * 50)
    file.write("\n\n")

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    file.write("Classification Report:\n\n")

    file.write(report)

print(
    f"Classification report saved to: "
    f"{report_path}"
)


# ============================================================
# Confusion matrix
# ============================================================

matrix = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(matrix)


display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

figure, axis = plt.subplots(
    figsize=(8, 6)
)

display.plot(
    ax=axis,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title(
    "Fire, Neutral and Smoke Classification"
)

plt.tight_layout()

confusion_matrix_path = (
    RESULTS_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix saved to: "
    f"{confusion_matrix_path}"
)


# ============================================================
# Save raw confusion matrix
# ============================================================

matrix_text_path = (
    RESULTS_DIR / "confusion_matrix.txt"
)

with open(
    matrix_text_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Rows = Actual Class\n"
        "Columns = Predicted Class\n\n"
    )

    file.write(
        "Class order: "
        + ", ".join(class_names)
        + "\n\n"
    )

    file.write(str(matrix))


# ============================================================
# Complete
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION COMPLETE")
print("=" * 60)

print("\nGenerated evidence:")
print("1. results/classification_report.txt")
print("2. results/confusion_matrix.png")
print("3. results/confusion_matrix.txt")

print(
    "\nThese results were obtained from the unseen "
    "test dataset."
)