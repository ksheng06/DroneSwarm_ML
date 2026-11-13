from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# Configuration
# ============================================================

TRAIN_DIR = Path("dataset/train")
VALIDATION_DIR = Path("dataset/validation")

MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
SEED = 42

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Load datasets
# ============================================================

print("=" * 60)
print("AUTONOMOUS DRONE FIRE AND SMOKE DETECTION")
print("=" * 60)

print("\nLoading training dataset...")

train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

validation_dataset = keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nDetected classes:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# Improve input pipeline performance
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)


# ============================================================
# MobileNetV2 transfer learning
# ============================================================

print("\nLoading MobileNetV2...")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained MobileNetV2 layers
base_model.trainable = False


# ============================================================
# Build classification model
# ============================================================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()


# ============================================================
# Training callbacks
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="model/best_fire_smoke_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# ============================================================
# Train
# ============================================================

print("\n" + "=" * 60)
print("STARTING MODEL TRAINING")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ============================================================
# Save final model
# ============================================================

model.save("model/fire_smoke_detector.keras")

print("\nModel saved successfully:")
print("model/fire_smoke_detector.keras")


# ============================================================
# Plot training results
# ============================================================

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

epochs_completed = range(
    1,
    len(training_accuracy) + 1
)


# Accuracy graph
plt.figure()

plt.plot(
    epochs_completed,
    training_accuracy,
    label="Training Accuracy"
)

plt.plot(
    epochs_completed,
    validation_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig(
    "results/training_accuracy.png",
    bbox_inches="tight"
)

plt.close()


# Loss graph
plt.figure()

plt.plot(
    epochs_completed,
    training_loss,
    label="Training Loss"
)

plt.plot(
    epochs_completed,
    validation_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.savefig(
    "results/training_loss.png",
    bbox_inches="tight"
)

plt.close()


print("\nTraining graphs saved:")
print("results/training_accuracy.png")
print("results/training_loss.png")


# ============================================================
# Final validation results
# ============================================================

validation_loss_result, validation_accuracy_result = model.evaluate(
    validation_dataset,
    verbose=0
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Final validation accuracy: "
    f"{validation_accuracy_result * 100:.2f}%"
)

print(
    f"Final validation loss: "
    f"{validation_loss_result:.4f}"
)

print("\nNext step: evaluate the model using the unseen test set.")