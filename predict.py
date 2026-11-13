from pathlib import Path
import sys

import numpy as np
from tensorflow import keras


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path("model/best_fire_smoke_model.keras")

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Fire",
    "Neutral",
    "Smoke"
]


# ============================================================
# Drone agent decision logic
# ============================================================

def agent_decision(prediction, confidence):

    if prediction == "Fire":

        hazard_status = "HIGH-RISK HAZARD"

        action = (
            "Fire detected. Send another drone to verify "
            "the location and notify the human operator."
        )

        human_action = (
            "Human verification required before "
            "emergency escalation."
        )

    elif prediction == "Smoke":

        hazard_status = "POTENTIAL HAZARD"

        action = (
            "Smoke detected. Deploy a verification drone "
            "and continue monitoring the affected area."
        )

        human_action = (
            "Notify human operator and provide image "
            "evidence and confidence score."
        )

    else:

        hazard_status = "NO IMMEDIATE HAZARD"

        action = (
            "No fire or smoke detected. "
            "Continue autonomous monitoring."
        )

        human_action = (
            "No immediate human intervention required."
        )


    # Confidence-based uncertainty handling
    if confidence < 60:

        hazard_status = "UNCERTAIN"

        action = (
            "Prediction confidence is low. "
            "Request another drone observation before "
            "making a decision."
        )

        human_action = (
            "Human review recommended because the "
            "ML prediction is uncertain."
        )


    return hazard_status, action, human_action


# ============================================================
# Load model
# ============================================================

print("=" * 65)
print("AUTONOMOUS DRONE DISASTER RESPONSE SYSTEM")
print("=" * 65)

print("\nLoading trained ML model...")

if not MODEL_PATH.exists():

    print("\nERROR: Trained model could not be found.")
    print(f"Expected model: {MODEL_PATH}")

    sys.exit(1)


model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Get image path
# ============================================================

print("\nEnter the path of an image.")

print(
    "Example: dataset/test/Fire/fire1.jpg"
)

image_input = input(
    "\nImage path: "
).strip().strip('"')


image_path = Path(image_input)


if not image_path.exists():

    print("\nERROR: Image does not exist.")
    print(image_path)

    sys.exit(1)


# ============================================================
# Prepare image
# ============================================================

try:

    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

except Exception as error:

    print("\nERROR: Unable to open image.")
    print(error)

    sys.exit(1)


# ============================================================
# ML prediction
# ============================================================

prediction_scores = model.predict(
    image_array,
    verbose=0
)[0]

predicted_index = np.argmax(
    prediction_scores
)

prediction = CLASS_NAMES[
    predicted_index
]

confidence = float(
    prediction_scores[predicted_index]
) * 100


# ============================================================
# Autonomous agent policy
# ============================================================

hazard_status, action, human_action = (
    agent_decision(
        prediction,
        confidence
    )
)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 65)
print("DRONE ML ANALYSIS")
print("=" * 65)

print(f"\nImage: {image_path}")

print("\nML CLASSIFICATION")
print("-" * 30)

print(
    f"Prediction: {prediction}"
)

print(
    f"Confidence: {confidence:.2f}%"
)


print("\nCLASS PROBABILITIES")
print("-" * 30)

for class_name, probability in zip(
    CLASS_NAMES,
    prediction_scores
):

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )


print("\nAUTONOMOUS AGENT ASSESSMENT")
print("-" * 30)

print(
    f"Hazard Status: {hazard_status}"
)


print("\nAGENT ACTION")
print("-" * 30)

print(action)


print("\nHUMAN-AGENT INTERACTION")
print("-" * 30)

print(human_action)


print("\n" + "=" * 65)
print("ANALYSIS COMPLETE")
print("=" * 65)