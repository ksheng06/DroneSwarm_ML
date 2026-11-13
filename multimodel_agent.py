from pathlib import Path
from datetime import datetime
import csv
import sys

import numpy as np
import ollama
from tensorflow import keras


# ============================================================
# CONFIGURATION
# ============================================================

MOBILENET_MODEL = Path("model/best_fire_smoke_model.keras")

QWEN_MODEL = "qwen3-vl:4b"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Fire",
    "Neutral",
    "Smoke"
]
INCIDENT_LOG = Path("results/incident_log.csv")

# ============================================================
# INCIDENT LOGGING
# ============================================================

def save_incident_log(
    image_path,
    mobile_prediction,
    mobile_confidence,
    mobile_probabilities,
    qwen_result,
    decision
):

    INCIDENT_LOG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = INCIDENT_LOG.exists()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    row = {
        "timestamp": timestamp,
        "image": str(image_path),

        "mobilenet_prediction":
            mobile_prediction,

        "mobilenet_confidence":
            f"{mobile_confidence:.2f}",

        "fire_probability":
            f"{mobile_probabilities['Fire']:.2f}",

        "neutral_probability":
            f"{mobile_probabilities['Neutral']:.2f}",

        "smoke_probability":
            f"{mobile_probabilities['Smoke']:.2f}",

        "qwen_fire":
            qwen_result["fire"],

        "qwen_smoke":
            qwen_result["smoke"],

        "qwen_classification":
            qwen_result["classification"],

        "qwen_hazard":
            qwen_result["hazard"],

        "qwen_evidence":
            qwen_result["evidence"],
            
        "risk_level":
            decision["risk_level"],

        "fusion_status":
            decision["status"],

        "final_action":
            decision["action"],

        "human_alert":
            decision["human_alert"],

        "reason":
            decision["reason"]
    }

    with open(
        INCIDENT_LOG,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(
        f"\nIncident recorded: {INCIDENT_LOG}"
    )

# ============================================================
# LOAD MOBILENETV2
# ============================================================

def load_mobilenet():

    if not MOBILENET_MODEL.exists():
        print("ERROR: MobileNetV2 model not found.")
        print(MOBILENET_MODEL)
        sys.exit(1)

    print("Loading MobileNetV2...")

    model = keras.models.load_model(
        MOBILENET_MODEL
    )

    print("MobileNetV2 loaded successfully.")

    return model


# ============================================================
# MOBILENETV2 PREDICTION
# ============================================================

def run_mobilenet(model, image_path):

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

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    ) * 100

    probabilities = {
        class_name: float(probability) * 100
        for class_name, probability
        in zip(CLASS_NAMES, predictions)
    }

    return (
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# QWEN3-VL VISUAL VERIFICATION
# ============================================================

def run_qwen(image_path):

    prompt = """
You are a secondary visual verifier for an autonomous
disaster-response drone.

Analyze ONLY what is visibly present in the image.

STRICT RULES:

1. Fire = YES only when visible flames or active burning
   are clearly visible.

2. Smoke = YES only when visible smoke is clearly present.

3. Do NOT infer fire merely because smoke is present.

4. Do NOT infer smoke merely because fire is present.

5. Do NOT infer the location, cause, pollution level,
   air quality, haze concentration, or people at risk.

6. You are advising an existing autonomous drone.
   Do not search for a drone inside the image.

7. Primary classification must be exactly one of:
   FIRE
   SMOKE
   NEUTRAL
   FIRE_AND_SMOKE

8. Drone action must be exactly one of:
   CONTINUE_MONITORING
   VERIFY_HAZARD
   ALERT_HUMAN_OPERATOR

Return EXACTLY these six lines and nothing else:

Fire: YES or NO
Smoke: YES or NO
Primary classification: FIRE, SMOKE, NEUTRAL, or FIRE_AND_SMOKE
Hazard level: LOW, MEDIUM, or HIGH
Visual evidence: one short sentence
Drone action: CONTINUE_MONITORING, VERIFY_HAZARD, or ALERT_HUMAN_OPERATOR
"""

    print("\nRunning Qwen3-VL visual verification...")

    response = ollama.chat(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)]
            }
        ],
        options={
            "num_ctx": 8192
        }
    )

    return response["message"]["content"]


# ============================================================
# PARSE QWEN RESULT
# ============================================================

def parse_qwen_response(response):

    result = {
        "fire": "UNKNOWN",
        "smoke": "UNKNOWN",
        "classification": "UNKNOWN",
        "hazard": "UNKNOWN",
        "evidence": "Not available",
        "action": "VERIFY_HAZARD"
    }

    for line in response.splitlines():

        line = line.strip()

        if line.lower().startswith("fire:"):
            result["fire"] = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

        elif line.lower().startswith("smoke:"):
            result["smoke"] = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

        elif line.lower().startswith(
            "primary classification:"
        ):
            result["classification"] = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

        elif line.lower().startswith(
            "hazard level:"
        ):
            result["hazard"] = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

        elif line.lower().startswith(
            "visual evidence:"
        ):
            result["evidence"] = (
                line.split(":", 1)[1]
                .strip()
            )

        elif line.lower().startswith(
            "drone action:"
        ):
            result["action"] = (
                line.split(":", 1)[1]
                .strip()
                .upper()
            )

    return result


# ============================================================
# NORMALISE QWEN FOR MODEL COMPARISON
# ============================================================

def qwen_hazard_state(qwen_result):

    classification = qwen_result[
        "classification"
    ]

    if classification == "NEUTRAL":
        return "NEUTRAL"

    if classification in {
        "FIRE",
        "SMOKE",
        "FIRE_AND_SMOKE"
    }:
        return "HAZARD"

    return "UNKNOWN"


def mobilenet_hazard_state(prediction):

    if prediction == "Neutral":
        return "NEUTRAL"

    if prediction in {
        "Fire",
        "Smoke"
    }:
        return "HAZARD"

    return "UNKNOWN"


# ============================================================
# FUSION / SAFETY POLICY
# ============================================================

def fusion_policy(
    mobilenet_prediction,
    mobilenet_confidence,
    qwen_result
):

    mobile_state = mobilenet_hazard_state(
        mobilenet_prediction
    )

    qwen_state = qwen_hazard_state(
        qwen_result
    )

    qwen_fire = qwen_result["fire"]
    qwen_smoke = qwen_result["smoke"]

    # ========================================================
    # LEVEL 4 - FAIL-SAFE
    # ========================================================

    if (
        mobile_state == "UNKNOWN"
        or qwen_state == "UNKNOWN"
    ):

        return {
            "risk_level": "LEVEL 4 - FAIL-SAFE",
            "status": "SYSTEM UNCERTAINTY",
            "action": "FAIL_SAFE_VERIFY",
            "human_alert": "REQUIRED",
            "reason": (
                "At least one model failed or returned "
                "an uncertain/unparseable result. "
                "The system defaults to human verification."
            )
        }

    # ========================================================
    # LEVEL 1 - MODEL CONFLICT
    # ========================================================

    if mobile_state != qwen_state:

        return {
            "risk_level": "LEVEL 1 - MODEL CONFLICT",
            "status": "MODEL DISAGREEMENT",
            "action": "VERIFY_HAZARD",
            "human_alert": "REQUIRED",
            "reason": (
                "MobileNetV2 and Qwen3-VL disagree. "
                "The area must not be declared safe "
                "until additional verification."
            )
        }

    # ========================================================
    # LEVEL 3 - FIRE HAZARD
    # ========================================================

    if (
        mobile_state == "HAZARD"
        and qwen_state == "HAZARD"
        and (
            mobilenet_prediction == "Fire"
            or qwen_fire == "YES"
        )
    ):

        return {
            "risk_level": "LEVEL 3 - FIRE HAZARD",
            "status": "FIRE HAZARD AGREEMENT",
            "action": "HIGH_PRIORITY_VERIFY",
            "human_alert": "REQUIRED",
            "reason": (
                "Both models support a hazardous event "
                "and visible fire evidence is present. "
                "High-priority verification and human "
                "review are required."
            )
        }

    # ========================================================
    # LEVEL 2 - SMOKE HAZARD
    # ========================================================

    if (
        mobile_state == "HAZARD"
        and qwen_state == "HAZARD"
        and (
            mobilenet_prediction == "Smoke"
            or qwen_smoke == "YES"
        )
    ):

        return {
            "risk_level": "LEVEL 2 - SMOKE HAZARD",
            "status": "SMOKE HAZARD AGREEMENT",
            "action": "VERIFY_HAZARD",
            "human_alert": "REQUIRED",
            "reason": (
                "Both models support a hazardous event "
                "with smoke evidence but without confirmed "
                "fire evidence requiring Level 3 response."
            )
        }

    # ========================================================
    # LEVEL 0 - NORMAL
    # ========================================================

    if (
        mobile_state == "NEUTRAL"
        and qwen_state == "NEUTRAL"
    ):

        # Extra protection if MobileNetV2 is uncertain
        if mobilenet_confidence < 60:

            return {
                "risk_level":
                    "LEVEL 1 - LOW CONFIDENCE",

                "status":
                    "LOW-CONFIDENCE AGREEMENT",

                "action":
                    "VERIFY_HAZARD",

                "human_alert":
                    "RECOMMENDED",

                "reason": (
                    "Both models indicate Neutral, "
                    "but MobileNetV2 confidence is below "
                    "the prototype 60% confidence threshold."
                )
            }

        return {
            "risk_level": "LEVEL 0 - NORMAL",
            "status": "NEUTRAL AGREEMENT",
            "action": "CONTINUE_MONITORING",
            "human_alert": "NOT REQUIRED",
            "reason": (
                "Both models found no immediate visible "
                "fire or smoke evidence."
            )
        }

    # ========================================================
    # FINAL FAIL-SAFE
    # ========================================================

    return {
        "risk_level": "LEVEL 4 - FAIL-SAFE",
        "status": "UNHANDLED STATE",
        "action": "FAIL_SAFE_VERIFY",
        "human_alert": "REQUIRED",
        "reason": (
            "The fusion policy encountered an unexpected "
            "combination of model outputs."
        )
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    
    print("=" * 70)
    print("MULTI-MODEL AUTONOMOUS DRONE HAZARD SYSTEM")
    print("=" * 70)

    print(
        "\nPrimary model: MobileNetV2"
    )

    print(
        "Secondary verifier: Qwen3-VL 4B"
    )

    model = load_mobilenet()


    # ============================================================
    # GET IMAGE
    # ============================================================

    print("\nEnter an image path.")

    image_input = input(
        "Image path: "
    ).strip().strip('"')

    image_path = Path(image_input)


    if not image_path.exists():

        print("\nERROR: Image does not exist.")
        print(image_path)

        sys.exit(1)


    # ============================================================
    # RUN MOBILENET
    # ============================================================

    print("\nRunning MobileNetV2...")

    (
        mobile_prediction,
        mobile_confidence,
        mobile_probabilities
    ) = run_mobilenet(
        model,
        image_path
    )


    # ============================================================
    # RUN QWEN
    # ============================================================

    try:

        qwen_response = run_qwen(
            image_path
        )

        qwen_result = parse_qwen_response(
            qwen_response
        )

    except Exception as error:

        print("\nQwen3-VL ERROR:")
        print(error)

        qwen_result = {
            "fire": "UNKNOWN",
            "smoke": "UNKNOWN",
            "classification": "UNKNOWN",
            "hazard": "UNKNOWN",
            "evidence": "Qwen3-VL unavailable",
            "action": "VERIFY_HAZARD"
        }


    # ============================================================
    # FUSE RESULTS
    # ============================================================

    decision = fusion_policy(
        mobile_prediction,
        mobile_confidence,
        qwen_result
    )



    # ============================================================
    # DISPLAY RESULTS
    # ============================================================

    print("\n" + "=" * 70)
    print("MOBILENETV2 RESULT")
    print("=" * 70)

    print(
        f"Prediction: {mobile_prediction}"
    )

    print(
        f"Confidence: {mobile_confidence:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in (
        mobile_probabilities.items()
    ):

        print(
            f"  {class_name}: "
            f"{probability:.2f}%"
        )


    print("\n" + "=" * 70)
    print("QWEN3-VL SECONDARY VERIFICATION")
    print("=" * 70)

    print(
        f"Fire: {qwen_result['fire']}"
    )

    print(
        f"Smoke: {qwen_result['smoke']}"
    )

    print(
        "Classification: "
        f"{qwen_result['classification']}"
    )

    print(
        f"Hazard level: "
        f"{qwen_result['hazard']}"
    )

    print(
        f"Visual evidence: "
        f"{qwen_result['evidence']}"
    )

    print(
        f"Suggested action: "
        f"{qwen_result['action']}"
    )


    print("\n" + "=" * 70)
    print("MULTI-MODEL FUSION")
    print("=" * 70)
    
    print(
        f"Risk level: {decision['risk_level']}"
    )

    print(
        f"Status: {decision['status']}"
    )

    print(
        f"Final action: {decision['action']}"
    )

    print(
        f"Human alert: "
        f"{decision['human_alert']}"
    )

    print(
        f"Reason: {decision['reason']}"
    )


    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    save_incident_log(
        image_path,
        mobile_prediction,
        mobile_confidence,
        mobile_probabilities,
        qwen_result,
        decision
    )
    
    
if __name__ == "__main__":
    main()