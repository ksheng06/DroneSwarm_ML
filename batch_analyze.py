from pathlib import Path
from datetime import datetime
import csv
import sys

from multimodel_agent import (
    load_mobilenet,
    run_mobilenet,
    run_qwen,
    parse_qwen_response,
    fusion_policy,
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

CSV_FIELDS = [
    "image",
    "mobilenet_prediction",
    "mobilenet_confidence",
    "fire_probability",
    "neutral_probability",
    "smoke_probability",
    "qwen_fire",
    "qwen_smoke",
    "qwen_classification",
    "qwen_hazard",
    "qwen_evidence",
    "risk_level",
    "fusion_status",
    "final_action",
    "human_alert",
    "reason",
]


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(folder):

    images = []

    for file_path in folder.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):
            images.append(file_path)

    return sorted(images)


# ============================================================
# CREATE BATCH FILE
# ============================================================

def create_batch_file(folder):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    folder_name = folder.name.replace(
        " ",
        "_"
    )

    return RESULTS_DIR / (
        f"batch_{folder_name}_{timestamp}.csv"
    )


# ============================================================
# INITIALISE CSV
# ============================================================

def initialise_csv(output_file):

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writeheader()


# ============================================================
# APPEND ONE RESULT IMMEDIATELY
# ============================================================

def append_result(
    output_file,
    row
):

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-MODEL BATCH IMAGE ANALYSIS")
    print("=" * 70)

    print("\nPrimary model: MobileNetV2")
    print("Secondary verifier: Qwen3-VL 4B")

    print(
        "\nEnter the folder containing "
        "images to analyse."
    )

    folder_input = input(
        "Folder path: "
    ).strip().strip('"')

    folder = Path(folder_input)

    if not folder.exists():

        print("\nERROR: Folder does not exist.")
        print(folder)

        sys.exit(1)

    if not folder.is_dir():

        print(
            "\nERROR: The supplied path "
            "is not a folder."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    images = find_images(folder)

    if not images:

        print(
            "\nNo supported images were found."
        )

        sys.exit(0)

    total_images = len(images)

    print(
        f"\nImages found: {total_images}"
    )


    # --------------------------------------------------------
    # CREATE OUTPUT FILE BEFORE ANALYSIS
    # --------------------------------------------------------

    output_file = create_batch_file(
        folder
    )

    initialise_csv(
        output_file
    )

    print(
        "\nBatch file created:"
    )

    print(output_file)

    print(
        "\nEach completed image will be "
        "saved immediately."
    )


    # --------------------------------------------------------
    # LOAD MODEL ONCE
    # --------------------------------------------------------

    print(
        "\nLoading MobileNetV2 once "
        "for the entire batch..."
    )

    model = load_mobilenet()


    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    completed = 0

    level_0 = 0
    level_1 = 0
    level_2 = 0
    level_3 = 0
    level_4 = 0


    # --------------------------------------------------------
    # PROCESS BATCH
    # --------------------------------------------------------

    try:

        for index, image_path in enumerate(
            images,
            start=1
        ):

            print(
                "\n" + "=" * 70
            )

            print(
                f"[{index}/{total_images}] "
                f"Processing: {image_path.name}"
            )

            print(
                "=" * 70
            )


            # ----------------------------------------------
            # MOBILENETV2
            # ----------------------------------------------

            print(
                "Running MobileNetV2..."
            )

            try:

                (
                    mobile_prediction,
                    mobile_confidence,
                    mobile_probabilities
                ) = run_mobilenet(
                    model,
                    image_path
                )

            except Exception as error:

                print(
                    "MobileNetV2 ERROR:"
                )

                print(error)

                continue


            # ----------------------------------------------
            # QWEN3-VL
            # ----------------------------------------------

            print(
                "Running Qwen3-VL..."
            )

            try:

                qwen_response = run_qwen(
                    image_path
                )

                qwen_result = (
                    parse_qwen_response(
                        qwen_response
                    )
                )

            except Exception as error:

                print(
                    "Qwen3-VL ERROR:"
                )

                print(error)

                qwen_result = {
                    "fire": "UNKNOWN",
                    "smoke": "UNKNOWN",
                    "classification": "UNKNOWN",
                    "hazard": "UNKNOWN",
                    "evidence":
                        "Qwen3-VL unavailable",
                    "action":
                        "VERIFY_HAZARD",
                }


            # ----------------------------------------------
            # FUSION
            # ----------------------------------------------

            decision = fusion_policy(
                mobile_prediction,
                mobile_confidence,
                qwen_result
            )


            # ----------------------------------------------
            # SAVE RESULT IMMEDIATELY
            # ----------------------------------------------

            row = {
                "image":
                    str(image_path),

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
                    decision["reason"],
            }

            append_result(
                output_file,
                row
            )

            completed += 1


            # ----------------------------------------------
            # UPDATE SUMMARY COUNTERS
            # ----------------------------------------------

            risk_level = decision["risk_level"]

            if risk_level.startswith("LEVEL 0"):
                level_0 += 1

            elif risk_level.startswith("LEVEL 1"):
                level_1 += 1

            elif risk_level.startswith("LEVEL 2"):
                level_2 += 1

            elif risk_level.startswith("LEVEL 3"):
                level_3 += 1

            elif risk_level.startswith("LEVEL 4"):
                level_4 += 1


            # ----------------------------------------------
            # DISPLAY
            # ----------------------------------------------

            print(
                f"\nMobileNetV2: "
                f"{mobile_prediction} "
                f"({mobile_confidence:.2f}%)"
            )

            print(
                "Qwen3-VL: "
                f"{qwen_result['classification']}"
            )

            print(
                "Fusion: "
                f"{decision['status']}"
            )
            
            print(
                "Risk level: "
                f"{decision['risk_level']}"
            )

            print(
                "Final action: "
                f"{decision['action']}"
            )

            print(
                "Human alert: "
                f"{decision['human_alert']}"
            )

            print(
                f"Result saved to: {output_file}"
            )


    # --------------------------------------------------------
    # SAFE CTRL+C
    # --------------------------------------------------------

    except KeyboardInterrupt:

        print(
            "\n\n" + "=" * 70
        )

        print(
            "BATCH INTERRUPTED BY USER"
        )

        print(
            "=" * 70
        )

        print(
            f"\nCompleted: "
            f"{completed}/{total_images}"
        )

        print(
            "\nCompleted results remain saved in:"
        )

        print(
            output_file
        )

        print(
            "\nMaster incident_log.csv "
            "was NOT modified."
        )

        return


    # --------------------------------------------------------
    # NORMAL COMPLETION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BATCH ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImages discovered: "
        f"{total_images}"
    )

    print(
        f"Images successfully analysed: "
        f"{completed}"
    )

    print(
        f"\nLevel 0 - Normal: "
        f"{level_0}"
    )

    print(
        f"Level 1 - Conflict / Low confidence: "
        f"{level_1}"
    )

    print(
        f"Level 2 - Smoke hazard: "
        f"{level_2}"
    )

    print(
        f"Level 3 - Fire hazard: "
        f"{level_3}"
    )

    print(
        f"Level 4 - Fail-safe: "
        f"{level_4}"
    )
    
    print(
        "\nBatch results saved to:"
    )

    print(
        output_file
    )

    print(
        "\nMaster incident_log.csv "
        "was NOT modified."
    )


if __name__ == "__main__":
    main()