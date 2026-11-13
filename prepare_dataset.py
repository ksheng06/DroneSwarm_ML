from pathlib import Path
import random
import shutil

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DIR = Path("dataset/train")
VALIDATION_DIR = Path("dataset/validation")

CLASSES = ["Fire", "Neutral", "Smoke"]

# Move 20% of the current training images into validation
VALIDATION_SPLIT = 0.20

# Fixed seed makes the split reproducible
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def create_validation_set():
    print("=" * 50)
    print("Creating validation dataset")
    print("=" * 50)

    for class_name in CLASSES:

        source_folder = TRAIN_DIR / class_name
        validation_folder = VALIDATION_DIR / class_name

        # Create validation class folder
        validation_folder.mkdir(parents=True, exist_ok=True)

        # Find image files
        image_files = [
            file
            for file in source_folder.iterdir()
            if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]

        # Shuffle images
        random.shuffle(image_files)

        number_to_move = int(len(image_files) * VALIDATION_SPLIT)

        validation_images = image_files[:number_to_move]

        print(f"\nClass: {class_name}")
        print(f"Original training images: {len(image_files)}")
        print(f"Moving to validation: {number_to_move}")

        for image_path in validation_images:

            destination = validation_folder / image_path.name

            shutil.move(str(image_path), str(destination))

        remaining_images = len(list(source_folder.iterdir()))

        print(f"Remaining training images: {remaining_images}")

    print("\n" + "=" * 50)
    print("Validation dataset created successfully.")
    print("=" * 50)


if __name__ == "__main__":
    create_validation_set()