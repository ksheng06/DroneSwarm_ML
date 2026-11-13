from pathlib import Path
import tensorflow as tf

DATASET_DIR = Path("dataset")

folders = [
    DATASET_DIR / "train",
    DATASET_DIR / "validation",
    DATASET_DIR / "test"
]

invalid_files = []

print("=" * 60)
print("TENSORFLOW IMAGE COMPATIBILITY CHECK")
print("=" * 60)

for dataset_folder in folders:

    print(f"\nChecking: {dataset_folder}")

    for class_folder in dataset_folder.iterdir():

        if not class_folder.is_dir():
            continue

        print(f"  Class: {class_folder.name}")

        for file_path in class_folder.iterdir():

            if not file_path.is_file():
                continue

            try:
                image_bytes = tf.io.read_file(str(file_path))

                # Uses TensorFlow's decoder rather than Pillow
                image = tf.io.decode_image(
                    image_bytes,
                    channels=3,
                    expand_animations=False
                )

                # Force TensorFlow to actually evaluate it
                _ = image.numpy()

            except Exception as error:

                print("\nINVALID FOR TENSORFLOW:")
                print(file_path)
                print(f"Extension: {file_path.suffix}")
                print(f"Reason: {error}")

                invalid_files.append(file_path)


print("\n" + "=" * 60)

if invalid_files:

    print(
        f"Found {len(invalid_files)} file(s) "
        "that TensorFlow cannot decode."
    )

    print("\nRemoving incompatible files...")

    for file_path in invalid_files:

        try:
            file_path.unlink()
            print(f"Removed: {file_path}")

        except Exception as error:
            print(f"Could not remove: {file_path}")
            print(error)

    print("\nIncompatible files removed successfully.")

else:

    print("All images are compatible with TensorFlow.")

print("=" * 60)