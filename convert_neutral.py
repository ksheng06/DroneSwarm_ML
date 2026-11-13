from pathlib import Path
from PIL import Image

NEUTRAL_DIR = Path("dataset/test/Neutral")

print("=" * 60)
print("CONVERTING NEUTRAL TEST IMAGES")
print("=" * 60)

converted = 0
failed = 0

for file_path in list(NEUTRAL_DIR.iterdir()):

    if not file_path.is_file():
        continue

    try:
        with Image.open(file_path) as image:

            # Convert to standard RGB
            image = image.convert("RGB")

            # Temporary JPEG path
            temp_path = file_path.with_name(
                file_path.stem + "_converted.jpg"
            )

            image.save(
                temp_path,
                format="JPEG",
                quality=95
            )

        # Delete original file
        file_path.unlink()

        # Rename converted file
        final_path = NEUTRAL_DIR / (
            file_path.stem + ".jpg"
        )

        temp_path.rename(final_path)

        converted += 1

    except Exception as error:

        print(f"FAILED: {file_path.name}")
        print(error)
        failed += 1


print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Converted: {converted}")
print(f"Failed: {failed}")