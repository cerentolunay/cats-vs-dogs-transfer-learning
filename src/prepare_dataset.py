import os
import random
import shutil
from pathlib import Path
from PIL import Image


# =========================
# Configuration
# =========================

RAW_DATA_DIR = Path("raw_data/PetImages")
CAT_DIR = RAW_DATA_DIR / "Cat"
DOG_DIR = RAW_DATA_DIR / "Dog"

OUTPUT_DIR = Path("dataset")

TRAIN_COUNT_PER_CLASS = 1000
VAL_COUNT_PER_CLASS = 200
TEST_COUNT_PER_CLASS = 200

RANDOM_SEED = 42
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def create_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def is_valid_image(image_path):
    """
    Bozuk görselleri elemek için kontrol yapar.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_valid_images(source_dir):
    """
    Kaynak klasörden geçerli görselleri alır.
    """
    images = []

    for file in source_dir.iterdir():
        if file.suffix.lower() in VALID_EXTENSIONS:
            if is_valid_image(file):
                images.append(file)

    return images


def copy_images(image_list, destination_dir):
    """
    Görselleri hedef klasöre kopyalar.
    """
    create_dir(destination_dir)

    for image_path in image_list:
        shutil.copy2(image_path, destination_dir / image_path.name)


def prepare_class_dataset(source_dir, class_name):
    """
    Bir sınıf için train/val/test ayrımı yapar.
    """
    print(f"\nProcessing class: {class_name}")
    print(f"Source directory: {source_dir}")

    images = get_valid_images(source_dir)
    random.shuffle(images)

    required_count = TRAIN_COUNT_PER_CLASS + VAL_COUNT_PER_CLASS + TEST_COUNT_PER_CLASS

    if len(images) < required_count:
        raise ValueError(
            f"{class_name} için yeterli görsel yok. "
            f"Gerekli: {required_count}, Bulunan: {len(images)}"
        )

    train_images = images[:TRAIN_COUNT_PER_CLASS]
    val_images = images[TRAIN_COUNT_PER_CLASS:TRAIN_COUNT_PER_CLASS + VAL_COUNT_PER_CLASS]
    test_images = images[
        TRAIN_COUNT_PER_CLASS + VAL_COUNT_PER_CLASS:
        TRAIN_COUNT_PER_CLASS + VAL_COUNT_PER_CLASS + TEST_COUNT_PER_CLASS
    ]

    copy_images(train_images, OUTPUT_DIR / "train" / class_name)
    copy_images(val_images, OUTPUT_DIR / "val" / class_name)
    copy_images(test_images, OUTPUT_DIR / "test" / class_name)

    print(f"Train {class_name}: {len(train_images)}")
    print(f"Val   {class_name}: {len(val_images)}")
    print(f"Test  {class_name}: {len(test_images)}")


def main():
    random.seed(RANDOM_SEED)

    if not CAT_DIR.exists():
        raise FileNotFoundError(f"Cat klasörü bulunamadı: {CAT_DIR}")

    if not DOG_DIR.exists():
        raise FileNotFoundError(f"Dog klasörü bulunamadı: {DOG_DIR}")

    prepare_class_dataset(CAT_DIR, "cats")
    prepare_class_dataset(DOG_DIR, "dogs")

    print("\nDataset preparation completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()