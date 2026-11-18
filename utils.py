import os
import shutil
import subprocess
from pathlib import Path
import config


def verify_dataset(
    dataset_root: Path = config.DATASET_ROOT,
    auto_download: bool = False
) -> bool:
    """Checks whether the Agriculture-Vision dataset is present and properly structured.

    Returns True if valid dataset is found, False otherwise.
    """
    print(f"Checking dataset root: {dataset_root}")

    required_dirs = [
        config.TRAIN_RGB_DIR,
        config.TRAIN_NIR_DIR,
        config.TRAIN_LABEL_DIR,
        config.VAL_RGB_DIR,
        config.VAL_NIR_DIR,
        config.VAL_LABEL_DIR
    ]

    all_exist = all(d.exists() for d in required_dirs)

    if all_exist:
        train_count = len(os.listdir(config.TRAIN_RGB_DIR))
        val_count = len(os.listdir(config.VAL_RGB_DIR))
        print(f"Dataset verified successfully!")
        print(f"   - Training RGB images: {train_count}")
        print(f"   - Validation RGB images: {val_count}")
        return True

    print("Dataset directory structure incomplete or not found.")

    # Search for alternative locations in Colab Drive or current parent folders
    potential_locations = [
        Path("/content/drive/MyDrive/SIH25099_Crop_Stress/dataset/raw/extracted/Agriculture-Vision-2021"),
        Path("/content/dataset/raw/extracted/Agriculture-Vision-2021"),
        Path.cwd() / "data" / "extracted" / "Agriculture-Vision-2021"
    ]

    for loc in potential_locations:
        if loc.exists() and (loc / "train" / "images" / "rgb").exists():
            print(f"Found existing dataset at: {loc}")
            config.DATASET_ROOT = loc
            config.TRAIN_DIR = loc / "train"
            config.VAL_DIR = loc / "val"
            config.TRAIN_RGB_DIR = config.TRAIN_DIR / "images" / "rgb"
            config.TRAIN_NIR_DIR = config.TRAIN_DIR / "images" / "nir"
            config.TRAIN_LABEL_DIR = config.TRAIN_DIR / "labels" / "nutrient_deficiency"
            config.VAL_RGB_DIR = config.VAL_DIR / "images" / "rgb"
            config.VAL_NIR_DIR = config.VAL_DIR / "images" / "nir"
            config.VAL_LABEL_DIR = config.VAL_DIR / "labels" / "nutrient_deficiency"
            return True

    if auto_download:
        return download_and_extract_dataset()

    print("\nPlease download the dataset or set the correct path in config.py.")
    return False


def download_and_extract_dataset(target_dir: Path = config.BASE_DIR) -> bool:
    """Automates AWS S3 CLI download of the Agriculture-Vision-2021 dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "Agriculture-Vision-2021.tar.gz"
    extracted_path = target_dir / "extracted"

    if not archive_path.exists():
        print(f"Downloading from {config.S3_DATASET_URL}...")
        cmd = [
            "aws", "s3", "cp",
            config.S3_DATASET_URL,
            str(archive_path),
            "--no-sign-request"
        ]
        try:
            subprocess.run(cmd, check=True)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return False

    print(f"Extracting {archive_path} to {extracted_path}...")
    extracted_path.mkdir(parents=True, exist_ok=True)
    try:
        shutil.unpack_archive(str(archive_path), str(extracted_path))
        print("Extraction complete.")
        return verify_dataset(extracted_path / "Agriculture-Vision-2021")
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

