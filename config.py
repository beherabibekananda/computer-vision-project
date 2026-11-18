import os
from pathlib import Path

# Base Paths (Local default with automatic Google Colab detection)
IN_COLAB = os.path.exists("/content")

if IN_COLAB:
    # Colab primary mount or fallback scratch path
    DRIVE_BASE = Path("/content/drive/MyDrive/SIH25099_Crop_Stress")
    LOCAL_DATA_DIR = Path("/content/dataset/raw")
    BASE_DIR = DRIVE_BASE if DRIVE_BASE.parent.exists() else LOCAL_DATA_DIR
else:
    PROJECT_ROOT = Path(__file__).resolve().parent
    BASE_DIR = PROJECT_ROOT / "data"

DATASET_ROOT = BASE_DIR / "extracted" / "Agriculture-Vision-2021"

# Specific Dataset Split Paths
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"

TRAIN_RGB_DIR = TRAIN_DIR / "images" / "rgb"
TRAIN_NIR_DIR = TRAIN_DIR / "images" / "nir"
TRAIN_LABEL_DIR = TRAIN_DIR / "labels" / "nutrient_deficiency"

VAL_RGB_DIR = VAL_DIR / "images" / "rgb"
VAL_NIR_DIR = VAL_DIR / "images" / "nir"
VAL_LABEL_DIR = VAL_DIR / "labels" / "nutrient_deficiency"

# Model Checkpoints & Output Directories
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

RGB_MODEL_PATH = CHECKPOINT_DIR / "unet_rgb_nutrient.keras"
RGB_NIR_MODEL_PATH = CHECKPOINT_DIR / "unet_rgb_nir_nutrient.keras"

# Training Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# S3 dataset URL if needed for automatic download
S3_DATASET_URL = "s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz"

