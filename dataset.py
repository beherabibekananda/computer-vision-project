import os
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf

import config


def get_image_files(rgb_dir: Path) -> List[str]:
    """Return sorted list of image filenames in rgb_dir."""
    if not rgb_dir.exists():
        return []
    return sorted([
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def find_positive_negative_files(
    filenames: List[str],
    label_dir: Path,
    sample_limit: int = None
) -> Tuple[List[str], List[str]]:
    """Separates files into positive (has nutrient deficiency mask pixels) and negative."""
    positive_files = []
    negative_files = []

    files_to_check = filenames if sample_limit is None else filenames[:sample_limit]

    for fname in files_to_check:
        stem = Path(fname).stem
        mask_path = label_dir / f"{stem}.png"
        if not mask_path.exists():
            continue

        mask = np.array(Image.open(mask_path))
        if np.count_nonzero(mask) > 0:
            positive_files.append(fname)
        else:
            negative_files.append(fname)

    return positive_files, negative_files


def create_balanced_filelist(
    positive_files: List[str],
    negative_files: List[str],
    samples_per_class: int = 512,
    seed: int = config.RANDOM_SEED
) -> List[str]:
    """Sample an equal number of positive and negative examples."""
    random.seed(seed)
    n_pos = min(len(positive_files), samples_per_class)
    n_neg = min(len(negative_files), samples_per_class)

    sel_pos = random.sample(positive_files, n_pos) if len(positive_files) >= n_pos else positive_files
    sel_neg = random.sample(negative_files, n_neg) if len(negative_files) >= n_neg else negative_files

    balanced = sel_pos + sel_neg
    random.shuffle(balanced)
    return balanced


def load_rgb_and_mask(
    filename: tf.Tensor,
    rgb_dir: str,
    label_dir: str,
    img_size: int = config.IMG_SIZE
):
    """Load and preprocess 3-channel RGB image and corresponding binary mask."""
    filename = tf.convert_to_tensor(filename, dtype=tf.string)

    # RGB image
    rgb_path = tf.strings.join([str(rgb_dir), "/", filename])
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_jpeg(rgb, channels=3)
    rgb = tf.image.resize(rgb, [img_size, img_size])
    rgb = tf.cast(rgb, tf.float32) / 255.0

    # Mask
    mask_name = tf.strings.regex_replace(filename, r"\.(jpg|jpeg|png)$", ".png")
    mask_path = tf.strings.join([str(label_dir), "/", mask_name])
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [img_size, img_size], method="nearest")
    mask = tf.cast(mask > 127, tf.float32)

    return rgb, mask


def load_rgb_nir_and_mask(
    filename: tf.Tensor,
    rgb_dir: str,
    nir_dir: str,
    label_dir: str,
    img_size: int = config.IMG_SIZE
):
    """Load and combine RGB (3 channels) + NIR (1 channel) into 4 channels, and binary mask."""
    filename = tf.convert_to_tensor(filename, dtype=tf.string)

    # RGB
    rgb_path = tf.strings.join([str(rgb_dir), "/", filename])
    rgb = tf.io.read_file(rgb_path)
    rgb = tf.image.decode_jpeg(rgb, channels=3)
    rgb = tf.image.resize(rgb, [img_size, img_size])
    rgb = tf.cast(rgb, tf.float32) / 255.0

    # NIR
    nir_path = tf.strings.join([str(nir_dir), "/", filename])
    nir = tf.io.read_file(nir_path)
    nir = tf.image.decode_jpeg(nir, channels=1)
    nir = tf.image.resize(nir, [img_size, img_size])
    nir = tf.cast(nir, tf.float32) / 255.0

    # 4 channels: (R, G, B, NIR)
    combined = tf.concat([rgb, nir], axis=-1)

    # Mask
    mask_name = tf.strings.regex_replace(filename, r"\.(jpg|jpeg|png)$", ".png")
    mask_path = tf.strings.join([str(label_dir), "/", mask_name])
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [img_size, img_size], method="nearest")
    mask = tf.cast(mask > 127, tf.float32)

    return combined, mask


def make_dataset(
    files: List[str],
    rgb_dir: Path,
    label_dir: Path,
    nir_dir: Path = None,
    batch_size: int = config.BATCH_SIZE,
    training: bool = False,
    img_size: int = config.IMG_SIZE
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for either RGB (3-channel) or RGB+NIR (4-channel)."""
    dataset = tf.data.Dataset.from_tensor_slices(files)

    if training:
        dataset = dataset.shuffle(buffer_size=min(len(files), 1000), reshuffle_each_iteration=True)

    if nir_dir is not None:
        dataset = dataset.map(
            lambda f: load_rgb_nir_and_mask(f, str(rgb_dir), str(nir_dir), str(label_dir), img_size=img_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            lambda f: load_rgb_and_mask(f, str(rgb_dir), str(label_dir), img_size=img_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

