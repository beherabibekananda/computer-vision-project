import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config
import dataset
import model as unet_module
import utils


def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.4,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """Computes Dice (F1), IoU (Jaccard), Precision, Recall, Specificity, and Confusion Matrix components."""
    y_true_f = (y_true > 0.5).astype(np.uint8).flatten()
    y_pred_f = (y_pred_prob >= threshold).astype(np.uint8).flatten()

    tp = np.sum((y_true_f == 1) & (y_pred_f == 1))
    fp = np.sum((y_true_f == 0) & (y_pred_f == 1))
    fn = np.sum((y_true_f == 1) & (y_pred_f == 0))
    tn = np.sum((y_true_f == 0) & (y_pred_f == 0))

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold
    }


def plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    title: str = "Confusion Matrix",
    save_path: Path = None
):
    """Plots and optionally saves a pixel-level confusion matrix heatmap."""
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)

    classes = ["Negative (Healthy)", "Positive (Stressed)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, rotation=15)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            ax.text(
                j, i, f"{count:,}",
                ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix plot to {save_path}")
    plt.close()


def visualize_predictions(
    model: tf.keras.Model,
    eval_dataset: tf.data.Dataset,
    num_samples: int = 5,
    threshold: float = 0.4,
    use_nir: bool = True,
    save_path: Path = None
):
    """Plots qualitative comparison: RGB, (optional NIR), Ground Truth, Predicted Mask, and Overlay."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples_shown = 0
    fig_cols = 5 if use_nir else 4
    fig, axes = plt.subplots(num_samples, fig_cols, figsize=(4 * fig_cols, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for batch_images, batch_masks in eval_dataset:
        preds = model.predict(batch_images, verbose=0)
        batch_size = batch_images.shape[0]

        for i in range(batch_size):
            if samples_shown >= num_samples:
                break

            img = batch_images[i].numpy()
            gt = batch_masks[i].numpy().squeeze()
            prob = preds[i].squeeze()
            pred_mask = (prob >= threshold).astype(np.uint8)

            rgb = img[:, :, :3]
            col_idx = 0

            # RGB Image
            axes[samples_shown, col_idx].imshow(rgb)
            axes[samples_shown, col_idx].set_title(f"Sample {samples_shown + 1} - RGB")
            axes[samples_shown, col_idx].axis("off")
            col_idx += 1

            # NIR Image (if present)
            if use_nir:
                nir = img[:, :, 3]
                axes[samples_shown, col_idx].imshow(nir, cmap="gray")
                axes[samples_shown, col_idx].set_title("NIR Channel")
                axes[samples_shown, col_idx].axis("off")
                col_idx += 1

            # Ground Truth
            axes[samples_shown, col_idx].imshow(gt, cmap="gray")
            axes[samples_shown, col_idx].set_title("Ground Truth Mask")
            axes[samples_shown, col_idx].axis("off")
            col_idx += 1

            # Prediction
            axes[samples_shown, col_idx].imshow(pred_mask, cmap="Reds", vmin=0, vmax=1)
            axes[samples_shown, col_idx].set_title(f"Prediction (th={threshold})")
            axes[samples_shown, col_idx].axis("off")
            col_idx += 1

            # Overlay
            overlay = rgb.copy()
            overlay[pred_mask == 1] = [1.0, 0.0, 0.0]  # Highlight stress in red
            axes[samples_shown, col_idx].imshow(overlay)
            axes[samples_shown, col_idx].set_title("Stress Overlay on RGB")
            axes[samples_shown, col_idx].axis("off")

            samples_shown += 1

        if samples_shown >= num_samples:
            break

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visual predictions to {save_path}")
    plt.close()


def evaluate_model(
    model_path: Path,
    use_nir: bool = True,
    sample_size: int = 100,
    thresholds: List[float] = [0.2, 0.3, 0.4, 0.5]
) -> Dict[str, Dict]:
    """Evaluates a saved model across multiple probability thresholds."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    custom_objects = {
        "combined_loss": unet_module.combined_loss,
        "dice_loss": unet_module.dice_loss
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Loaded model from {model_path}")

    # Build evaluation test set with balanced positive and negative val images
    val_files = dataset.get_image_files(config.VAL_RGB_DIR)
    pos_val, neg_val = dataset.find_positive_negative_files(val_files, config.VAL_LABEL_DIR)

    random.seed(config.RANDOM_SEED)
    k = min(len(pos_val), len(neg_val), sample_size // 2)
    eval_files = random.sample(pos_val, k) + random.sample(neg_val, k)
    random.shuffle(eval_files)

    eval_ds = dataset.make_dataset(
        eval_files,
        rgb_dir=config.VAL_RGB_DIR,
        label_dir=config.VAL_LABEL_DIR,
        nir_dir=config.VAL_NIR_DIR if use_nir else None,
        batch_size=config.BATCH_SIZE,
        training=False
    )

    all_y_true = []
    all_y_pred = []

    for images, masks in eval_ds:
        preds = model.predict(images, verbose=0)
        all_y_true.append(masks.numpy())
        all_y_pred.append(preds)

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)

    results = {}
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS ({'RGB+NIR' if use_nir else 'RGB'} on {len(eval_files)} validation samples)")
    print("=" * 65)
    print(f"{'Threshold':<10} | {'Dice (F1)':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)

    for th in thresholds:
        m = compute_metrics(y_true_all, y_pred_all, threshold=th)
        results[f"th_{th}"] = m
        print(f"{th:<10.2f} | {m['dice']:<10.4f} | {m['iou']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f}")

    # Best threshold by Dice
    best_th = max(thresholds, key=lambda t: results[f"th_{t}"]["dice"])
    best_m = results[f"th_{best_th}"]

    # Save confusion matrix for best threshold
    tag = "rgb_nir" if use_nir else "rgb"
    cm_path = config.OUTPUT_DIR / f"{tag}_confusion_matrix.png"
    plot_confusion_matrix(
        best_m["tp"], best_m["fp"], best_m["fn"], best_m["tn"],
        title=f"Confusion Matrix ({tag.upper()} - Th={best_th})",
        save_path=cm_path
    )

    # Save visual prediction samples
    vis_path = config.OUTPUT_DIR / f"{tag}_predictions.png"
    visualize_predictions(
        model, eval_ds, num_samples=5, threshold=best_th, use_nir=use_nir, save_path=vis_path
    )

    return results


def compare_rgb_vs_rgb_nir():
    """Runs side-by-side comparison between RGB model and RGB+NIR model."""
    print("Running RGB vs RGB+NIR benchmark...")
    rgb_results = None
    rgb_nir_results = None

    if config.RGB_MODEL_PATH.exists():
        rgb_results = evaluate_model(config.RGB_MODEL_PATH, use_nir=False)
    else:
        print(f"RGB model not found at {config.RGB_MODEL_PATH}. Skipping.")

    if config.RGB_NIR_MODEL_PATH.exists():
        rgb_nir_results = evaluate_model(config.RGB_NIR_MODEL_PATH, use_nir=True)
    else:
        print(f"RGB+NIR model not found at {config.RGB_NIR_MODEL_PATH}. Skipping.")

    if rgb_results and rgb_nir_results:
        print("\n" + "=" * 50)
        print("MODEL COMPARISON (Threshold 0.40)")
        print("=" * 50)
        m_rgb = rgb_results["th_0.4"]
        m_nir = rgb_nir_results["th_0.4"]

        print(f"{'Metric':<15} | {'RGB Only':<12} | {'RGB + NIR':<12} | {'Delta':<10}")
        print("-" * 55)
        for metric in ["dice", "iou", "precision", "recall"]:
            v1 = m_rgb[metric]
            v2 = m_nir[metric]
            diff = v2 - v1
            print(f"{metric.capitalize():<15} | {v1:<12.4f} | {v2:<12.4f} | {diff:+10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Crop Stress Segmentation Models")
    parser.add_argument("--model-path", type=str, default=str(config.RGB_NIR_MODEL_PATH), help="Path to .keras model")
    parser.add_argument("--rgb-only", action="store_true", help="Evaluate RGB-only model (3 channels)")
    parser.add_argument("--compare", action="store_true", help="Compare RGB and RGB+NIR models")
    args = parser.parse_args()

    if args.compare:
        compare_rgb_vs_rgb_nir()
    else:
        evaluate_model(Path(args.model_path), use_nir=(not args.rgb_only))

