import argparse
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

import config
import model as unet_module


class CropStressPredictor:
    """Inference engine for crop nutrient deficiency segmentation."""

    def __init__(self, model_path: Optional[Path] = None, use_nir: bool = True):
        self.use_nir = use_nir
        if model_path is None:
            self.model_path = config.RGB_NIR_MODEL_PATH if use_nir else config.RGB_MODEL_PATH
        else:
            self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        custom_objects = {
            "combined_loss": unet_module.combined_loss,
            "dice_loss": unet_module.dice_loss
        }
        self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
        print(f"Loaded predictor model from: {self.model_path}")

    def preprocess(self, rgb_path: Path, nir_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and prepares images as model input tensor and raw RGB display array."""
        rgb_raw = Image.open(rgb_path).convert("RGB")
        rgb_resized = rgb_raw.resize((config.IMG_SIZE, config.IMG_SIZE))
        rgb_arr = np.array(rgb_resized, dtype=np.float32) / 255.0

        if self.use_nir:
            if nir_path is None or not Path(nir_path).exists():
                raise ValueError("NIR path is required for 4-channel model prediction.")
            nir_raw = Image.open(nir_path).convert("L")
            nir_resized = nir_raw.resize((config.IMG_SIZE, config.IMG_SIZE))
            nir_arr = np.array(nir_resized, dtype=np.float32)[:, :, np.newaxis] / 255.0

            combined = np.concatenate([rgb_arr, nir_arr], axis=-1)
            return np.expand_dims(combined, axis=0), np.array(rgb_raw)
        else:
            return np.expand_dims(rgb_arr, axis=0), np.array(rgb_raw)

    def predict(
        self,
        rgb_path: Path,
        nir_path: Optional[Path] = None,
        threshold: float = 0.4
    ) -> dict:
        """Runs prediction and returns probability map, binary mask, and stress statistics."""
        inp_tensor, original_rgb = self.preprocess(rgb_path, nir_path)
        prob_map = self.model.predict(inp_tensor, verbose=0)[0, :, :, 0]
        binary_mask = (prob_map >= threshold).astype(np.uint8)

        # Calculate stress percentage
        total_pixels = binary_mask.size
        stressed_pixels = int(np.sum(binary_mask))
        stress_percentage = (stressed_pixels / total_pixels) * 100.0

        return {
            "probability_map": prob_map,
            "binary_mask": binary_mask,
            "stressed_pixels": stressed_pixels,
            "stress_percentage": round(stress_percentage, 2),
            "threshold": threshold,
            "input_rgb_256": (inp_tensor[0, :, :, :3] * 255).astype(np.uint8),
            "original_rgb": original_rgb
        }


def save_prediction_visual(result: dict, output_path: Path):
    """Saves visual summary of the prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rgb = result["input_rgb_256"]
    mask = result["binary_mask"]
    prob = result["probability_map"]

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB Image")
    axes[0].axis("off")

    im = axes[1].imshow(prob, cmap="magma")
    axes[1].set_title("Stress Probability Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    overlay = rgb.copy()
    overlay[mask == 1] = [255, 0, 0]  # Mark affected zone in red
    axes[2].imshow(overlay)
    axes[2].set_title(f"Affected Zone ({result['stress_percentage']}% Stressed)")
    axes[2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Visual prediction saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Stress Predictor CLI")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--nir", type=str, default=None, help="Path to NIR image (if model uses NIR)")
    parser.add_argument("--model", type=str, default=None, help="Path to model (.keras)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Probability threshold")
    parser.add_argument("--output", type=str, default="output_prediction.png", help="Path to save result plot")

    args = parser.parse_args()
    use_nir = args.nir is not None
    predictor = CropStressPredictor(model_path=args.model, use_nir=use_nir)

    res = predictor.predict(Path(args.rgb), Path(args.nir) if args.nir else None, threshold=args.threshold)
    print(f"Stress Analysis: {res['stress_percentage']}% of field area affected by nutrient deficiency.")
    save_prediction_visual(res, Path(args.output))

