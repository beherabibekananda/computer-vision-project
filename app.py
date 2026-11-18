import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

import config
from predict import CropStressPredictor

st.set_page_config(
    page_title="SIH25099 - Crop Stress Early Warning",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 SIH25099: Crop Stress Early Warning System")
st.markdown("### Aerial Multi-Spectral Nutrient Deficiency Detection (RGB + NIR U-Net)")

# Sidebar Settings
st.sidebar.header("Configuration & Model Settings")
mode = st.sidebar.radio("Model Input Mode", ["RGB + NIR (4 channels)", "RGB Only (3 channels)"])
use_nir = (mode == "RGB + NIR (4 channels)")

model_default_path = config.RGB_NIR_MODEL_PATH if use_nir else config.RGB_MODEL_PATH
model_path_str = st.sidebar.text_input("Model Checkpoint Path", str(model_default_path))
threshold = st.sidebar.slider("Stress Decision Threshold", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

@st.cache_resource
def get_model(path: str, nir: bool):
    p = Path(path)
    if not p.exists():
        return None
    return CropStressPredictor(model_path=p, use_nir=nir)

predictor = get_model(model_path_str, use_nir)

if predictor is None:
    st.sidebar.warning(f"Model file not found at: {model_path_str}. Please train the model or provide valid path.")

col1, col2 = st.columns(2)

with col1:
    rgb_file = st.file_uploader("Upload Aerial RGB Image", type=["jpg", "jpeg", "png"])

with col2:
    nir_file = None
    if use_nir:
        nir_file = st.file_uploader("Upload Aerial NIR Image (Grayscale)", type=["jpg", "jpeg", "png"])
    else:
        st.info("NIR channel upload disabled (RGB-only mode).")

if rgb_file and (not use_nir or nir_file):
    temp_dir = Path("/tmp/crop_stress_app")
    temp_dir.mkdir(parents=True, exist_ok=True)

    rgb_temp_path = temp_dir / "temp_rgb.jpg"
    with open(rgb_temp_path, "wb") as f:
        f.write(rgb_file.getbuffer())

    nir_temp_path = None
    if use_nir and nir_file:
        nir_temp_path = temp_dir / "temp_nir.jpg"
        with open(nir_temp_path, "wb") as f:
            f.write(nir_file.getbuffer())

    st.divider()
    if st.button("🚀 Analyze Crop Stress", type="primary"):
        if predictor is None:
            st.error("Please load a valid model checkpoint first.")
        else:
            with st.spinner("Analyzing spectral signatures and running U-Net inference..."):
                result = predictor.predict(
                    rgb_path=rgb_temp_path,
                    nir_path=nir_temp_path,
                    threshold=threshold
                )

            st.success("Analysis Complete!")
            stress_pct = result["stress_percentage"]

            # Metric Cards
            m1, m2, m3 = st.columns(3)
            m1.metric("Field Stress Index", f"{stress_pct}%")
            m2.metric("Stressed Pixels", f"{result['stressed_pixels']:,}")
            m3.metric("Severity Category", "High" if stress_pct > 20 else ("Moderate" if stress_pct > 5 else "Low"))

            # Display Plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(result["input_rgb_256"])
            axes[0].set_title("Input RGB")
            axes[0].axis("off")

            im1 = axes[1].imshow(result["probability_map"], cmap="magma")
            axes[1].set_title(f"Nutrient Stress Heatmap (Th={threshold})")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            overlay = result["input_rgb_256"].copy()
            overlay[result["binary_mask"] == 1] = [255, 0, 0]
            axes[2].imshow(overlay)
            axes[2].set_title(f"Deficiency Delineation ({stress_pct}%)")
            axes[2].axis("off")

            st.pyplot(fig)

