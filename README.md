# 🌾 SIH25099: Aerial Multi-Spectral Crop Stress & Nutrient Deficiency Early Warning System

> **A Comprehensive Technical Report and Engineering Reference**  
> *Autonomous Aerial Drone Multi-Spectral Telemetry, Deep Semantic Segmentation (4-Channel U-Net), and Precision Agronomy AI Copilot.*

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Scientific Foundation: Multi-Spectral Physics](#2-scientific-foundation-multi-spectral-physics)
3. [Dataset Architecture & Preprocessing](#3-dataset-architecture--preprocessing)
4. [Deep Learning Model Architecture: 4-Channel U-Net](#4-deep-learning-model-architecture-4-channel-u-net)
5. [Training Strategy & Loss Formulation](#5-training-strategy--loss-formulation)
6. [Empirical Evaluation & Results (RGB vs. RGB+NIR)](#6-empirical-evaluation--results-rgb-vs-rgbnir)
7. [Full-Stack Telemetry System & Drone HUD Architecture](#7-full-stack-telemetry-system--drone-hud-architecture)
8. [AGRI-BOT: AI Precision Agronomist](#8-agri-bot-ai-precision-agronomist)
9. [Module & File Reference](#9-module--file-reference)
10. [Step-by-Step Reproduction Guide](#10-step-by-step-reproduction-guide)

---

## 1. Project Overview & Problem Statement

### The Problem
In precision farming and commercial agriculture, **Nitrogen (N) Deficiency and Chlorosis** are primary drivers of reduced crop yield. When vegetation undergoes nutrient stress, cellular chlorophyll begins breaking down long before physical symptoms appear:
- Traditional scouting relies on visible leaf yellowing or stunted growth.
- **The Delay**: By the time chlorosis is visible to human scouts or RGB cameras (7–10 days into onset), root elongation has stalled and irreversible yield penalties (up to 30–45%) are already locked in.

### The Solution
This project introduces an automated drone-based early warning system that leverages **Near-Infrared (NIR)** reflectance in combination with standard RGB imagery. By processing 4-band multi-spectral aerial imagery through a specialized deep learning segmentation model (Modified 4-Channel U-Net), the system identifies sub-visual cellular breakdown at the pixel level, outputs actionable spatial prescriptions, and integrates an AI agronomist copilot (AGRI-BOT) for real-time dosage calculations.

---

## 2. Scientific Foundation: Multi-Spectral Physics

### Normalized Difference Vegetation Index (NDVI) & Cellular Scattering
Photosynthetic plants interact with sunlight in distinct spectral bands:
1. **Visible Red Band ($\approx 660\text{ nm}$)**: Chlorophyll pigments strongly absorb red light to drive photosynthesis.
2. **Near-Infrared (NIR) Band ($\approx 840\text{ nm}$)**: Light penetrates the leaf epidermis and is strongly scattered/reflected by the hydration-rich spongy mesophyll cell structure.

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

```
       [ Sunlight (Incident Radiation) ]
             │
             ├── Red (~660 nm)  ──────> [ Absorbed by Chlorophyll A & B ]
             │
             └── NIR (~840 nm)  ──────> [ Scattered by Spongy Mesophyll ]
                                             │
                                             ▼ (When Nutrient Stress Occurs)
                                        [ Mesophyll Collapses -> NIR Reflectance Plummets ]
```

- **Healthy Canopy**: High NIR reflectance + Low Red reflectance $\rightarrow \text{NDVI} \in [0.65, 0.90]$.
- **Early-Stage Stressed Canopy**: Mesophyll begins losing turgor pressure; NIR drops while visible color remains deceptively green $\rightarrow \text{NDVI} \in [0.20, 0.45]$.

---

## 3. Dataset Architecture & Preprocessing

The model is trained on the benchmark **Agriculture-Vision Challenge Dataset (CVPR)**:
- **Sensor Modality**: High-resolution nadir aerial imagery captured via UAVs and aerial camera rigs across diverse agricultural zones.
- **Spectral Bands**: 4 synchronized channels — Red, Green, Blue, and Near-Infrared (NIR).
- **Spatial Resolution**: 512×512 tiles with ground sample distance (GSD) of 10–15 cm/pixel.
- **Target Anomaly**: `nutrient_deficiency` binary masks, filtered against background field masks and boundary artifacts.

### Data Processing Pipeline (`dataset.py`)
1. **Bilateral Normalization**: Float32 casting with min-max scaling to $[0.0, 1.0]$.
2. **Channel Fusion**: Interleaving 3-channel RGB with single-channel NIR into 4D tensor batches $(B, 512, 512, 4)$.
3. **Data Augmentation**:
   - Random horizontal and vertical flips (`p = 0.5`).
   - Random 90° rotations to simulate varying drone flight angles.
   - Brightness and contrast perturbations on visible bands.
4. **Class Balancing**: Nutrient deficiency anomalies constitute only $10\% - 25\%$ of typical aerial frames. Balanced sampling filters out non-field empty tiles to prevent gradient saturation towards the background class.

---

## 4. Deep Learning Model Architecture: 4-Channel U-Net

```
Input: (Batch, 512, 512, 4) [R, G, B, NIR]
  │
  ├── [ConvBlock 1] 2x (3x3 Conv, 32 filters, BN, ReLU) ───[Skip 1]───► MaxPool (256x256)
  │     │
  │     └── [ConvBlock 2] 2x (3x3 Conv, 64 filters, BN, ReLU) ───[Skip 2]───► MaxPool (128x128)
  │           │
  │           └── [ConvBlock 3] 2x (3x3 Conv, 128 filters, BN, ReLU) ───[Skip 3]───► MaxPool (64x64)
  │                 │
  │                 └── [ConvBlock 4] 2x (3x3 Conv, 256 filters, BN, ReLU) ───[Skip 4]───► MaxPool (32x32)
  │                       │
  │                 [Bottleneck] 2x (3x3 Conv, 512 filters, BN, ReLU) + Spatial Dropout (0.3)
  │                       │
  │                       ▼
  │                 [UpConv 4] (Conv2DTranspose, 256) ◄── Concat [Skip 4] ── (64x64)
  │                 [UpConv 3] (Conv2DTranspose, 128) ◄── Concat [Skip 3] ── (128x128)
  │                 [UpConv 2] (Conv2DTranspose, 64)  ◄── Concat [Skip 2] ── (256x256)
  │                 [UpConv 1] (Conv2DTranspose, 32)  ◄── Concat [Skip 1] ── (512x512)
  │
  └── 1x1 Conv (1 filter, Sigmoid Activation) ──► Pixel Stress Probability Map (512, 512, 1)
```

### Architectural Rationale
- **Direct Multi-Spectral Input**: Instead of computing simple mathematical NDVI beforehand, feeding raw RGB+NIR channels directly into the convolutional encoder allows the network to automatically learn non-linear band interactions and edge gradients.
- **Skip Connections**: Vital for preserving fine row boundaries and localized canopy stress margins that would otherwise be lost during 16x spatial downsampling.
- **Spatial Dropout (0.3)**: Placed in the bottleneck to prevent co-adaptation of features on recurring crop row geometries.

---

## 5. Training Strategy & Loss Formulation

### Hybrid Compound Loss
Standard Binary Cross-Entropy (BCE) fails on severely imbalanced aerial segmentation tasks, as the background class dominates the gradient. We implemented a hybrid loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

Where $y_i$ is the ground-truth binary label, $\hat{y}_i$ is the predicted probability, and $\epsilon = 10^{-6}$ provides numerical stability.

### Training Hyperparameters
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$)
- **Initial Learning Rate**: $1 \times 10^{-4}$
- **Batch Size**: 8
- **Learning Rate Schedule**: `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`
- **Regularization**: `EarlyStopping(patience=7, restore_best_weights=True)`
- **Mixed Precision**: Enabled (`mixed_float16`) on NVIDIA T4/A100 GPUs for ~3x throughput acceleration.

---

## 6. Empirical Evaluation & Results (RGB vs. RGB+NIR)

The baseline model (standard 3-channel RGB U-Net) and the proposed multi-spectral model (4-channel RGB+NIR U-Net) were benchmarked on identical, unseen validation subsets:

| Metric | Baseline RGB (3-Channel) | Multi-Spectral RGB+NIR (4-Channel) | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Dice Coefficient** | $0.628$ | **$0.764$** | $+0.136$ | **$+21.6\%$** |
| **Mean IoU (Jaccard)** | $0.458$ | **$0.618$** | $+0.160$ | **$+34.9\%$** |
| **Precision** | $0.691$ | **$0.785$** | $+0.094$ | **$+13.6\%$** |
| **Recall (Sensitivity)** | $0.575$ | **$0.744$** | $+0.169$ | **$+29.4\%$** |
| **False Negative Rate** | $42.5\%$ | **$25.6\%$** | $-16.9\%$ | **$-39.8\%$ (Less Misses)** |

### Confusion Matrix Breakdown
- **False Positives (RGB)**: RGB-only models frequently classified shadow cast by tall canopies, bare topsoil ridges, and tire tracks as nutrient stress.
- **False Negatives (RGB)**: Early-stage chlorosis without pronounced yellowing was consistently overlooked.
- **NIR Advantage**: The NIR band cleanly separates soil reflectance (which has a monotonic spectral slope) from green vegetation (which exhibits the characteristic steep "red edge" jump at 700–750 nm), virtually eliminating false shadow alarms.

---

## 7. Full-Stack Telemetry System & Drone HUD Architecture

The end-to-end user application is served via a unified architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Browser                        │
│            http://localhost:5173 (Drone HUD UI)             │
└──────────────┬───────────────────────────────▲──────────────┘
               │                               │
               ▼                               │
┌─────────────────────────────────────────────────────────────┐
│                    Python Full-Stack Core                   │
│                     (full_stack_app.py)                     │
│                                                             │
│   ├── Port 5173: Frontend Web Server & Proxy Handler        │
│   ├── Port 8000: Multi-Spectral Vision API Engine           │
│   └── Endpoints:                                            │
│        - GET  /              -> Serves preview.html         │
│        - GET  /health        -> Health telemetry            │
│        - POST /predict       -> Stress segmentation & masks │
│        - POST /api/chat      -> AGRI-BOT Agronomy Copilot   │
└──────────────┬───────────────────────────────▲──────────────┘
               │                               │
               ▼                               │
┌─────────────────────────────────────────────────────────────┐
│                      Cloud AI Service                       │
│              Groq LLaMA / Qwen-27B Inference                │
│    (Precision fertilizer dosage, spray rates, schedules)    │
└─────────────────────────────────────────────────────────────┘
```

### Drone Viewport Highlights (`preview.html`)
- **HUD Telemetry Overlay**: Real-time crosshairs, altitude readings, GPS coordinates, and scan progress status.
- **Laser Scanning Effect**: Animated multi-spectral beam sweeping across user-uploaded imagery.
- **Bounding Target Boxes**: Automatic spatial tracking identifying primary and secondary stress polygons.
- **Diagnostic Report Card**: Instant metrics summary display with one-click PDF export functionality.

---

## 8. AGRI-BOT: AI Precision Agronomist

Integrated directly into the HUD via a floating mascot avatar and chat panel:
- **Inference Engine**: Powered by Groq Cloud High-Speed LLM inference (`qwen/qwen3.8-27b`).
- **Domain Guardrails**: Strictly restricted to agronomic nutrition, soil fertility, drone spray volumes, and field scan explanations.
- **Actionable Chemical & Organic Prescriptions**:
  - *Foliar Urea Spray*: 1.5% to 2% solution (150–200g per 10L water).
  - *Nano Urea*: 4 ml/liter water for accelerated foliar stomatal absorption.
  - *Drone Flight Specifications*: 20–30 L/ha spray volume, deployed during early morning (6–9 AM) or late afternoon (after 5 PM) to minimize evaporative loss.

---

## 9. Module & File Reference

```text
├── config.py                     # Central hyperparameters, path definitions, and GPU environment checks
├── dataset.py                    # tf.data multi-spectral generator, augmentations, and NIR loader
├── model.py                      # Keras implementation of 3-channel and 4-channel U-Net architectures
├── train.py                      # Resumable training script with ModelCheckpoint & ReduceLROnPlateau
├── evaluate.py                   # Comprehensive evaluation suite (Dice, IoU, confusion matrix, RGB vs NIR)
├── predict.py                    # CLI prediction tool with multi-spectral false-color overlay export
├── utils.py                      # Automated dataset verification, S3 downloads, and unzipping helpers
├── app.py                        # Standalone Streamlit field analysis dashboard
│
├── full_stack_app.py             # Unified HTTP server hosting Drone HUD (5173), API (8000), & Groq AI
├── preview.html                  # Sci-Fi Drone Telemetry HUD single-page application
├── agribot_avatar.png            # 256x256 transparent mascot avatar icon
├── agribot_transparent.png       # High-resolution full-body mascot graphic asset
│
├── computer_vision.ipynb         # Interactive exploratory data analysis & training notebook
├── colab_crop_stress_runner.ipynb# 1-Click Google Colab T4 GPU automated pipeline runner
├── model_diagnostics_suite.ipynb # 5-step model diagnostic, metrics, and visualization notebook
│
├── test_samples/                 # Pre-packaged multi-spectral test imagery
│   ├── sample1_rgb.jpg           # Healthy field RGB reference
│   ├── sample1_nir.jpg           # Healthy field NIR band
│   ├── sample2_stressed_rgb.jpg  # Nitrogen-deficient field RGB
│   └── sample2_stressed_nir.jpg  # Nitrogen-deficient field NIR
│
├── requirements.txt              # Project Python dependencies
└── run_system.sh                 # Environment setup and launch automation script
```

---

## 10. Step-by-Step Reproduction Guide

### 1. Local Environment Setup
```bash
git clone https://github.com/beherabibekananda/computer-vision-project.git
cd computer-vision-project
pip install -r requirements.txt
```

### 2. Launch the Full-Stack Drone HUD & AGRI-BOT
```bash
python full_stack_app.py
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser:
- Drag and drop your drone imagery or click **"Load Stressed Field Sample"**.
- Click **"Run Multi-Spectral AI Scan"** to watch the laser scan and view generated stress polygons.
- Click the **AGRI-BOT Mascot** on the bottom right to query tailored fertilizer prescriptions.

### 3. CLI Prediction on Unseen Imagery
```bash
python predict.py \
  --rgb test_samples/sample2_stressed_rgb.jpg \
  --nir test_samples/sample2_stressed_nir.jpg \
  --threshold 0.5 \
  --output stress_overlay.png
```

### 4. Training from Scratch (Multi-Spectral RGB + NIR)
```bash
python train.py --nir --epochs 25 --batch-size 8 --lr 0.0001
```

### 5. Running the Model Comparison Benchmark
```bash
python evaluate.py --compare
```

---

## 📜 Acknowledgments & Citation
- **Challenge & Dataset**: [Agriculture-Vision: A Large Aerial Anomaly Dataset (CVPR)](https://www.agriculture-vision.com/) by Intelinair & UIUC.
- **Inference Engine**: Groq High-Speed Cloud Inference API.
- **Hackathon**: Developed for Smart India Hackathon (SIH25099).
