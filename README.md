# Industrial & Inventory Vision Ensemble 🏭🔍

An end-to-end computer vision system powered by a **5-model YOLO11 ensemble** for multi-domain inventory management, automated counting, quality inspection, and parts classification.

---

## 📌 Overview

Standard single-model object detectors struggle with heterogeneous inventories containing microscopic fasteners, heavy raw materials, pharmaceutical bottles, and hand tools all at once. 

This project partitions the detection problem into **5 domain-specialized YOLO11 models**:

| Model | Domain | Target Classes & Components |
|---|---|---|
| **Model 1: Small Parts & Fasteners** (`small_parts`) | Hardware & Fasteners | Hex bolts (M6/M10/M16), standard/lock nuts, flat & spring washers, Phillips/Allen screws, pop rivets. |
| **Model 2: MRO Hand Tools** (`mro_tools`) | Maintenance & Workshop | Open-end & ring wrenches, flat & Phillips screwdrivers, pliers, claw hammers, calipers, tape measures. |
| **Model 3: Heavy Industrial Materials** (`heavy_materials`) | Bulk & Heavy Stock | PVC & steel pipes, cable reels, sheet metal rolls, 200L drums, pallets, industrial gas cylinders, I-beams. |
| **Model 4: Medical & Pharma Supplies** (`medical_pharma`) | Medical & Healthcare | Syringes (5ml/20ml), IV infusion bags, blister packs, amber medicine bottles, surgical gloves, N95 masks. |
| **Model 5: Irregular & Specialized Goods** (`irregular_specialized`) | Electrical & Irregular Parts | Loose cable bundles, coiled hoses, filter cartridges, rubber gaskets, O-rings, populated PCBs, transformers. |

---

## 🛠️ Repository Structure

```text
├── cam_only_test.py                # Direct webcam testing for trained weights
├── webcam_test.py                  # Live multi-model webcam inference with HUD & hotkey switching
├── simple_webcam.py                # Basic camera check
├── train_ensemble.py               # Automated YOLO11 training pipeline across models
├── rf_workspace_setup.py           # Roboflow workspace setup, harmonization, and sync
├── rf_find_datasets.py             # Public dataset discovery & verification
├── rf_async_download.py            # Asynchronous dataset downloader
├── rf_retry_failed.py              # Download retry utility
├── heavy_materials_fetch.py        # Custom data fetcher for heavy industrial items
├── heavy_materials_openimages.py   # OpenImages dataset fetcher
├── run_pipeline.sh                 # Full bash pipeline script (Download -> Merge -> Upload -> Train)
├── requirements.txt                # Python dependencies
├── datasets/                       # Training, validation, and test datasets
├── raw_downloads/                  # Raw dataset archives
└── runs/                           # YOLO11 experiment runs, metrics, and trained model weights
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/murugavel03/industrial-inventory-vision-ensemble.git
cd industrial-inventory-vision-ensemble
pip install -r requirements.txt
```

Recommended dependencies:
- Python 3.10+
- PyTorch (CUDA-enabled recommended for GPU training & inference)
- Ultralytics YOLO (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- Roboflow SDK (`pip install roboflow`)

---

### 2. Live Camera Inference

Run real-time inference using the trained weights:

```bash
# Run multi-model interactive viewer (supports switching models on the fly with keys 1-5)
python webcam_test.py --cam 0 --conf 0.45

# Or run simple test with default model
python cam_only_test.py
```

**Hotkeys in `webcam_test.py`**:
- `1` - `5`: Switch between the 5 specialized models
- `+` / `-`: Increase or decrease confidence threshold
- `S`: Save screenshot
- `Q`: Quit

---

### 3. Training the Ensemble Models

To train all models or a specific domain:

```bash
# Train all models sequentially
python train_ensemble.py --mode all

# Train a specific specialized model
python train_ensemble.py --mode single --model small_parts

# Resume an interrupted training session
python train_ensemble.py --mode resume --model irregular_specialized
```

---

### 4. Dataset Synchronization & Pipeline

Set your Roboflow API key and run the automated dataset orchestration pipeline:

```bash
# Setup projects on Roboflow workspace
python rf_workspace_setup.py --api_key YOUR_API_KEY --mode setup

# Or run the complete automated pipeline
bash run_pipeline.sh
```

---

## 📊 Model Performance Highlights

- **Irregular & Specialized (`irregular_specialized_v1`)**: mAP50 of **~82.0%**, mAP50-95 of **~63.3%**
- **Small Parts & Fasteners (`small_parts_v1`)**: mAP50 of **~99.1%** on target fastener classes

---

## 📜 License

This project is licensed under the MIT License.
