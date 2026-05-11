# Binary Change Detection on EO-SAR Image Pairs

**GalaxEye Space — AI Research Intern Technical Assignment**

Pixel-level binary change detection on co-registered Electro-Optical and SAR image pairs using a Siamese U-Net++ with EfficientNet-B0 backbone.

---

## Approach

Early-fusion Siamese U-Net: pre-event and post-event EO+SAR images are concatenated along the channel axis (8 channels total) and fed to a U-Net++ decoder with an ImageNet-pretrained EfficientNet-B0 encoder. Class imbalance is handled via Dice+BCE combined loss with a `pos_weight=5` on the change class.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (works on CPU but slowly)

```
torch==2.1.2
torchvision==0.16.2
segmentation-models-pytorch==0.3.3
timm==0.9.12
albumentations==1.3.1
rasterio==1.3.9
Pillow==10.2.0
numpy==1.26.4
PyYAML==6.0.1
tqdm==4.66.1
matplotlib==3.8.2
```

---

## Environment Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/galaxeye-change-detection.git
cd galaxeye-change-detection

# Create and activate environment
conda create -n galaxeye python=3.10 -y
conda activate galaxeye

# Install PyTorch (choose your CUDA version from pytorch.org)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

After downloading the dataset, place it as follows:

```
data/
  train/
    pre_eo/       ← pre-event Electro-Optical images  (.png / .tif)
    pre_sar/      ← pre-event SAR images              (.png / .tif)
    post_eo/      ← post-event Electro-Optical images
    post_sar/     ← post-event SAR images
    masks/        ← annotation masks (original 0-3 labels)
  val/
    (same structure)
  test/
    (same structure)
```

Label remapping is applied automatically:

| Original | Class | → | Remapped | Class |
|---|---|---|---|---|
| 0 | Background | → | 0 | No-Change |
| 1 | Intact | → | 0 | No-Change |
| 2 | Damaged | → | 1 | Change |
| 3 | Destroyed | → | 1 | Change |

---

## Data Exploration (run first!)

```bash
python explore_data.py --config config.yaml
```

Outputs class distribution plots and sample visualisations to `outputs/eda/`.

---

## Training

```bash
python train.py --config config.yaml
```

Resume from checkpoint:
```bash
python train.py --config config.yaml --resume checkpoints/epoch_020.pth
```

Checkpoints are saved to `checkpoints/`. The best model (by validation F1) is saved as `checkpoints/best.pth`.

---

## Evaluation

```bash
# Test split
python eval.py --config config.yaml --weights checkpoints/best.pth --split test

# Validation split
python eval.py --config config.yaml --weights checkpoints/best.pth --split val

# With custom data path
python eval.py --config config.yaml --weights /path/to/best.pth \
               --data_path /path/to/data --split test
```

Outputs metrics, confusion matrix, and prediction visualisations to `outputs/`.

---

## Model Weights

Download the trained checkpoint:
- **Google Drive / HuggingFace Hub**: [LINK — add after training]

---

## Results

| Split | IoU | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| Val   | — | — | — | — |
| Test  | — | — | — | — |

*Results to be filled after training.*

---

## Citation / References

- ChangeFormer: Bandara & Patel (2022) — *A Transformer-Based Siamese Network for Change Detection*
- BIT-CD: Chen et al. (2021) — *Remote Sensing Image Change Detection with Transformers*
- SNUNet: Fang et al. (2021) — *SNUNet-CD: A Densely Connected Siamese Network for Change Detection*
- segmentation-models-pytorch: https://github.com/qubvel/segmentation_models.pytorch
- albumentations: https://github.com/albumentations-team/albumentations
