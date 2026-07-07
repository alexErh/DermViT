# DermViT – CNN vs. ViT vs. Pretrained ViT vs. Pretrained CNN

Skin lesion classification on the **HAM10000** dataset, comparing four models to
study two questions side by side:

- **Architecture:** classic CNN vs. Vision Transformer (ViT)
- **Transfer learning:** training from scratch vs. fine-tuning pretrained weights

> **Research question:** Can a Vision Transformer outperform a classic CNN – and
> what does transfer learning contribute to CNN and ViT respectively?

Based on the paper *An Image is Worth 16×16 Words: Transformers for Image
Recognition at Scale* (Dosovitskiy et al., ICLR 2021).

Module: **Concepts of Deep Learning**

---

## The four models

| Model | Architecture | Pretrained | Input | Section in notebook |
|-------|--------------|------------|-------|---------------------|
| **A** | CNN (4 conv blocks) | No | 64×64 | 3 |
| **B** | Vision Transformer (from scratch) | No | 64×64 | 4 |
| **C** | ViT-Small (`timm`) | ImageNet-21k | 224×224 | 5 |
| **D** | ResNetV2-50x1 / BiT (`timm`) | ImageNet-21k | 224×224 | 6 |

Models C and D are both pretrained on **ImageNet-21k** and use a single linear
head, so the C↔D comparison isolates the architecture (Transformer vs. CNN).

The four models span both comparison axes: A↔B compares architectures trained
from scratch, C↔D compares architectures with transfer learning, A↔D shows the
effect of transfer learning for CNNs, and B↔C shows it for ViTs.

---

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
("Human Against Machine with 10000 training images") – 10,015 dermatoscopic
images across 7 lesion classes.

| Abbreviation | Name | Type |
|------|------|------|
| `mel`   | Melanoma | malignant |
| `bcc`   | Basal cell carcinoma | malignant |
| `akiec` | Actinic keratosis | potentially malignant |
| `bkl`   | Benign keratosis | benign |
| `nv`    | Melanocytic nevus | benign |
| `df`    | Dermatofibroma | benign |
| `vasc`  | Vascular lesion | benign |

The dataset is **heavily imbalanced**: `nv` alone accounts for ~67% of all
images. This is addressed with class-weighted cross-entropy loss and reported via
**balanced accuracy** in addition to plain accuracy.

### Setup

1. Create a [Kaggle](https://www.kaggle.com) account and download the dataset.
2. Place `ham10000.zip` in the project root and run `Unzipper.ipynb`, **or**
   extract it manually into `./data/ham10000/`.

Expected folder structure:

```
data/ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   (*.jpg)
└── HAM10000_images_part_2/   (*.jpg)
```

---

## Project structure

| File | Purpose |
|------|---------|
| `config.py` | Global configuration: paths, hyperparameters, class mapping. `init()` sets up `DEVICE`, seeds, and loads the dataframe. |
| `dataset.py` | `HAM10000Dataset`, online 8-variant augmentation, and DataLoaders for each model variant. |
| `models.py` | All architectures: `CNN`, `ViT` (from scratch), and factory functions for the pretrained `timm` ViT and ResNet50. |
| `training.py` | Training loops: single-phase for A & B, two-phase (head-only → full fine-tuning) for C & D. |
| `visualization.py` | Evaluation (`get_predictions`) and interpretability: ViT attention rollout + CNN Grad-CAM. |
| `DermViT_4models.ipynb` | Main notebook – runs the full comparison end to end. |
| `Unzipper.ipynb` | Helper to extract the dataset archive. |

Running the notebook also creates a `histories/` folder with the saved training
curves (see *Key implementation details*).

---

## Usage

### Requirements

The first notebook cell installs everything:

```python
torch, torchvision, einops, matplotlib, seaborn,
pandas, scikit-learn, tqdm, Pillow, timm
```

### Run

Open `DermViT_4models.ipynb` and run the cells top to bottom. The notebook will:

1. Load and analyze the dataset (class distribution, example images).
2. Build and train all four models (saving each training history to `histories/`).
3. Evaluate on the test set (accuracy, balanced accuracy, confusion matrices, per-class F1).
4. Generate interpretability visualizations (attention rollout vs. Grad-CAM).
5. Produce a final summary comparing all four models.

> **Note on compute:** The default `IMG_SIZE = 64` keeps Models A & B fast on
> CPU. Models C & D use 224×224 inputs and are considerably heavier — training
> on CPU takes hours. Use a GPU where possible (set `IMG_SIZE = 224` for the
> from-scratch models too if you have the budget).

Trained weights are saved as `best_<name>.pth`, and figures are written as
`.png` files in the project root.

---

## Key implementation details

- **Expanded geometric augmentation** (`VariantAugmentation8`): the training set
  is expanded 8× — every image is combined with all 8 rotation/flip variant
  indices, so **a single epoch contains all 8 geometric variants of every
  image** (no extra disk usage, all transforms in-memory). The dataset's flat
  index maps to `(row, variant)` via `row = index // 8`, `variant = index % 8`.
  Because the variants come from the dataset structure (not from chance), all 4
  models train on the exact same expanded sample set. `ColorJitter` is left
  commented out. **Note:** each epoch is 8× larger, so training is ~8× slower per
  epoch — `NUM_EPOCHS` is set accordingly.
- **Two-phase fine-tuning** (Models C & D): first train only the classification
  head with the backbone frozen, then fine-tune all weights with a small learning
  rate (cosine schedule). The phase lengths are configurable in `config.py` via
  `PRETRAINED_HEAD_EPOCHS` (default 5) and `PRETRAINED_FINETUNE_EPOCHS`
  (default 10).
- **Training-history logging:** after each model is trained, the notebook calls
  `save_history(...)`, which writes the full training curves to `histories/` as
  `history_<name>.pkl` (reload with `load_history`), `history_<name>.csv`, and a
  `history_<name>_metadata.json` (config snapshot, best val-acc/loss, training
  time). This lets you re-run the analysis/plots later without retraining.
- **Class imbalance** is handled with inverse-frequency class weights in the
  cross-entropy loss.
- **Interpretability:** ViT uses *attention rollout* (Abnar & Zuidema, 2020)
  accumulated across all transformer blocks; the CNN uses *Grad-CAM* on its last
  convolutional layer.

---

## Reproducibility

A fixed seed (`SEED = 42`) is set for `random`, `numpy`, and `torch` in
`config.init()`. The train/val/test split is stratified (70/15/15).
