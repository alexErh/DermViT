"""
config.py
─────────
Global configuration for DermViT.
All hyperparameters and paths in one place.
init() initializes DEVICE, SEED and class mapping.
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path('./data/ham10000')
CSV_PATH  = DATA_DIR / 'HAM10000_metadata.csv'
IMG_DIR_1 = DATA_DIR / 'HAM10000_images_part_1'
IMG_DIR_2 = DATA_DIR / 'HAM10000_images_part_2'

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE     = 64       # 64×64 px – for CPU; increase to 224 for GPU
BATCH_SIZE   = 64
NUM_EPOCHS   = 15       # for CNN and ViT from scratch (Models A & B)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
SEED         = 42

# ── Two-phase training for pretrained models (Models C & D) ───────────────────
PRETRAINED_HEAD_EPOCHS     = 5    # phase 1: train classification head only
PRETRAINED_FINETUNE_EPOCHS = 10   # phase 2: fine-tune all weights

# ── ViT-specific hyperparameters ──────────────────────────────────────────────
PATCH_SIZE = 8
EMBED_DIM  = 256
NUM_HEADS  = 8
NUM_LAYERS = 6
MLP_DIM    = 512
DROPOUT    = 0.1

# ── timm ViT ──────────────────────────────────────────────────────────────────
IMG_SIZE_TIMM = 224

# ── timm ResNet50 ─────────────────────────────────────────────────────────────
IMG_SIZE_RESNET = 224   # ResNet50 also expects 224×224

# ── Normalization values (HAM10000-specific) ──────────────────────────────────
MEAN = [0.7630, 0.5456, 0.5700]
STD  = [0.1409, 0.1521, 0.1691]

# ── Classes ───────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesion'
}

# Populated by init()
DEVICE      = None
NUM_CLASSES = None
CLASSES     = None
CLASS2IDX   = None
IDX2CLASS   = None


def init(csv_path=None):
    """
    Initializes all global variables.
    Must be called once at the beginning of the notebook.

    Returns:
        df (pd.DataFrame): Loaded and prepared dataset
    """
    global DEVICE, NUM_CLASSES, CLASSES, CLASS2IDX, IDX2CLASS

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEVICE.type == 'cpu':
        torch.set_num_threads(min(16, os.cpu_count() or 1))
    print(f'Device: {DEVICE}  |  CPU threads: {torch.get_num_threads()}')

    # Load dataset
    path = Path(csv_path) if csv_path else CSV_PATH
    df = pd.read_csv(path)

    # Class mapping
    CLASSES   = sorted(df['dx'].unique().tolist())
    CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
    IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
    NUM_CLASSES = len(CLASSES)

    # Assign image paths
    def find_image(image_id):
        for d in [IMG_DIR_1, IMG_DIR_2]:
            p = d / f'{image_id}.jpg'
            if p.exists():
                return str(p)
        return None

    df['path']  = df['image_id'].apply(find_image)
    df = df.dropna(subset=['path']).reset_index(drop=True)
    df['label'] = df['dx'].map(CLASS2IDX)

    print(f'Dataset: {len(df)} images, {NUM_CLASSES} classes')
    print(df['dx'].value_counts().to_string())
    return df
