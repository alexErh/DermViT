"""
dataset.py
──────────
Dataset class, online augmentation and DataLoaders for HAM10000.
"""

import random

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config


class RandomAugmentation8:
    """
    Online augmentation with 8 variants.
    Each call randomly selects one of the 8 variants:

      0 – Original
      1 – rotated 90°
      2 – rotated 180°
      3 – rotated 270°
      4 – horizontally flipped
      5 – vertically flipped
      6 – horizontally flipped + rotated 90°
      7 – horizontally flipped + rotated 270°

    No extra memory needed – all transforms are in-memory.
    """
    def __call__(self, img):
        variant = random.randint(0, 7)
        if variant == 0: return img
        if variant == 1: return TF.rotate(img, 90)
        if variant == 2: return TF.rotate(img, 180)
        if variant == 3: return TF.rotate(img, 270)
        if variant == 4: return TF.hflip(img)
        if variant == 5: return TF.vflip(img)
        if variant == 6: return TF.rotate(TF.hflip(img), 90)
        if variant == 7: return TF.rotate(TF.hflip(img), 270)


class HAM10000Dataset(Dataset):
    """PyTorch Dataset for the HAM10000 dataset."""

    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(row['label'])


def get_transforms():
    """
    Returns train_tf and val_tf.
    train_tf: RandomAugmentation8 + ColorJitter + Normalize
    val_tf:   Resize + Normalize only (no randomness)
    """
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        RandomAugmentation8(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return train_tf, val_tf


def get_dataloaders(df_train, df_val, df_test):
    """
    Creates DataLoaders for CNN and ViT (Models A & B).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_tf, val_tf = get_transforms()

    train_loader = DataLoader(
        HAM10000Dataset(df_train, train_tf),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        HAM10000Dataset(df_val, val_tf),
        batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        HAM10000Dataset(df_test, val_tf),
        batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    print(f'DataLoaders created  |  Train: {len(train_loader)} batches'
          f'  |  Val: {len(val_loader)} batches')
    return train_loader, val_loader, test_loader
