"""
dataset.py
──────────
Dataset class, online augmentation and DataLoaders for HAM10000.

The training set is expanded by the geometric augmentation: every df row is
combined with all 8 variant indices, so a single epoch contains all 8
geometric variants of every image (8× the original number of samples).
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config


class VariantAugmentation8:
    """
    Geometric augmentation with 8 variants. No extra memory needed –
    all transforms are in-memory. The variant is selected by its index:

      0 – Original
      1 – rotated 90°
      2 – rotated 180°
      3 – rotated 270°
      4 – horizontally flipped
      5 – vertically flipped
      6 – horizontally flipped + rotated 90°
      7 – horizontally flipped + rotated 270°

    The variant index comes from the dataset expansion (see HAM10000Dataset),
    so every image is shown in all 8 variants within a single epoch.
    """
    N_VARIANTS = 8

    def __call__(self, img, variant: int):
        if variant == 0: return img
        if variant == 1: return TF.rotate(img, 90)
        if variant == 2: return TF.rotate(img, 180)
        if variant == 3: return TF.rotate(img, 270)
        if variant == 4: return TF.hflip(img)
        if variant == 5: return TF.vflip(img)
        if variant == 6: return TF.rotate(TF.hflip(img), 90)
        if variant == 7: return TF.rotate(TF.hflip(img), 270)


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 dataset.

    When `augment` is given, the dataset is expanded by a factor of
    `augment.N_VARIANTS`: each df row is combined with every variant index,
    so a single epoch contains all 8 geometric variants of every image.
    The flat index maps to (row, variant) as:
        row     = index // n_variants
        variant = index %  n_variants
    """

    def __init__(self, dataframe, transform=None, augment=None):
        self.df         = dataframe.reset_index(drop=True)
        self.transform  = transform
        self.augment    = augment
        self.n_variants = augment.N_VARIANTS if augment is not None else 1

    def __len__(self):
        return len(self.df) * self.n_variants

    def __getitem__(self, index):
        row_idx = index // self.n_variants
        variant = index %  self.n_variants
        row     = self.df.iloc[row_idx]
        image   = Image.open(row['path']).convert('RGB')
        if self.augment:
            image = self.augment(image, variant)
        if self.transform:
            image = self.transform(image)
        return image, int(row['label'])


def get_transforms():
    """
    Returns train_tf and val_tf for CNN and ViT from scratch.
    Geometric augmentation is handled separately in HAM10000Dataset
    to make it deterministic across models.
    """
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return train_tf, val_tf


def get_timm_transforms():
    """
    Transforms for Model C (timm ViT) – 224×224, ImageNet normalization.
    Geometric augmentation is handled separately in HAM10000Dataset
    to make it deterministic across models.
    """
    timm_train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE_TIMM, config.IMG_SIZE_TIMM)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    timm_val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE_TIMM, config.IMG_SIZE_TIMM)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return timm_train_tf, timm_val_tf


def get_dataloaders(df_train, df_val, df_test):
    """
    Creates DataLoaders for CNN and ViT (Models A & B).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_tf, val_tf = get_transforms()

    train_loader = DataLoader(
        HAM10000Dataset(df_train, train_tf, augment=VariantAugmentation8()),
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
    n_train = len(train_loader.dataset)          # expanded: images × variants
    print(f'DataLoaders created  |  Train: {len(df_train)} images × '
          f'{n_train // len(df_train)} variants = {n_train} samples '
          f'({len(train_loader)} batches)  |  Val: {len(val_loader.dataset)} samples')
    return train_loader, val_loader, test_loader


def get_timm_dataloaders(df_train, df_val, df_test):
    """
    Creates DataLoaders for Model C (timm ViT, 224×224).

    Returns:
        timm_train_loader, timm_val_loader, timm_test_loader
    """
    timm_train_tf, timm_val_tf = get_timm_transforms()

    timm_train_loader = DataLoader(
        HAM10000Dataset(df_train, timm_train_tf, augment=VariantAugmentation8()),
        batch_size=32, shuffle=True,
        num_workers=0, pin_memory=False
    )
    timm_val_loader = DataLoader(
        HAM10000Dataset(df_val, timm_val_tf),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=False
    )
    timm_test_loader = DataLoader(
        HAM10000Dataset(df_test, timm_val_tf),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=False
    )
    n_train = len(timm_train_loader.dataset)     # expanded: images × variants
    print(f'timm DataLoaders created  |  Train: {len(df_train)} images × '
          f'{n_train // len(df_train)} variants = {n_train} samples '
          f'({len(timm_train_loader)} batches)')
    return timm_train_loader, timm_val_loader, timm_test_loader


def get_resnet_transforms():
    """
    Transforms for Model D (ResNet50) – 224×224, ImageNet normalization.
    Identical to timm ViT transforms for a fair comparison.
    Geometric augmentation is handled separately in HAM10000Dataset
    to make it deterministic across models.
    """
    resnet_train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE_RESNET, config.IMG_SIZE_RESNET)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    resnet_val_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE_RESNET, config.IMG_SIZE_RESNET)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return resnet_train_tf, resnet_val_tf


def get_resnet_dataloaders(df_train, df_val, df_test):
    """
    Creates DataLoaders for Model D (ResNet50, 224×224).

    Returns:
        resnet_train_loader, resnet_val_loader, resnet_test_loader
    """
    resnet_train_tf, resnet_val_tf = get_resnet_transforms()

    resnet_train_loader = DataLoader(
        HAM10000Dataset(df_train, resnet_train_tf, augment=VariantAugmentation8()),
        batch_size=32, shuffle=True,
        num_workers=0, pin_memory=False
    )
    resnet_val_loader = DataLoader(
        HAM10000Dataset(df_val, resnet_val_tf),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=False
    )
    resnet_test_loader = DataLoader(
        HAM10000Dataset(df_test, resnet_val_tf),
        batch_size=32, shuffle=False,
        num_workers=0, pin_memory=False
    )
    n_train = len(resnet_train_loader.dataset)   # expanded: images × variants
    print(f'ResNet50 DataLoaders created  |  Train: {len(df_train)} images × '
          f'{n_train // len(df_train)} variants = {n_train} samples '
          f'({len(resnet_train_loader)} batches)')
    return resnet_train_loader, resnet_val_loader, resnet_test_loader
