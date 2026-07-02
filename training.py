"""
training.py
───────────
Training functions for CNN and ViT from scratch (Models A & B).
  - train_epoch  : one training epoch
  - eval_epoch   : one evaluation epoch
  - run_training : full loop for CNN and ViT scratch
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config


def _get_criterion(class_weights):
    """Creates CrossEntropyLoss with class weights."""
    return nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))


def train_epoch(model, loader, optimizer, criterion):
    """One training epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc='  Train', leave=False):
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        with torch.no_grad():
            correct += (logits.argmax(1) == labels).sum().item()
        total += len(imgs)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    """One evaluation epoch. Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(imgs)

    return total_loss / total, correct / total


def run_training(model, name, train_loader, val_loader, class_weights):
    """
    Full training loop for CNN (Model A) and ViT scratch (Model B).

    Args:
        model         : PyTorch model
        name          : name for checkpoint file (e.g. 'cnn', 'vit')
        train_loader  : DataLoader for training data
        val_loader    : DataLoader for validation data
        class_weights : tensor with class weights

    Returns:
        history (dict), total_time (float in seconds)
    """
    criterion = _get_criterion(class_weights)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS)

    history   = {'train_loss': [], 'val_loss': [],
                 'train_acc':  [], 'val_acc':  []}
    best_acc  = 0.0
    best_path = f'best_{name}.pth'

    print(f'\n── Training: {name} ──────────────────────────────────────────')
    print(f'{"Ep":>4} | {"T-Loss":>8} | {"T-Acc":>7} | '
          f'{"V-Loss":>8} | {"V-Acc":>7} | Time')
    print('-' * 58)

    t_start = time.time()
    for ep in range(1, config.NUM_EPOCHS + 1):
        t0       = time.time()
        tl, ta   = train_epoch(model, train_loader, optimizer, criterion)
        vl, va   = eval_epoch(model,  val_loader,   criterion)
        scheduler.step()

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)

        marker = ' ✓' if va > best_acc else ''
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_path)

        print(f'{ep:>4} | {tl:>8.4f} | {ta:>6.2%} | '
              f'{vl:>8.4f} | {va:>6.2%} | {time.time()-t0:.0f}s{marker}')

    total_time = time.time() - t_start
    print(f'\nBest val-acc: {best_acc:.2%}  |  '
          f'Total time: {total_time/60:.1f} min')
    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    return history, total_time
