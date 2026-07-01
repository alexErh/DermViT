"""
models.py
─────────
Model architectures for DermViT:
  - CNN : classic approach (Model A)
"""

import torch
import torch.nn as nn

import config


# ══════════════════════════════════════════════════════════════════════════════
# Model A: CNN
# ══════════════════════════════════════════════════════════════════════════════

class CNN(nn.Module):
    """
    Classic Convolutional Neural Network (Model A).
    4 conv blocks: Conv → BN → ReLU → MaxPool
    Filter count doubles per block: 32 → 64 → 128 → 256
    """

    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout * 0.5)
            )

        self.features = nn.Sequential(
            conv_block(3,   32),   # 64 → 32
            conv_block(32,  64),   # 32 → 16
            conv_block(64,  128),  # 16 → 8
            conv_block(128, 256),  #  8 → 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ══════════════════════════════════════════════════════════════════════════════
# Factory functions
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn(device=None):
    """Creates and returns CNN."""
    dev = device or config.DEVICE
    model = CNN(num_classes=config.NUM_CLASSES).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f'CNN parameters: {params:,}')
    return model
