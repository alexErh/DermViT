"""
models.py
─────────
All model architectures for DermViT:
  - CNN          : classic approach (Model A)
  - ViT          : custom implementation following Dosovitskiy et al. (Model B)
  - get_timm_vit : pretrained ViT from timm (Model C)
  - ResNet50Timm : pretrained ResNet50 from timm (Model D)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

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
# Model B: Vision Transformer (from scratch)
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Step 1: Split image into patches and linearly project them.
    Conv2d with kernel=stride=patch_size handles both in one step.

    Input:  (B, C, H, W)
    Output: (B, num_patches, embed_dim)
    """

    def __init__(self, img_size, patch_size, in_ch=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                           # (B, D, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) · V
    Attention weights are stored for visualization.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads    = num_heads
        self.head_dim     = embed_dim // num_heads
        self.scale        = self.head_dim ** -0.5
        self.qkv          = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj_out     = nn.Linear(embed_dim, embed_dim)
        self.drop         = nn.Dropout(dropout)
        self.attn_weights = None  # for attention rollout

    def forward(self, x):
        B, N, C = x.shape
        qkv = rearrange(
            self.qkv(x),
            'b n (three h d) -> three b h n d',
            three=3, h=self.num_heads
        )
        q, k, v = qkv.unbind(0)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()
        attn = self.drop(attn)
        out  = torch.einsum('bhij,bhjd->bhid', attn, v)
        return self.proj_out(rearrange(out, 'b h n d -> b n (h d)'))


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Encoder Block:
      x → LayerNorm → MSA → Residual
        → LayerNorm → FFN → Residual
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer – full implementation following Dosovitskiy et al. (ICLR 2021).

    Pipeline:
      Image → PatchEmbedding → [CLS] + PositionEmbedding
            → L × TransformerBlock
            → LayerNorm → CLS token → MLP Head → Class
    """

    def __init__(self, img_size, patch_size, num_classes,
                 embed_dim, num_heads, num_layers, mlp_dim, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        n = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, embed_dim) * 0.02)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([repeat(self.cls_token, '1 1 d -> b 1 d', b=B), x], dim=1)
        x = self.drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])

    def get_attention_maps(self, x):
        """Returns attention maps from all blocks (for attention rollout)."""
        self.eval()
        with torch.no_grad():
            B = x.shape[0]
            x = self.patch_embed(x)
            x = torch.cat([repeat(self.cls_token, '1 1 d -> b 1 d', b=B), x], dim=1)
            x = self.drop(x + self.pos_embed)
            maps = []
            for block in self.blocks:
                x = block(x)
                maps.append(block.attn.attn_weights.cpu())
        return maps


# ══════════════════════════════════════════════════════════════════════════════
# Model C: Pretrained ViT from timm
# ══════════════════════════════════════════════════════════════════════════════

def get_timm_vit(num_classes=7):
    """
    Loads pretrained ViT-Small from timm (Model C).
    Weights from ImageNet-21k (14M images).

    Returns:
        vit_timm (nn.Module): pretrained ViT-Small model
    """
    import timm
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=True,
        num_classes=num_classes
    )
    return model


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


def build_vit(device=None):
    """Creates and returns ViT (from scratch)."""
    dev = device or config.DEVICE
    model = ViT(
        img_size    = config.IMG_SIZE,
        patch_size  = config.PATCH_SIZE,
        num_classes = config.NUM_CLASSES,
        embed_dim   = config.EMBED_DIM,
        num_heads   = config.NUM_HEADS,
        num_layers  = config.NUM_LAYERS,
        mlp_dim     = config.MLP_DIM,
        dropout     = config.DROPOUT
    ).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f'ViT parameters: {params:,}')
    return model


def build_timm_vit(device=None):
    """Loads pretrained ViT-Small from timm."""
    dev = device or config.DEVICE
    model = get_timm_vit(num_classes=config.NUM_CLASSES).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f'timm ViT-Small parameters: {params:,}')
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model D: Pretrained ResNet50 from timm
# ══════════════════════════════════════════════════════════════════════════════

class ResNet50Timm(nn.Module):
    """
    Wrapper around timm ResNet50 – ensures .fc (head) is accessible
    uniformly like model.head in ViT (for two-phase training).
    """

    def __init__(self, num_classes=7):
        super().__init__()
        import timm
        base = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.backbone  = base
        in_features    = base.num_features          # 2048 for ResNet50
        self.head      = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)   # (B, 2048)
        return self.head(features)


def build_resnet50(device=None):
    """
    Loads pretrained ResNet50 from timm (Model D).
    Pretrained on ImageNet-1k.
    Same input size as Model C: 224×224 px.
    """
    dev   = device or config.DEVICE
    model = ResNet50Timm(num_classes=config.NUM_CLASSES).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f'ResNet50 parameters: {params:,}')
    print(f'  → Backbone: ResNet50 (ImageNet-1k pretrained)')
    print(f'  → Head:     Linear(2048→512→{config.NUM_CLASSES})')
    return model
