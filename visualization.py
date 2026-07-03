"""
visualization.py
────────────────
Interpretability visualizations and evaluation:
  - get_predictions     : predictions on the test set
  - attention_rollout   : accumulate attention maps across all ViT blocks
  - GradCAM             : Grad-CAM for CNN
  - visualize_comparison: Original | ViT Attention | CNN Grad-CAM side by side
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config


@torch.no_grad()
def get_predictions(model, loader):
    """
    Runs inference on a DataLoader.

    Returns:
        preds  (np.array): predicted classes
        labels (np.array): true classes
        probs  (np.array): softmax probabilities
    """
    model.eval()
    preds, labels, probs_all = [], [], []

    for imgs, lbls in loader:
        logits = model(imgs.to(config.DEVICE)).cpu()
        probs  = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(1).numpy())
        labels.extend(lbls.numpy())
        probs_all.extend(probs.numpy())

    return np.array(preds), np.array(labels), np.array(probs_all)


def attention_rollout(attn_maps):
    """
    Accumulates attention weights across all transformer blocks.
    Method: Abnar & Zuidema (2020).

    Args:
        attn_maps: list of attention tensors per block

    Returns:
        mask (Tensor): normalized CLS token attention over all patches
    """
    result = torch.eye(attn_maps[0].shape[-1])
    for attn in attn_maps:
        a = attn[0].mean(0)
        a = a + torch.eye(a.shape[0])
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    mask = result[0, 1:]  # CLS → all patches
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)


class GradCAM:
    """
    Grad-CAM for the last conv layer of the CNN.
    Visualizes which image regions activate the CNN.
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target_layer.register_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def visualize_comparison(vit_model, cnn_model, gradcam,
                          image_pil, true_lbl, val_tf,
                          class_names, idx2class,
                          patch_size, img_size):
    """
    Shows four panels side by side:
      Original | ViT Attention Rollout | CNN Grad-CAM | Activation comparison

    Args:
        vit_model  : ViT model (scratch or timm)
        cnn_model  : CNN model
        gradcam    : GradCAM instance
        image_pil  : PIL image
        true_lbl   : true label (int)
        val_tf     : validation transform
        class_names: dict {abbreviation: name}
        idx2class  : dict {int: abbreviation}
        patch_size : patch size of the ViT
        img_size   : image size

    Returns:
        fig (matplotlib.figure.Figure)
    """
    import matplotlib.pyplot as plt

    img_t = val_tf(image_pil).unsqueeze(0).to(config.DEVICE)

    # ViT attention rollout
    attn_maps = vit_model.get_attention_maps(img_t)
    mask_vit  = attention_rollout(attn_maps)
    n_side    = img_size // patch_size
    mask_vit  = mask_vit.reshape(n_side, n_side).numpy()
    mask_vit  = np.array(
        Image.fromarray((mask_vit * 255).astype(np.uint8))
             .resize((img_size, img_size), Image.BILINEAR)
    ) / 255.0

    # CNN Grad-CAM
    mask_cnn = gradcam(img_t.clone())

    # Predictions
    with torch.no_grad():
        vit_pred = vit_model(img_t).argmax(1).item()
        cnn_pred = cnn_model(img_t).argmax(1).item()

    img_np = np.array(image_pil.resize((img_size, img_size))) / 255.0

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(img_np)
    axes[0].set_title(
        f'Original\nActual: {class_names[idx2class[true_lbl]]}',
        fontsize=9)

    axes[1].imshow(img_np * 0.4 + plt.cm.hot(mask_vit)[:, :, :3] * 0.6)
    check = '✓' if vit_pred == true_lbl else '✗'
    axes[1].set_title(
        f'ViT Attention {check}\n{class_names[idx2class[vit_pred]]}',
        fontsize=9)

    axes[2].imshow(img_np * 0.4 + plt.cm.jet(mask_cnn)[:, :, :3] * 0.6)
    check = '✓' if cnn_pred == true_lbl else '✗'
    axes[2].set_title(
        f'CNN Grad-CAM {check}\n{class_names[idx2class[cnn_pred]]}',
        fontsize=9)

    axes[3].bar(['ViT', 'CNN'], [mask_vit.max(), mask_cnn.max()],
                color=['#534AB7', '#D85A30'])
    axes[3].set_title('Max. Activation', fontsize=9)
    axes[3].set_ylim(0, 1)

    for ax in axes[:3]:
        ax.axis('off')
    plt.tight_layout()
    return fig
