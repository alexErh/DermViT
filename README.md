# DermViT – CNN vs. ViT on HAM10000

Skin lesion classification on the **HAM10000** dataset. The goal is to compare a
classic CNN against a Vision Transformer, and to study what transfer learning
contributes to each.

> **Research question:** Can a Vision Transformer outperform a classic CNN – and
> what does transfer learning contribute to CNN and ViT respectively?

Based on the paper *An Image is Worth 16×16 Words: Transformers for Image
Recognition at Scale* (Dosovitskiy et al., ICLR 2021).

Module: **Concepts of Deep Learning**

---

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
("Human Against Machine with 10000 training images") – 10,015 dermatoscopic
images across 7 lesion classes (heavily imbalanced; `nv` ≈ 67%).

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

## Configuration

All paths, hyperparameters and the class mapping live in `config.py`.
`config.init()` sets the random seed, selects the device (CPU/GPU) and loads the
metadata CSV into a prepared dataframe.
