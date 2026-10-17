"""Visualization for segmentation predictions, separate from classifier XAI."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_localization_figure(image_path: str | Path, target_mask_path: str | Path, predicted_mask: np.ndarray, output_path: str | Path) -> Path:
    """Write input, target, prediction, and red prediction overlay panels."""
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))
    with Image.open(target_mask_path) as mm:
        target = np.asarray(mm.convert("L")) > 0
    pred = np.asarray(predicted_mask) >= 0.5
    if image.shape[:2] != (256, 256) or target.shape != (256, 256) or pred.shape != (256, 256):
        raise ValueError("Visualization expects prepared 256x256 image and masks")
    overlay = image.astype(np.float32).copy()
    overlay[pred] = 0.55 * overlay[pred] + 0.45 * np.array([255, 25, 25], dtype=np.float32)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for axis, data, title, cmap in zip(axes, (image, target, pred, overlay),
                                       ("Input", "Ground truth", "Prediction", "Prediction overlay"),
                                       (None, "gray", "gray", None)):
        axis.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
