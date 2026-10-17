"""Isolated ManTraNet pixel-localization evaluation on one IMD2020 sample."""
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "external" / "ManTraNet-pytorch" / "MantraNet"
IMAGE_PATH = ROOT / "dataset 16.09" / "IMD2020" / "z3" / "00009_fake.jpg"
MASK_PATH = ROOT / "dataset 16.09" / "IMD2020" / "z3" / "00009_fake_mask.png"
OUTPUT_PATH = ROOT / "artifacts" / "mantranet_localization" / "00009_fake_evaluation.png"
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)


def main():
    if not IMAGE_PATH.is_file() or not MASK_PATH.is_file():
        raise FileNotFoundError(f"Missing evaluation input: {IMAGE_PATH} or {MASK_PATH}")

    # mantranet.py loads IMTFEv4.pt using a path relative to the process cwd.
    previous_cwd = Path.cwd()
    os.chdir(MODEL_DIR)
    try:
        sys.path.insert(0, str(MODEL_DIR))
        from mantranet import MantraNet

        device = torch.device("cpu")
        model = MantraNet(device=device)
        state = torch.load("MantraNetv4.pt", map_location=device)
        model.load_state_dict(state)
    finally:
        os.chdir(previous_cwd)

    model.to(device).eval()
    rgb = np.asarray(Image.open(IMAGE_PATH).convert("RGB"), dtype=np.uint8)
    mask_rgb = np.asarray(Image.open(MASK_PATH).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != mask_rgb.shape[:2]:
        raise ValueError(f"Image/mask dimension mismatch: {rgb.shape} vs {mask_rgb.shape}")

    # Inspect binary mask encoding before converting it to a foreground map.
    mask_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    if not np.all(np.isin(mask_colors, ([0, 0, 0], [255, 255, 255]))):
        raise ValueError(f"Expected a black/white mask; found colors such as {mask_colors[:8]}")
    truth = np.all(mask_rgb == 255, axis=2)

    # ManTraNet's forward normalizes internally from 0..255; keep that input scale.
    image_tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
    with torch.inference_mode():
        output = model(image_tensor.to(device))
    if output.ndim != 4 or output.shape[0] != 1 or output.shape[1] != 1:
        raise ValueError(f"Unexpected model output shape: {tuple(output.shape)}")
    prediction = output[0, 0].cpu().numpy()
    if prediction.shape != truth.shape:
        raise ValueError(f"Prediction/mask dimension mismatch: {prediction.shape} vs {truth.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")

    print(f"Image and mask dimensions: {rgb.shape[1]} x {rgb.shape[0]}")
    print(f"Model output shape: {tuple(output.shape)}")
    print(f"Prediction range: min={prediction.min():.6f}, max={prediction.max():.6f}")
    print("Output is sigmoid-bounded [0, 1], but is treated as a score, not a calibrated probability.")
    print("Threshold  Dice       IoU")
    for threshold in THRESHOLDS:
        predicted = prediction >= threshold
        intersection = np.count_nonzero(predicted & truth)
        pred_count = np.count_nonzero(predicted)
        truth_count = np.count_nonzero(truth)
        union = pred_count + truth_count - intersection
        dice = (2 * intersection / (pred_count + truth_count)) if pred_count + truth_count else 1.0
        iou = (intersection / union) if union else 1.0
        print(f"{threshold:>9.1f}  {dice:.6f}  {iou:.6f}")

    # Show a representative 0.2-threshold overlay, matching ManTraNet's demo convention.
    overlay = rgb.astype(np.float32) / 255.0
    suspicious = prediction >= 0.2
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    overlay[suspicious] = 0.55 * overlay[suspicious] + 0.45 * red[suspicious]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Fake image")
    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-truth mask")
    im = axes[2].imshow(prediction, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("ManTraNet score map")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (score ≥ 0.2)")
    for ax in axes:
        ax.axis("off")
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
