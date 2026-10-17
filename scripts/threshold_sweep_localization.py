"""Diagnostic threshold sweep for the saved localization model (no training)."""
from __future__ import annotations

from pathlib import Path

import torch

from ai.localization.config import LocalizationConfig
from ai.localization.dataset import create_localization_loader
from ai.localization.inference import load_localization_checkpoint, resolve_device
from ai.localization.metrics import segmentation_metrics
from ai.localization.train import validate_manifest_splits


THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.50)


def main() -> int:
    config = LocalizationConfig()
    checkpoint = config.checkpoint_dir / "localization_best.pt"
    split = "test"
    validate_manifest_splits(config.dataset_root)
    device = resolve_device(config.device)
    model, payload = load_localization_checkpoint(checkpoint, device)
    loader = create_localization_loader(
        config.dataset_root,
        split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )

    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, masks in loader:
            probabilities.append(torch.sigmoid(model(images.to(device))).cpu())
            targets.append(masks.cpu())

    probs = torch.cat(probabilities, dim=0)
    truth = torch.cat(targets, dim=0)
    results = [
        (threshold, segmentation_metrics(probs, truth, threshold=threshold))
        for threshold in THRESHOLDS
    ]

    print(f"Split: {split} | Checkpoint: {checkpoint} | Epoch: {payload.get('epoch')}")
    columns = (
        ("Threshold", "threshold"),
        ("Dice", "dice"),
        ("IoU", "iou"),
        ("Pixel precision", "pixel_precision"),
        ("Pixel recall", "pixel_recall"),
        ("Pixel F1", "pixel_f1"),
        ("Pred fg %", "predicted_foreground_percent"),
        ("GT fg %", "ground_truth_foreground_percent"),
    )
    widths = (9, 9, 9, 16, 13, 9, 11, 9)
    print(" | ".join(name.rjust(width) for (name, _), width in zip(columns, widths)))
    print("-+-".join("-" * width for width in widths))
    for threshold, metrics in results:
        values = {"threshold": threshold, **metrics}
        print(" | ".join(f"{values[key]:>{width}.4f}" for (_, key), width in zip(columns, widths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
