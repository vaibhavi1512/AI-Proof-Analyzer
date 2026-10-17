"""Evaluate a saved localization checkpoint on an explicitly selected split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ai.localization.config import LocalizationConfig
from ai.localization.dataset import create_localization_loader
from ai.localization.inference import load_localization_checkpoint, resolve_device
from ai.localization.losses import BCEDiceLoss
from ai.localization.metrics import segmentation_metrics
from ai.localization.train import _run_epoch, validate_manifest_splits


def evaluate(config: LocalizationConfig, split: str = "test", checkpoint: Path | None = None) -> dict[str, float]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    validate_manifest_splits(config.dataset_root)
    device = resolve_device(config.device)
    path = checkpoint or config.checkpoint_dir / "localization_best.pt"
    model, payload = load_localization_checkpoint(path, device)
    loader = create_localization_loader(config.dataset_root, split, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    metrics = _run_epoch(model, loader, BCEDiceLoss(config.bce_weight, config.dice_weight), device, threshold=config.threshold)
    print(json.dumps({"split": split, "checkpoint": str(path), "epoch": payload.get("epoch"), **metrics}, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config = LocalizationConfig(dataset_root=args.dataset_root)
    if args.device:
        config.device = args.device
    evaluate(config, args.split, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
