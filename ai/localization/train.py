"""Train the separate U-Net localization baseline (never starts automatically)."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from ai.localization.config import LocalizationConfig
from ai.localization.dataset import create_localization_loader
from ai.localization.inference import resolve_device
from ai.localization.losses import BCEDiceLoss
from ai.localization.metrics import segmentation_metrics
from ai.localization.model import LocalizationUNet

logger = logging.getLogger("maya.localization.train")


def validate_manifest_splits(dataset_root: Path) -> None:
    manifest = dataset_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Required split manifest missing: {manifest}")
    folders: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            folder = row.get("source_folder", "")
            if split in folders and folder:
                folders[split].add(folder)
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = folders[first] & folders[second]
        if overlap:
            raise ValueError(f"Parent-folder leakage between {first} and {second}: {sorted(overlap)[:5]}")
    if not all(folders.values()):
        raise ValueError("Manifest does not contain source folders for each split")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    pixels = 0
    metric_counts = {k: 0.0 for k in ("tp", "fp", "fn", "pred", "truth")}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.item()) * images.shape[0]
            pixels += images.shape[0]
            pred = torch.sigmoid(logits) >= threshold
            truth = masks >= 0.5
            metric_counts["tp"] += (pred & truth).sum().item()
            metric_counts["fp"] += (pred & ~truth).sum().item()
            metric_counts["fn"] += (~pred & truth).sum().item()
            metric_counts["pred"] += pred.sum().item()
            metric_counts["truth"] += truth.sum().item()
    tp, fp, fn = (metric_counts[k] for k in ("tp", "fp", "fn"))
    eps = 1e-7
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "loss": loss_sum / max(pixels, 1),
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "pixel_precision": precision, "pixel_recall": recall,
        "pixel_f1": 2 * precision * recall / (precision + recall + eps),
        "predicted_foreground_percent": 100 * metric_counts["pred"] / max(pixels * 256 * 256, 1),
        "ground_truth_foreground_percent": 100 * metric_counts["truth"] / max(pixels * 256 * 256, 1),
    }


def train(config: LocalizationConfig) -> dict:
    validate_manifest_splits(config.dataset_root)
    _seed_everything(config.random_seed)
    device = resolve_device(config.device)
    train_loader = create_localization_loader(config.dataset_root, "train", batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = create_localization_loader(config.dataset_root, "val", batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    model = LocalizationUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = BCEDiceLoss(config.bce_weight, config.dice_weight)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer, config.threshold)
        val_metrics = _run_epoch(model, val_loader, criterion, device, None, config.threshold)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        logger.info("epoch=%d train_loss=%.5f val_loss=%.5f val_dice=%.5f", epoch, train_metrics["loss"], val_metrics["loss"], val_metrics["dice"])
        payload = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                   "validation_metrics": val_metrics, "config": {**config.to_dict(), "base_channels": 16}}
        torch.save(payload, config.checkpoint_dir / "localization_final.pt")
        if val_metrics["loss"] < best:
            best, best_epoch, stale = val_metrics["loss"], epoch, 0
            torch.save(payload, config.checkpoint_dir / "localization_best.pt")
        else:
            stale += 1
        if config.early_stopping_patience and stale >= config.early_stopping_patience:
            logger.info("Early stopping after %d non-improving epochs", stale)
            break
    (config.checkpoint_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_epoch": best_epoch, "best_val_loss": best, "history": history}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    config = LocalizationConfig(dataset_root=args.dataset_root, checkpoint_dir=args.checkpoint_dir)
    for name in ("epochs", "batch_size", "device"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
