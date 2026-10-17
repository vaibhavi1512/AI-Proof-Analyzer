"""Train the isolated weighted-BCE + Dice localization experiment."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn

from ai.localization.config import LocalizationConfig
from ai.localization.dataset import create_localization_loader
from ai.localization.inference import resolve_device
from ai.localization.model import LocalizationUNet
from ai.localization.train import _run_epoch, _seed_everything, validate_manifest_splits


EXPERIMENT_ID = "weighted_bce_dice_posw3"
POS_WEIGHT = 3.0
logger = logging.getLogger("maya.localization.weighted_experiment")


class WeightedBCEDiceLoss(nn.Module):
    """Baseline soft Dice objective with BCE positive-class weight set to 3."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([POS_WEIGHT], dtype=torch.float32, device=device)
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"Logits and target shapes differ: {logits.shape} vs {target.shape}")
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits).flatten(1)
        truth = target.float().flatten(1)
        eps = 1e-6
        dice = (2 * (probs * truth).sum(1) + eps) / (
            probs.sum(1) + truth.sum(1) + eps
        )
        return bce + (1 - dice.mean())


def run_experiment(config: LocalizationConfig) -> dict:
    validate_manifest_splits(config.dataset_root)
    _seed_everything(config.random_seed)
    device = resolve_device(config.device)

    # BCEWithLogitsLoss uses pos_weight only on the positive target term:
    # -[pos_weight*y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))].
    criterion = WeightedBCEDiceLoss(device)
    actual_pos_weight = criterion.bce.pos_weight.detach().cpu().tolist()
    if actual_pos_weight != [POS_WEIGHT]:
        raise RuntimeError(f"Unexpected BCE pos_weight: {actual_pos_weight}")
    probe_logits = torch.zeros((1,), device=device)
    weighted_positive = criterion.bce(probe_logits, torch.ones_like(probe_logits))
    weighted_negative = criterion.bce(probe_logits, torch.zeros_like(probe_logits))
    unweighted_term = torch.nn.functional.binary_cross_entropy_with_logits(
        probe_logits, torch.ones_like(probe_logits)
    )
    if not torch.isclose(weighted_positive, unweighted_term * POS_WEIGHT):
        raise RuntimeError("BCE pos_weight is not weighting the positive class by 3.0")
    if not torch.isclose(weighted_negative, unweighted_term):
        raise RuntimeError("BCE pos_weight unexpectedly changed the negative-class term")
    print(
        f"Experiment: {EXPERIMENT_ID} | BCE pos_weight={POS_WEIGHT} applies to positive/manipulated pixels only | "
        "verified against positive/negative BCE terms | BCE weight=1.0 | Dice weight=1.0"
    )

    train_loader = create_localization_loader(
        config.dataset_root, "train", batch_size=config.batch_size, num_workers=config.num_workers
    )
    val_loader = create_localization_loader(
        config.dataset_root, "val", batch_size=config.batch_size,
        num_workers=config.num_workers, shuffle=False
    )
    model = LocalizationUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer, threshold=config.threshold
        )
        val_metrics = _run_epoch(
            model, val_loader, criterion, device, threshold=config.threshold
        )
        history.append({
            "experiment_id": EXPERIMENT_ID,
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        })
        logger.info(
            "experiment=%s epoch=%d train_loss=%.5f val_loss=%.5f val_dice=%.5f",
            EXPERIMENT_ID, epoch, train_metrics["loss"], val_metrics["loss"], val_metrics["dice"]
        )

        payload = {
            "experiment_id": EXPERIMENT_ID,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "validation_metrics": val_metrics,
            "config": {
                **config.to_dict(),
                "base_channels": 16,
                "experiment_id": EXPERIMENT_ID,
                "loss": "BCEWithLogitsLoss(pos_weight=3.0) + soft Dice",
                "pos_weight": POS_WEIGHT,
                "pos_weight_applies_to": "positive/manipulated class only",
                "bce_weight": 1.0,
                "dice_weight": 1.0,
            },
        }
        torch.save(payload, config.checkpoint_dir / f"{EXPERIMENT_ID}_final.pt")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss, best_epoch, stale_epochs = val_metrics["loss"], epoch, 0
            torch.save(payload, config.checkpoint_dir / f"{EXPERIMENT_ID}_best.pt")
        else:
            stale_epochs += 1
        if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
            logger.info("experiment=%s early stopping after %d non-improving epochs", EXPERIMENT_ID, stale_epochs)
            break

    history_path = config.checkpoint_dir / f"{EXPERIMENT_ID}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(config.checkpoint_dir / f"{EXPERIMENT_ID}_best.pt"),
        "history": history,
    }
    print(
        f"Completed {EXPERIMENT_ID}: best epoch={best_epoch}, val_loss={best_val_loss:.5f}, "
        f"checkpoint={result['best_checkpoint']}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints" / "localization",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = LocalizationConfig(
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=8,
        learning_rate=1e-4,
        epochs=30,
        early_stopping_patience=7,
        bce_weight=1.0,
        dice_weight=1.0,
        device=args.device,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
