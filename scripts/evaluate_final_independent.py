"""Evaluate processed_final on the independent IMD2020 REAL/FAKE corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.datasets.dataloader import build_transforms
from ai.datasets.dataset import MayaImageDataset
from ai.engine.checkpoint import load_checkpoint
from ai.evaluation.metrics import (
    accuracy,
    balanced_accuracy,
    f1_score,
    precision,
    recall,
)
from ai.models.model_config import ModelConfig, ModelName
from ai.models.model_factory import ModelFactory

CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "processed_final" / "best.pt"
INDEPENDENT_DATASET_DIR = ROOT / "dataset" / "independent_imd2020_v1"
BATCH_SIZE = 8
REAL = 0
FAKE = 1


def main() -> int:
    """Load the processed_final checkpoint and print independent-corpus test metrics."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {CHECKPOINT_PATH}")
    if not INDEPENDENT_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Independent dataset not found: {INDEPENDENT_DATASET_DIR}"
        )

    # Reuse MAYA's standard evaluation transform: resize to 224, tensor conversion,
    # and ImageNet normalization.  The independent corpus is already its root.
    dataset = MayaImageDataset(
        INDEPENDENT_DATASET_DIR,
        transform=build_transforms("eval", image_size=224),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Keep the CPU EfficientNet-B0 evaluation configuration used by the processed_final evaluation.
    model_config = ModelConfig(
        project_root=ROOT,
        model_name=ModelName.EFFICIENTNET_B0.value,
        num_classes=2,
        image_size=224,
        batch_size=BATCH_SIZE,
        device_preference="cpu",
        pretrained_weights=False,
    )
    model_config.model_name = ModelName.EFFICIENTNET_B0.value
    model_config.num_classes = 2
    model_config.image_size = 224
    model_config.batch_size = BATCH_SIZE
    model_config.device_preference = "cpu"
    model = ModelFactory.create(config=model_config).to("cpu")
    load_checkpoint(CHECKPOINT_PATH, model=model, map_location="cpu")
    model.eval()

    actual: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images.to("cpu"))
            actual.extend(labels.cpu().tolist())
            # Preserve MAYA's standard two-class argmax decision; no threshold
            # is introduced or changed by this evaluator.
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    y_true = np.asarray(actual, dtype=int)
    y_pred = np.asarray(predicted, dtype=int)
    real_count = int(np.sum(y_true == REAL))
    fake_count = int(np.sum(y_true == FAKE))
    real_to_fake = int(np.sum((y_true == REAL) & (y_pred == FAKE)))
    fake_to_real = int(np.sum((y_true == FAKE) & (y_pred == REAL)))
    confusion = np.array(
        [
            [int(np.sum((y_true == REAL) & (y_pred == REAL))), real_to_fake],
            [fake_to_real, int(np.sum((y_true == FAKE) & (y_pred == FAKE)))],
        ]
    )

    print(f"REAL test images: {real_count}")
    print(f"FAKE test images: {fake_count}")
    print(f"Total test images: {len(y_true)}")
    print(f"Accuracy: {accuracy(y_true, y_pred):.6f}")
    print(f"Precision (FAKE): {precision(y_true, y_pred, positive_class=FAKE):.6f}")
    print(f"Recall (FAKE): {recall(y_true, y_pred, positive_class=FAKE):.6f}")
    print(f"F1 (FAKE): {f1_score(y_true, y_pred, positive_class=FAKE):.6f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy(y_true, y_pred, positive_class=FAKE):.6f}"
    )
    print("Confusion matrix (rows=true [REAL, FAKE], columns=predicted [REAL, FAKE]):")
    print(confusion)
    print(f"REAL -> FAKE errors: {real_to_fake}")
    print(f"FAKE -> REAL errors: {fake_to_real}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
