"""EfficientNet-B0 backbone builder for MAYA authenticity classification.

Why this module exists
----------------------
Isolate torchvision EfficientNet construction (weights, freeze, classifier)
from the factory and from any future training loop.

Architecture role
-----------------
AI Analysis Layer — model definition only. Called exclusively via
``ModelFactory``; never by Flask routes.

Design decisions
----------------
* EfficientNet-B0: strong accuracy / parameter trade-off for 8 GB RAM CPU hosts.
* ImageNet pretrained weights + frozen feature extractor = transfer learning.
* Binary head (``num_classes=2``) for REAL vs FAKE.
* No optimizer, loss, dataloader, or training loop in this file.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from ai.models.model_config import ModelConfig, get_model_config

logger = logging.getLogger("maya.ai.models.efficientnet")


def _build_classifier(in_features: int, num_classes: int, dropout: float = 0.2) -> nn.Sequential:
    """Create a compact classification head matching torchvision's layout."""

    return nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )


def build_efficientnet_b0(config: ModelConfig | None = None) -> nn.Module:
    """Load EfficientNet-B0, optionally freeze the backbone, replace the head.

    Returns:
        Configured ``nn.Module`` ready for a later training sprint (not trained here).
    """

    cfg = config or get_model_config()
    logger.info(
        "Building EfficientNet-B0 (pretrained=%s freeze=%s fine_tune_blocks=%s num_classes=%s)",
        cfg.pretrained_weights,
        cfg.freeze_backbone,
        cfg.fine_tune_last_feature_blocks,
        cfg.num_classes,
    )

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained_weights else None
    model = efficientnet_b0(weights=weights)

    if cfg.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        logger.info("Frozen EfficientNet-B0 feature extractor parameters")

        fine_tune_blocks = int(cfg.fine_tune_last_feature_blocks)
        if fine_tune_blocks < 0:
            raise ValueError("fine_tune_last_feature_blocks must be non-negative")
        if fine_tune_blocks:
            feature_blocks = list(model.features.children())
            if fine_tune_blocks > len(feature_blocks):
                raise ValueError(
                    "fine_tune_last_feature_blocks exceeds the number of "
                    f"EfficientNet-B0 feature blocks ({len(feature_blocks)})"
                )
            for block in feature_blocks[-fine_tune_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            logger.info(
                "Unfroze final %s EfficientNet-B0 feature block(s)", fine_tune_blocks
            )
            trainable_feature_names = [
                name for name, parameter in model.features.named_parameters()
                if parameter.requires_grad
            ]
            logger.info(
                "Trainable EfficientNet feature layers: %s",
                ", ".join(trainable_feature_names),
            )

    in_features = int(model.classifier[1].in_features)
    model.classifier = _build_classifier(in_features, cfg.num_classes)
    logger.info(
        "Replaced classifier head: Linear(%s → %s)",
        in_features,
        cfg.num_classes,
    )

    # Ensure we do not retain dangling references to ImageNet head weights
    # beyond the new Sequential assignment (single model instance only).
    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts without allocating tensors."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def summarize_model(model: nn.Module) -> str:
    """Produce a short text summary suitable for artefact files."""

    trainable, total = count_trainable_parameters(model)
    lines = [
        "MAYA EfficientNet-B0 Summary",
        "============================",
        f"Module type     : {type(model).__name__}",
        f"Total params    : {total:,}",
        f"Trainable params: {trainable:,}",
        f"Frozen params   : {total - trainable:,}",
        "",
        "Classifier:",
        f"  {model.classifier}",
    ]
    return "\n".join(lines)
