"""ImageNet EfficientNet-B0 with a new single FAKE logit.

The MAYA checkpoint is not loaded. ImageNet supplies the backbone initialization
when ``pretrained`` is true. Reloading a saved FF++ checkpoint uses
``pretrained=False`` and then the saved state dict.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .constants import CLASSIFIER_DROPOUT, IMAGE_SIZE
from .exceptions import VisualModelError

MODEL_NAME = "ffpp_visual_efficientnet_b0"


def build_visual_model(*, pretrained: bool) -> nn.Module:
    """Build EfficientNet-B0 and replace the ImageNet classifier with one FAKE logit."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    classifier = model.classifier
    if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Linear):
        raise VisualModelError("EfficientNet-B0 classifier is not the expected linear head.")
    in_features = int(classifier[-1].in_features)
    if in_features != 1280:
        raise VisualModelError(f"EfficientNet-B0 classifier expected 1280 inputs, found {in_features}.")
    model.classifier = nn.Sequential(
        nn.Dropout(p=CLASSIFIER_DROPOUT, inplace=True),
        nn.Linear(in_features, 1),
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def trainable_parameter_count(model: nn.Module) -> dict[str, int]:
    """Count backbone and classifier parameters. Frozen counts stay at zero for this model."""
    classifier_ids = {id(parameter) for parameter in model.classifier.parameters()}
    backbone = 0
    classifier = 0
    frozen = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        if not parameter.requires_grad:
            frozen += count
            continue
        if id(parameter) in classifier_ids:
            classifier += count
        else:
            backbone += count
    return {
        "backbone_trainable": backbone,
        "classifier_trainable": classifier,
        "frozen": frozen,
    }


def load_visual_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    """Reload an FF++ visual checkpoint without downloading ImageNet weights."""
    if not path.is_file():
        raise VisualModelError(f"Visual checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("model_name") != MODEL_NAME:
        raise VisualModelError(f"{path.name} is not an FF++ visual EfficientNet checkpoint.")
    model = build_visual_model(pretrained=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise VisualModelError(f"{path.name} is missing model weights.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def assert_input_output_shapes(model: nn.Module, batch_size: int) -> torch.Tensor:
    """Run one CPU forward pass and require logits shaped ``(batch, 1)``."""
    if batch_size < 1:
        raise VisualModelError("Forward pass requires at least one image.")
    frames = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(frames)
    finally:
        model.train(was_training)
    if tuple(logits.shape) != (batch_size, 1):
        raise VisualModelError(f"Expected logits {(batch_size, 1)}, got {tuple(logits.shape)}.")
    if not torch.isfinite(logits).all():
        raise VisualModelError("Forward pass produced non-finite logits.")
    return logits
