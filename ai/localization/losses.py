"""Numerically stable BCE-with-logits plus soft Dice segmentation loss."""
from __future__ import annotations

import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    """BCE mean + weighted soft Dice loss, with Dice reduced per image then mean.

    Dice loss is 1 - (2*sum(sigmoid(logits)*target)+eps) /
    (sum(probabilities)+sum(target)+eps). The smoothing epsilon avoids division
    instability for empty masks; BCEWithLogitsLoss uses a stable fused formula.
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        if min(bce_weight, dice_weight) < 0 or bce_weight + dice_weight == 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")
        self.bce_weight, self.dice_weight, self.eps = bce_weight, dice_weight, eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"Logits and target shapes differ: {logits.shape} vs {target.shape}")
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits).flatten(1)
        truth = target.float().flatten(1)
        dice = (2 * (probs * truth).sum(1) + self.eps) / (probs.sum(1) + truth.sum(1) + self.eps)
        return self.bce_weight * bce + self.dice_weight * (1 - dice.mean())
