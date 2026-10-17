"""Pixel metrics globally aggregated across every pixel in the batch."""
from __future__ import annotations

import torch


def segmentation_metrics(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7, *, from_logits: bool = False) -> dict[str, float]:
    """Compute thresholded global Dice/IoU/precision/recall/F1 and foreground rates.

    By default ``prediction`` is treated as probabilities. Set ``from_logits``
    when supplying raw model logits. Results are aggregated over all pixels.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    pred = prediction.detach().float()
    if from_logits:
        pred = torch.sigmoid(pred)
    truth = target.detach() >= 0.5
    predicted = pred >= threshold
    if predicted.shape != truth.shape:
        raise ValueError(f"Prediction and target shapes differ: {predicted.shape} vs {truth.shape}")
    tp = (predicted & truth).sum().item()
    fp = (predicted & ~truth).sum().item()
    fn = (~predicted & truth).sum().item()
    total = max(truth.numel(), 1)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "dice": float(dice), "iou": float(iou), "pixel_precision": float(precision),
        "pixel_recall": float(recall), "pixel_f1": float(f1),
        "predicted_foreground_percent": float(predicted.sum().item() * 100 / total),
        "ground_truth_foreground_percent": float(truth.sum().item() * 100 / total),
    }
