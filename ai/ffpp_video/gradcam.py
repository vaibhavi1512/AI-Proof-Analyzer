"""Grad-CAM of the final LSTM FAKE logit at the last EfficientNet map.

The visual classifier is not used. Gradients start at the video logit, pass
through the LSTM and global pooling, and reach the last convolutional map.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import FEATURE_DIM, IMAGE_SIZE, NUM_FRAMES
from .exceptions import XaiError
from .lstm_model import VideoLstmV2
from .temporal_importance import normalize_temporal_importance, temporal_gradient_magnitude


def prediction_from_logit(logit: float, threshold: float = 0.5) -> tuple[int, str, float]:
    """Map one FAKE logit to a label. Scores at the threshold are FAKE."""
    probability = float(torch.sigmoid(torch.tensor(logit)))
    if not torch.isfinite(torch.tensor(probability)):
        raise XaiError("P(FAKE) is not finite.")
    label = 1 if probability >= threshold else 0
    name = "FAKE" if label == 1 else "REAL"
    return label, name, probability


def gradcam_from_logit(
    frames: torch.Tensor,
    visual: nn.Module,
    lstm: VideoLstmV2,
) -> dict:
    """Explain ``frames`` shaped ``(16, 3, 224, 224)`` with one backward pass."""
    if tuple(frames.shape) != (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE) or frames.dtype != torch.float32:
        raise XaiError(f"Expected {(NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)} float32, got {tuple(frames.shape)} {frames.dtype}.")
    if not torch.isfinite(frames).all():
        raise XaiError("Input frames contain non-finite values.")
    visual.eval()
    lstm.eval()
    classifier_calls = 0

    def _count_classifier(_module, _inputs, _output) -> None:
        nonlocal classifier_calls
        classifier_calls += 1

    hook = visual.classifier.register_forward_hook(_count_classifier)
    try:
        with torch.enable_grad():
            activation = visual.features(frames.detach().contiguous())
            if activation.ndim != 4 or activation.shape[0] != NUM_FRAMES or activation.shape[1] != FEATURE_DIM:
                raise XaiError(f"Last convolutional map is {tuple(activation.shape)}.")
            if activation.shape[2] < 2 or activation.shape[3] < 2:
                raise XaiError("Grad-CAM needs a spatial feature map.")
            pooled = visual.avgpool(activation)
            vectors = torch.flatten(pooled, 1)
            if tuple(vectors.shape) != (NUM_FRAMES, FEATURE_DIM):
                raise XaiError(f"Pooled features are {tuple(vectors.shape)}.")
            logit_tensor = lstm(vectors.unsqueeze(0)).reshape(())
            activation_grad, vector_grad = torch.autograd.grad(logit_tensor, [activation, vectors])
    finally:
        hook.remove()
    if classifier_calls != 0:
        raise XaiError("The visual classifier ran during the video explanation.")
    if activation_grad is None or vector_grad is None:
        raise XaiError("The final video logit did not connect to the convolutional map.")
    if not torch.isfinite(activation_grad).all() or not torch.isfinite(vector_grad).all():
        raise XaiError("Explanation gradients are not finite.")
    weights = activation_grad.mean(dim=(2, 3), keepdim=True)
    raw_map = torch.relu((weights * activation.detach()).sum(dim=1))
    if tuple(raw_map.shape) != (NUM_FRAMES, activation.shape[2], activation.shape[3]):
        raise XaiError(f"Grad-CAM map is {tuple(raw_map.shape)}.")
    if not torch.isfinite(raw_map).all():
        raise XaiError("Grad-CAM values are not finite.")
    display_map = F.interpolate(
        raw_map.unsqueeze(1),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    display_map = _unit_scale(display_map)
    magnitude = temporal_gradient_magnitude(vector_grad)
    logit = float(logit_tensor.detach())
    label, name, probability = prediction_from_logit(logit)
    return {
        "logit": logit,
        "p_fake": probability,
        "prediction": label,
        "prediction_name": name,
        "temporal_gradient_magnitude": [float(value) for value in magnitude],
        "temporal_importance": [float(value) for value in normalize_temporal_importance(magnitude)],
        "heatmaps": display_map.detach().cpu().contiguous(),
        "heatmap_raw_shape": [int(size) for size in raw_map.shape],
        "activation_shape": [int(size) for size in activation.shape],
        "spatial_target": "final_video_fake_logit",
        "spatial_layer": "efficientnet_b0.features",
    }


def _unit_scale(maps: torch.Tensor) -> torch.Tensor:
    """Scale each frame's heatmap to [0, 1] for display. A flat map stays zero."""
    lowest = maps.amin(dim=(1, 2), keepdim=True)
    highest = maps.amax(dim=(1, 2), keepdim=True)
    span = (highest - lowest).clamp_min(0)
    scaled = torch.where(span > 0, (maps - lowest) / span.clamp_min(1e-12), torch.zeros_like(maps))
    return scaled.clamp(0, 1)
