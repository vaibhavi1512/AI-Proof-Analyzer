"""Relative timestep contribution from the final video logit.

The values are gradient magnitudes. They are not probabilities and they are
not confidence.
"""

from __future__ import annotations

import torch

from .constants import NUM_FRAMES
from .exceptions import XaiError


def temporal_gradient_magnitude(feature_gradients: torch.Tensor) -> torch.Tensor:
    """Sum absolute gradients across the 1280 features of each timestep."""
    if feature_gradients.ndim != 2 or feature_gradients.shape[0] != NUM_FRAMES:
        raise XaiError(
            f"Expected gradients {(NUM_FRAMES, feature_gradients.shape[-1])}, "
            f"got {tuple(feature_gradients.shape)}."
        )
    if not torch.isfinite(feature_gradients).all():
        raise XaiError("Temporal gradients contain non-finite values.")
    magnitude = feature_gradients.detach().abs().sum(dim=1)
    if not torch.isfinite(magnitude).all():
        raise XaiError("Temporal gradient magnitudes are not finite.")
    return magnitude.cpu()


def normalize_temporal_importance(magnitude: torch.Tensor) -> torch.Tensor:
    """Scale 16 magnitudes for display. A zero total becomes equal shares."""
    if tuple(magnitude.shape) != (NUM_FRAMES,):
        raise XaiError(f"Expected {NUM_FRAMES} magnitudes, got {tuple(magnitude.shape)}.")
    if not torch.isfinite(magnitude).all() or bool((magnitude < 0).any()):
        raise XaiError("Temporal magnitudes must be finite and non-negative.")
    total = float(magnitude.sum())
    if total == 0.0:
        return torch.full((NUM_FRAMES,), 1.0 / NUM_FRAMES)
    return magnitude / total
