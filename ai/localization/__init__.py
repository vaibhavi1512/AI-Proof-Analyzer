"""Isolated pixel-level manipulation localization tools.

This package remains separate from authenticity classification and production
inference. Dataset preparation and segmentation can be used independently.
"""

from ai.localization.prepare_dataset import (
    LocalizationPrepConfig,
    LocalizationPrepError,
    PrepReport,
    prepare_localization_dataset,
)

__all__ = [
    "LocalizationPrepConfig",
    "LocalizationPrepError",
    "PrepReport",
    "prepare_localization_dataset",
]
