"""Checkpointed model loading for MAYA inference (no prediction).

Purpose
-------
Load the trained EfficientNet (via ModelFactory + checkpoint) once, place it on
the selected device, and keep it in eval mode for reuse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ai.engine.checkpoint import load_checkpoint
from ai.engine.device import resolve_device
from ai.inference.inference_config import InferenceConfig, get_inference_config
from ai.models.model_config import ModelConfig
from ai.models.model_factory import ModelFactory

logger = logging.getLogger("maya.ai.inference.model_loader")


@dataclass
class LoadedModel:
    """Handle to a device-placed eval-mode network."""

    model: nn.Module
    device: torch.device
    checkpoint_path: Path
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""


class ModelLoader:
    """Load and cache a single MAYA classifier instance."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or get_inference_config()
        self._cached: LoadedModel | None = None

    def load(self, *, force_reload: bool = False) -> LoadedModel:
        """Return a loaded model, reusing the in-memory cache when possible."""

        if self._cached is not None and not force_reload:
            logger.debug("Reusing cached inference model")
            return self._cached

        cfg = self.config
        device = resolve_device(cfg.device_preference)
        model_cfg = ModelConfig(
            project_root=cfg.project_root,
            model_name=cfg.model_name,
            image_size=cfg.image_size,
            device_preference=cfg.device_preference,
            pretrained_weights=True,
            freeze_backbone=True,
        )
        model = ModelFactory.create(config=model_cfg)
        ckpt = Path(cfg.checkpoint_path) if cfg.checkpoint_path else None
        meta: dict[str, Any] = {}
        if ckpt is None:
            raise FileNotFoundError("Inference checkpoint path is not configured")
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        meta = load_checkpoint(ckpt, model=model, map_location=device)
        logger.info("Inference model loaded from %s", ckpt)

        model.to(device)
        model.eval()
        self._cached = LoadedModel(
            model=model,
            device=device,
            checkpoint_path=ckpt,
            checkpoint_meta=meta,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )
        return self._cached

    def clear(self) -> None:
        """Drop the cached model (frees GPU/CPU references)."""

        self._cached = None
        logger.info("Cleared cached inference model")
