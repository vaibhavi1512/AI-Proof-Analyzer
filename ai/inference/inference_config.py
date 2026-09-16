"""Inference configuration for MAYA Sprint 3.4.

Purpose
-------
Centralize threshold, confidence bands, paths, and class names so inference
modules never hardcode forensic cut-offs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ai.datasets.dataset_config import DEFAULT_SUPPORTED_EXTENSIONS
from ai.models.model_config import ModelName, _project_root


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ConfidenceBands:
    """Inclusive lower bounds for confidence levels (fraction 0–1)."""

    very_high: float = 0.95  # ≥95% → Very High
    high: float = 0.85  # 85–95% → High
    medium: float = 0.70  # 70–85% → Medium
    # below medium → Low


@dataclass
class InferenceConfig:
    """Tunables for single-image / batch investigation inference."""

    threshold: float = 0.5
    positive_class: int = 1  # FAKE
    class_names: tuple[str, ...] = ("REAL", "FAKE")
    confidence_bands: ConfidenceBands = field(default_factory=ConfidenceBands)

    image_size: int = 224
    supported_extensions: tuple[str, ...] = DEFAULT_SUPPORTED_EXTENSIONS
    device_preference: str = "auto"
    model_name: str = ModelName.EFFICIENTNET_B0.value
    model_version: str = "processed_final-best"

    project_root: Path = field(default_factory=_project_root)
    checkpoint_path: Path | None = None
    artifact_dir: Path | None = None
    id_state_path: Path | None = None

    def __post_init__(self) -> None:
        self.threshold = _env_float("MAYA_INFER_THRESHOLD", self.threshold)
        self.project_root = Path(self.project_root)

        if self.checkpoint_path is None:
            self.checkpoint_path = (
                self.project_root
                / "artifacts"
                / "checkpoints"
                / "processed_final"
                / "best.pt"
            )
        self.checkpoint_path = Path(self.checkpoint_path)

        if self.artifact_dir is None:
            self.artifact_dir = self.project_root / "artifacts" / "phase3" / "sprint4"
        self.artifact_dir = Path(self.artifact_dir)

        if self.id_state_path is None:
            self.id_state_path = self.artifact_dir / "investigation_id_state.json"
        self.id_state_path = Path(self.id_state_path)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")


def get_inference_config() -> InferenceConfig:
    return InferenceConfig()
