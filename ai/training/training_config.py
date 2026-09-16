"""Training profiles and configuration for MAYA Phase 3 Sprint 2.

Why this module exists
----------------------
Centralize epoch/batch/optimizer/scheduler settings and named profiles so the
trainer never hardcodes ``debug`` vs ``production`` numbers.

Architecture role
-----------------
Configuration adjunct to ``ModelConfig``. Profiles mutate a ``TrainingConfig``
instance; ``ModelFactory`` still builds the model from ``ModelConfig``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from ai.models.model_config import ModelConfig, _project_root, get_model_config

logger = logging.getLogger("maya.ai.training.config")


class TrainingProfile(str, Enum):
    """Named training intensity presets (easy to extend)."""

    DEBUG = "debug"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


# Profile overrides applied on top of TrainingConfig defaults
PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    TrainingProfile.DEBUG.value: {"epochs": 2, "batch_size": 4},
    TrainingProfile.DEVELOPMENT.value: {"epochs": 10, "batch_size": 8},
    TrainingProfile.PRODUCTION.value: {"epochs": 25, "batch_size": 8},
}


@dataclass
class TrainingConfig:
    """Full training-run configuration (memory-safe CPU defaults)."""

    profile: str = TrainingProfile.DEVELOPMENT.value
    model: ModelConfig = field(default_factory=get_model_config)

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 0
    random_seed: int = 42

    # Regularization / stability
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    monitor_metric: str = "val_loss"  # lower is better
    monitor_mode: str = "min"

    # Optional cap for smoke / unit tests (None = full loader)
    max_batches_per_epoch: int | None = None

    # Paths
    project_root: Path = field(default_factory=_project_root)
    checkpoint_dir: Path | None = None
    experiment_dir: Path | None = None
    artifact_dir: Path | None = None
    log_dir: Path | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "artifacts" / "checkpoints"
        if self.experiment_dir is None:
            self.experiment_dir = self.project_root / "artifacts" / "experiments"
        if self.artifact_dir is None:
            self.artifact_dir = self.project_root / "artifacts" / "phase3" / "sprint2"
        if self.log_dir is None:
            self.log_dir = self.project_root / "logs"

        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.experiment_dir = Path(self.experiment_dir)
        self.artifact_dir = Path(self.artifact_dir)
        self.log_dir = Path(self.log_dir)

        # Keep model config aligned with training batch / workers / epochs
        self.model = replace(
            self.model,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed,
            project_root=self.project_root,
            checkpoint_dir=self.checkpoint_dir,
            log_dir=self.log_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (paths as strings)."""

        raw = asdict(self)
        raw["project_root"] = str(self.project_root)
        raw["checkpoint_dir"] = str(self.checkpoint_dir)
        raw["experiment_dir"] = str(self.experiment_dir)
        raw["artifact_dir"] = str(self.artifact_dir)
        raw["log_dir"] = str(self.log_dir)
        raw["model"] = {
            "model_name": self.model.model_name,
            "num_classes": self.model.num_classes,
            "image_size": self.model.image_size,
            "pretrained_weights": self.model.pretrained_weights,
            "freeze_backbone": self.model.freeze_backbone,
            "fine_tune_last_feature_blocks": self.model.fine_tune_last_feature_blocks,
            "backbone_learning_rate": self.model.backbone_learning_rate,
            "device_preference": self.model.device_preference,
        }
        return raw


def apply_profile(profile: str, base: TrainingConfig | None = None) -> TrainingConfig:
    """Return a ``TrainingConfig`` with named profile overrides applied."""

    name = profile.strip().lower()
    if name not in PROFILE_OVERRIDES:
        known = ", ".join(sorted(PROFILE_OVERRIDES))
        raise ValueError(f"Unknown training profile '{profile}'. Known: {known}")

    overrides = PROFILE_OVERRIDES[name]
    if base is None:
        cfg = TrainingConfig(profile=name, **overrides)
    else:
        cfg = replace(base, profile=name, **overrides)

    logger.info("Applied training profile %s → epochs=%s batch_size=%s", name, cfg.epochs, cfg.batch_size)
    return cfg


def register_profile(name: str, **overrides: Any) -> None:
    """Register or replace a future profile without changing the trainer."""

    PROFILE_OVERRIDES[name.strip().lower()] = dict(overrides)
    logger.info("Registered training profile %s → %s", name, overrides)


def list_profiles() -> tuple[str, ...]:
    """Return known profile names."""

    return tuple(sorted(PROFILE_OVERRIDES.keys()))
