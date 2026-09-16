"""Central AI / model configuration for MAYA Phase 3.

Why this module exists
----------------------
Training and inference must never scatter magic numbers (LR, epochs, paths).
A single dataclass keeps the AI Analysis Layer swappable and testable.

Architecture role
-----------------
Configuration Layer under ``ai/models/``. Downstream modules
(``efficientnet``, ``ModelFactory``, future training engine) read from here
only — they do not hardcode paths or hyper-parameters.

Design decisions
----------------
* ``dataclass`` for typed, documented fields with defaults safe for 8 GB RAM.
* Paths resolved from project root via ``pathlib``.
* Enum for supported model names so the factory stays consistent.
* Defaults align with Phase 2 processed corpus (224×224, 2 classes).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelName(str, Enum):
    """Registered model identifiers understood by ``ModelFactory``.

    Only ``EFFICIENTNET_B0`` is implemented in Sprint 3.1. Remaining values
    reserve the contract for future sprints without rewriting the factory API.
    """

    EFFICIENTNET_B0 = "efficientnet_b0"
    # Future (not yet implemented)
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    MOBILENET_V3 = "mobilenet_v3"
    VIT = "vit"


def _project_root() -> Path:
    """Resolve repository root (…/MAYA)."""

    return Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class ModelConfig:
    """All tunable AI parameters for MAYA model construction (Sprint 3.1).

    Fields use ALL-CAPS *names in docs*; runtime attributes are snake_case
    for PEP 8 while remaining the single source of truth.
    """

    # --- Identity / task ---
    model_name: str = ModelName.EFFICIENTNET_B0.value
    num_classes: int = 2
    image_size: int = 224
    random_seed: int = 42

    # --- Loader defaults (consumed later; unused this sprint) ---
    batch_size: int = 8
    num_workers: int = 0

    # --- Training placeholders (config only — no training code here) ---
    learning_rate: float = 1e-4
    epochs: int = 10

    # --- Weights / device preference ---
    pretrained_weights: bool = True
    freeze_backbone: bool = True
    fine_tune_last_feature_blocks: int = 2
    backbone_learning_rate: float | None = 1e-5
    device_preference: str = "auto"  # auto | cuda | mps | cpu

    # --- Artefact roots ---
    project_root: Path = field(default_factory=_project_root)
    checkpoint_dir: Path | None = None
    log_dir: Path | None = None

    def __post_init__(self) -> None:
        env_name = os.getenv("MAYA_MODEL_NAME")
        if env_name:
            self.model_name = env_name.strip().lower()

        self.num_classes = _env_int("MAYA_NUM_CLASSES", self.num_classes)
        self.image_size = _env_int("MAYA_IMAGE_SIZE", self.image_size)
        self.random_seed = _env_int("MAYA_RANDOM_SEED", self.random_seed)
        self.batch_size = _env_int("MAYA_BATCH_SIZE", self.batch_size)
        self.num_workers = _env_int("MAYA_NUM_WORKERS", self.num_workers)
        self.epochs = _env_int("MAYA_EPOCHS", self.epochs)
        self.learning_rate = _env_float("MAYA_LEARNING_RATE", self.learning_rate)

        env_device = os.getenv("MAYA_DEVICE")
        if env_device:
            self.device_preference = env_device.strip().lower()

        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.project_root / "artifacts" / "checkpoints"
        if self.log_dir is None:
            self.log_dir = self.project_root / "logs"

        # Ensure Path instances when callers pass strings
        self.project_root = Path(self.project_root)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)

    # --- Sprint brief aliases (UPPER_CASE accessors) ---

    @property
    def MODEL_NAME(self) -> str:
        return self.model_name

    @property
    def NUM_CLASSES(self) -> int:
        return self.num_classes

    @property
    def IMAGE_SIZE(self) -> int:
        return self.image_size

    @property
    def RANDOM_SEED(self) -> int:
        return self.random_seed

    @property
    def BATCH_SIZE(self) -> int:
        return self.batch_size

    @property
    def LEARNING_RATE(self) -> float:
        return self.learning_rate

    @property
    def EPOCHS(self) -> int:
        return self.epochs

    @property
    def NUM_WORKERS(self) -> int:
        return self.num_workers

    @property
    def CHECKPOINT_DIR(self) -> Path:
        return self.checkpoint_dir  # type: ignore[return-value]

    @property
    def LOG_DIR(self) -> Path:
        return self.log_dir  # type: ignore[return-value]

    @property
    def DEVICE(self) -> str:
        return self.device_preference

    @property
    def PRETRAINED_WEIGHTS(self) -> bool:
        return self.pretrained_weights


def get_model_config() -> ModelConfig:
    """Factory for the default model configuration instance."""

    return ModelConfig()
