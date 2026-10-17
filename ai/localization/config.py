"""Configuration for the standalone manipulation segmentation baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class LocalizationConfig:
    project_dir: Path = field(default_factory=project_root)
    dataset_root: Path | None = None
    checkpoint_dir: Path | None = None
    image_size: int = 256
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 30
    threshold: float = 0.5
    random_seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    early_stopping_patience: int = 7

    def __post_init__(self) -> None:
        self.project_dir = Path(self.project_dir)
        self.dataset_root = Path(self.dataset_root) if self.dataset_root else self.project_dir / "artifacts" / "localization_dataset"
        self.checkpoint_dir = Path(self.checkpoint_dir) if self.checkpoint_dir else self.project_dir / "artifacts" / "checkpoints" / "localization"
        if self.image_size != 256:
            raise ValueError("Prepared localization data is fixed at 256x256.")
        if self.batch_size < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size, epochs, and learning_rate must be positive.")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        if self.bce_weight < 0 or self.dice_weight < 0 or self.bce_weight + self.dice_weight == 0:
            raise ValueError("Loss weights must be nonnegative and not both zero.")

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}
