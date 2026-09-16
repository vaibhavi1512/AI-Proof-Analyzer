"""Central configuration for MAYA evidence dataset engineering.

Purpose:
    Keep paths, split sizes, seeds, and loader defaults out of algorithm code.

Architecture:
    Pure configuration dataclasses — no I/O side effects besides path resolution
    and optional environment overrides.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


logger = logging.getLogger("maya.datasets.config")


class SplitName(str, Enum):
    """Supported dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ClassName(str, Enum):
    """Canonical class folder names written to processed/."""

    REAL = "REAL"
    FAKE = "FAKE"


# Map common folder name variants → canonical class
CLASS_ALIASES: dict[str, ClassName] = {
    "real": ClassName.REAL,
    "fake": ClassName.FAKE,
    "authentic": ClassName.REAL,
    "genuine": ClassName.REAL,
    "original": ClassName.REAL,
    "deepfake": ClassName.FAKE,
    "synthetic": ClassName.FAKE,
    "generated": ClassName.FAKE,
}

# ImageNet stats used only in runtime transforms (never baked into files)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)

PROJECT_VERSION = "0.2.5"
DEFAULT_DATASET_VERSION = "v1"
DEFAULT_DATASET_NAME = "MAYA-Kaggle-Balanced"


@dataclass(frozen=True)
class FutureDatasetSpec:
    """Descriptor for a future corpus that plugs into the same pipeline."""

    key: str
    display_name: str
    expected_layout: str
    notes: str


# Registered future corpora — adapters plug in without changing core modules
FUTURE_DATASETS: dict[str, FutureDatasetSpec] = {
    "faceforensics++": FutureDatasetSpec(
        key="faceforensics++",
        display_name="FaceForensics++",
        expected_layout="raw/<source>/<REAL|FAKE>/…",
        notes="Place extract under dataset/raw or set MAYA_RAW_DATASET_DIR; class aliases apply.",
    ),
    "celeb-df": FutureDatasetSpec(
        key="celeb-df",
        display_name="Celeb-DF",
        expected_layout="raw/<REAL|FAKE>/…",
        notes="Same inventory → integrity → sample → preprocess contract as the Kaggle corpus.",
    ),
}


def _project_root() -> Path:
    """Resolve repository root (…/MAYA)."""

    return Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class SplitBudget:
    """Per-class image counts for one split."""

    real: int
    fake: int


@dataclass
class DatasetConfig:
    """All tunable parameters for the dataset engineering pipeline."""

    project_root: Path = field(default_factory=_project_root)
    dataset_name: str = DEFAULT_DATASET_NAME
    project_version: str = PROJECT_VERSION
    dataset_version: str = DEFAULT_DATASET_VERSION

    raw_dir_override: Path | None = None
    processed_dir_override: Path | None = None
    catalogue_csv: Path | None = None

    random_seed: int = 42
    image_size: int = 224
    supported_extensions: tuple[str, ...] = DEFAULT_SUPPORTED_EXTENSIONS

    # Target balanced corpus (per class)
    train_budget: SplitBudget = field(default_factory=lambda: SplitBudget(1800, 1800))
    validation_budget: SplitBudget = field(default_factory=lambda: SplitBudget(300, 300))
    test_budget: SplitBudget = field(default_factory=lambda: SplitBudget(500, 500))

    # DataLoader defaults (CPU-friendly for 8 GB RAM)
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False

    # Visualization sampling
    sample_grid_per_class: int = 8

    # Version seal: streaming checksum over processed files (one at a time)
    compute_version_checksum: bool = True

    def __post_init__(self) -> None:
        if self.raw_dir_override is None:
            env_raw = _env_path("MAYA_RAW_DATASET_DIR")
            if env_raw is not None:
                self.raw_dir_override = env_raw

        if self.processed_dir_override is None:
            env_processed = _env_path("MAYA_PROCESSED_DATASET_DIR")
            if env_processed is None:
                raise ValueError(
                    "MAYA_PROCESSED_DATASET_DIR must be set explicitly; "
                    "refusing to fall back to a legacy dataset."
                )
            self.processed_dir_override = env_processed

        env_seed = os.getenv("MAYA_RANDOM_SEED")
        if env_seed is not None and env_seed.strip():
            try:
                self.random_seed = int(env_seed)
            except ValueError:
                pass

        self.batch_size = _env_int("MAYA_BATCH_SIZE", self.batch_size)
        self.num_workers = _env_int("MAYA_NUM_WORKERS", self.num_workers)
        self.image_size = _env_int("MAYA_IMAGE_SIZE", self.image_size)

        env_version = os.getenv("MAYA_DATASET_VERSION")
        if env_version:
            self.dataset_version = env_version.strip()

        if self.catalogue_csv is None:
            env_csv = _env_path("MAYA_CATALOGUE_CSV")
            self.catalogue_csv = env_csv or (self.project_root / "archive" / "FINAL_DATASET.csv")

        self._validate_processed_dataset()

    def _validate_processed_dataset(self) -> None:
        """Require an explicit, complete train/validation/test dataset root."""

        if self.processed_dir_override is None:
            raise ValueError("An explicit processed dataset directory is required.")
        if not self.processed_dir_override.is_dir():
            raise FileNotFoundError(
                f"MAYA_PROCESSED_DATASET_DIR does not exist: "
                f"{self.processed_dir_override}"
            )
        if self.processed_dir_override.name == "final" and not (
            self.processed_dir_override / "manifest.csv"
        ).is_file():
            raise FileNotFoundError(
                f"Final dataset manifest is missing: "
                f"{self.processed_dir_override / 'manifest.csv'}"
            )
        for split in SplitName:
            split_dir = self.processed_dir_override / split.value
            if not split_dir.is_dir():
                raise FileNotFoundError(
                    f"Dataset split is missing from {self.processed_dir_override}: "
                    f"{split_dir}"
                )
            for class_name in ClassName:
                class_dir = split_dir / class_name.value
                if not class_dir.is_dir():
                    raise FileNotFoundError(
                        f"Dataset class directory is missing: {class_dir}"
                    )
        counts = {
            split.value: {
                class_name.value: sum(
                    1
                    for path in (
                        self.processed_dir_override / split.value / class_name.value
                    ).iterdir()
                    if path.is_file() and path.suffix.lower() in self.supported_extensions
                )
                for class_name in ClassName
            }
            for split in SplitName
        }
        if any(count == 0 for split in counts.values() for count in split.values()):
            raise ValueError(
                f"Dataset at {self.processed_dir_override} must contain both "
                f"classes in every split: {counts}"
            )
        logger.info("DATASET ROOT: %s", self.processed_dir_override)
        for split, split_counts in counts.items():
            logger.info(
                "%s PATH: %s | REAL COUNT: %s | FAKE COUNT: %s",
                split.upper(),
                self.processed_dir_override / split,
                split_counts["REAL"],
                split_counts["FAKE"],
            )

    # --- Convenience split-size accessors (per-class totals) ---

    @property
    def train_size_per_class(self) -> int:
        return self.train_budget.real

    @property
    def validation_size_per_class(self) -> int:
        return self.validation_budget.real

    @property
    def test_size_per_class(self) -> int:
        return self.test_budget.real

    @property
    def dataset_root(self) -> Path:
        """Root for all dataset artefacts (raw / processed / versions / reports)."""

        return self.project_root / "dataset"

    @property
    def raw_dir(self) -> Path:
        """Immutable source dataset root."""

        if self.raw_dir_override is not None:
            return self.raw_dir_override
        return self.dataset_root / "raw"

    @property
    def processed_dir(self) -> Path:
        """Derived train/validation/test corpus root (active working set)."""

        if self.processed_dir_override is not None:
            return self.processed_dir_override
        raise ValueError("MAYA_PROCESSED_DATASET_DIR must be set explicitly.")

    @property
    def versions_dir(self) -> Path:
        """Sealed dataset version records (metadata + manifests)."""

        return self.dataset_root / "versions"

    @property
    def version_dir(self) -> Path:
        """Directory for the active dataset version id (e.g. versions/v1)."""

        return self.versions_dir / self.dataset_version

    @property
    def reports_dir(self) -> Path:
        """Statistical and validation report output directory."""

        return self.dataset_root / "reports"

    @property
    def future_datasets(self) -> dict[str, FutureDatasetSpec]:
        """Registered future corpora that reuse this pipeline."""

        return FUTURE_DATASETS

    def split_dir(self, split: SplitName | str) -> Path:
        """Return processed path for a split name."""

        name = split.value if isinstance(split, SplitName) else str(split)
        return self.processed_dir / name

    def class_dir(self, split: SplitName | str, cls: ClassName | str) -> Path:
        """Return processed path for a split/class pair."""

        class_name = cls.value if isinstance(cls, ClassName) else str(cls)
        return self.split_dir(split) / class_name

    def budget_for(self, split: SplitName) -> SplitBudget:
        """Return per-class budget for a split."""

        mapping = {
            SplitName.TRAIN: self.train_budget,
            SplitName.VALIDATION: self.validation_budget,
            SplitName.TEST: self.test_budget,
        }
        return mapping[split]

    @property
    def total_target_images(self) -> int:
        """Total images expected in the processed corpus."""

        total = 0
        for split in SplitName:
            budget = self.budget_for(split)
            total += budget.real + budget.fake
        return total


def get_dataset_config() -> DatasetConfig:
    """Factory for a default configuration instance."""

    return DatasetConfig()
