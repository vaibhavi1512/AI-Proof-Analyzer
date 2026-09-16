"""DataLoader factory utilities for MAYA datasets."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Literal

from PIL import Image
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

from ai.datasets.dataset import MayaImageDataset, build_split_dataset
from ai.datasets.dataset_config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DatasetConfig,
    SplitName,
    get_dataset_config,
)

logger = logging.getLogger("maya.datasets.dataloader")

TransformName = Literal["none", "tensor", "train", "eval"]


class ClassBalancedCaseSampler(Sampler[int]):
    """Sample classes evenly while selecting IMD2020 cases uniformly."""

    def __init__(self, dataset: MayaImageDataset, *, seed: int = 42) -> None:
        if len(dataset.samples) != len(dataset.sample_group_ids):
            raise ValueError("Dataset samples and case metadata must have equal lengths.")
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset)
        self.indices_by_class_and_case: dict[int, dict[str, list[int]]] = {
            0: defaultdict(list),
            1: defaultdict(list),
        }
        for index, ((_, label), case_id) in enumerate(
            zip(dataset.samples, dataset.sample_group_ids)
        ):
            if label not in (0, 1):
                raise ValueError(f"Unsupported dataset label: {label}")
            self.indices_by_class_and_case[label][case_id].append(index)
        if any(not groups for groups in self.indices_by_class_and_case.values()):
            raise ValueError("Class-balanced sampling requires both REAL and FAKE classes.")
        self.expected_class_counts = {
            0: (self.num_samples + 1) // 2,
            1: self.num_samples // 2,
        }

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        labels = [
            position % 2
            for position in range(self.num_samples)
        ]
        for label in labels:
            groups = list(self.indices_by_class_and_case[label])
            group_index = int(
                torch.randint(len(groups), (1,), generator=generator).item()
            )
            indices = self.indices_by_class_and_case[label][groups[group_index]]
            sample_index = int(
                torch.randint(len(indices), (1,), generator=generator).item()
            )
            yield indices[sample_index]

    def __len__(self) -> int:
        return self.num_samples


def build_transforms(
    name: TransformName,
    *,
    image_size: int = 224,
) -> Callable[[Image.Image], Any] | None:
    """Return a torchvision transform pipeline by name.

    Notes:
        Processed images are already 224×RGB. Normalization is applied only
        in ``train`` / ``eval`` transforms — never baked into saved files.
    """

    if name == "none":
        return None
    if name == "tensor":
        return transforms.ToTensor()
    if name == "eval":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    if name == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    raise ValueError(f"Unknown transform selection: {name}")


def create_dataloader(
    split: SplitName | str,
    *,
    config: DatasetConfig | None = None,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int | None = None,
    image_size: int | None = None,
    transform_name: TransformName = "tensor",
    transform: Callable[[Image.Image], Any] | None = None,
    balanced_sampling: bool = True,
) -> DataLoader:
    """Create a configurable DataLoader for a processed split.

    Args:
        split: ``train`` / ``validation`` / ``test``.
        config: Optional dataset configuration.
        batch_size: Override batch size.
        shuffle: Override shuffle (defaults to True for train).
        num_workers: Override worker processes (keep 0 on 8 GB RAM if unsure).
        image_size: Transform resize target.
        transform_name: Built-in transform preset when ``transform`` is None.
        transform: Explicit transform callable (wins over ``transform_name``).
    """

    cfg = config or get_dataset_config()
    split_name = SplitName(split) if not isinstance(split, SplitName) else split

    size = image_size or cfg.image_size
    chosen_transform = transform
    if chosen_transform is None:
        chosen_transform = build_transforms(transform_name, image_size=size)

    dataset: MayaImageDataset = build_split_dataset(
        split_name,
        config=cfg,
        transform=chosen_transform,
    )

    sampler = None
    if split_name == SplitName.TRAIN and balanced_sampling:
        sampler = ClassBalancedCaseSampler(dataset, seed=cfg.random_seed)
        shuffle = False
        logger.info(
            "Class-balanced case sampler: REAL=%s FAKE=%s cases=(%s,%s)",
            sampler.expected_class_counts[0],
            sampler.expected_class_counts[1],
            len(sampler.indices_by_class_and_case[0]),
            len(sampler.indices_by_class_and_case[1]),
        )
    elif shuffle is None:
        shuffle = split_name == SplitName.TRAIN

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        pin_memory=cfg.pin_memory,
    )
    logger.info(
        "DataLoader ready split=%s batch=%s shuffle=%s n=%s",
        split_name.value,
        loader.batch_size,
        shuffle,
        len(dataset),
    )
    return loader
