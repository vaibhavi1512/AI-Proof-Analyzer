"""Core training loop coordinator for MAYA (Sprint 3.2).

Why this module exists
----------------------
Own the epoch loop only: train batches → validate → callbacks → history.
Never instantiate models (``ModelFactory``) and never talk to Flask.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ai.engine.checkpoint import load_checkpoint, save_checkpoint
from ai.engine.device import get_device
from ai.engine.experiment import ExperimentTracker, read_dataset_version
from ai.engine.history import TrainingHistory
from ai.models.model_factory import ModelFactory
from ai.training.callbacks import (
    Callback,
    CallbackContext,
    EarlyStopping,
    LearningRateSchedulerHook,
    ModelCheckpoint,
)
from ai.training.training_config import TrainingConfig
from ai.training.validate import run_validation

logger = logging.getLogger("maya.ai.engine.trainer")


@dataclass
class TrainerResult:
    """Outcome of a ``Trainer.fit`` run."""

    history: TrainingHistory
    best_checkpoint: Path | None
    experiment_path: Path | None
    metrics: dict[str, Any]
    duration_seconds: float
    epochs_ran: int


@dataclass
class Trainer:
    """Coordinate training using injected loaders + ModelFactory model."""

    config: TrainingConfig
    train_loader: DataLoader
    val_loader: DataLoader
    callbacks: list[Callback] = field(default_factory=list)
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        self.device = get_device(self.config.model)
        self.history = TrainingHistory()
        self.model = ModelFactory.create(config=self.config.model)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "expected_class_counts"):
            logger.info(
                "Effective training class distribution per epoch: REAL=%s FAKE=%s",
                sampler.expected_class_counts[0],
                sampler.expected_class_counts[1],
            )
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        logger.info("Trainable parameters: %s", trainable)
        self.optimizer = self._build_optimizer()
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.epochs),
        )
        self.start_epoch = 1
        self.best_checkpoint: Path | None = None

        if not self.callbacks:
            self.callbacks = self._default_callbacks()

        if self.resume_from is not None:
            self._resume(self.resume_from)

    def _build_optimizer(self) -> AdamW:
        """Build AdamW, optionally using a lower LR for unfrozen features."""

        backbone_lr = self.config.model.backbone_learning_rate
        features = getattr(self.model, "features", None)
        if backbone_lr is not None and features is not None:
            feature_params = [p for p in features.parameters() if p.requires_grad]
            feature_ids = {id(p) for p in feature_params}
            other_params = [
                p
                for p in self.model.parameters()
                if p.requires_grad and id(p) not in feature_ids
            ]
            if feature_params and other_params:
                logger.info(
                    "Using differential learning rates: classifier/other=%s backbone=%s",
                    self.config.learning_rate,
                    backbone_lr,
                )
                return AdamW(
                    [
                        {"params": other_params, "lr": self.config.learning_rate},
                        {"params": feature_params, "lr": backbone_lr},
                    ],
                    weight_decay=self.config.weight_decay,
                )

        return AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _default_callbacks(self) -> list[Callback]:
        def _save(ctx: CallbackContext, path: Path) -> Path:
            saved = save_checkpoint(
                path,
                model=ctx.model,
                optimizer=ctx.optimizer,
                scheduler=ctx.scheduler,
                epoch=ctx.epoch,
                metrics=ctx.metrics,
                config=ctx.config.to_dict(),
            )
            self.best_checkpoint = saved
            return saved

        return [
            EarlyStopping(
                monitor=self.config.monitor_metric,
                mode=self.config.monitor_mode,
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
            ),
            ModelCheckpoint(
                save_fn=_save,
                directory=self.config.checkpoint_dir,
                monitor=self.config.monitor_metric,
                mode=self.config.monitor_mode,
                filename="best.pt",
            ),
            LearningRateSchedulerHook(),
        ]

    def _resume(self, path: Path) -> None:
        payload = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.start_epoch = int(payload.get("epoch", 0)) + 1
        logger.info("Resuming training from epoch %s", self.start_epoch)

    def _set_seed(self) -> None:
        seed = self.config.random_seed
        random.seed(seed)
        torch.manual_seed(seed)

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        max_batches = self.config.max_batches_per_epoch

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.config.grad_clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size

            del images, labels, outputs, loss

        if total_samples == 0:
            return {"train_loss": float("nan"), "train_accuracy": float("nan")}

        return {
            "train_loss": total_loss / total_samples,
            "train_accuracy": total_correct / total_samples,
        }

    def fit(self) -> TrainerResult:
        """Run the full training loop and return artefacts-ready results."""

        self._set_seed()
        started = time.perf_counter()
        logger.info(
            "Trainer start profile=%s epochs=%s batch=%s device=%s",
            self.config.profile,
            self.config.epochs,
            self.config.batch_size,
            self.device,
        )

        lr0 = float(self.optimizer.param_groups[0]["lr"])
        begin_ctx = CallbackContext(
            epoch=self.start_epoch - 1,
            metrics={},
            learning_rate=lr0,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            history=self.history,
        )
        for callback in self.callbacks:
            callback.on_train_begin(begin_ctx)

        epochs_ran = 0
        last_metrics: dict[str, float] = {}

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            logger.info("=== Epoch %s/%s ===", epoch, self.config.epochs)
            train_metrics = self._train_one_epoch()
            val_metrics = run_validation(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                max_batches=self.config.max_batches_per_epoch,
            )
            metrics = {**train_metrics, **val_metrics}
            lr = float(self.optimizer.param_groups[0]["lr"])
            metrics.setdefault("learning_rate", lr)

            self.history.append(epoch, metrics)
            last_metrics = metrics
            epochs_ran += 1

            ctx = CallbackContext(
                epoch=epoch,
                metrics=metrics,
                learning_rate=lr,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                config=self.config,
                history=self.history,
            )
            for callback in self.callbacks:
                callback.on_epoch_end(ctx)
                if ctx.stop_training:
                    break

            logger.info(
                "Epoch %s done train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                metrics.get("train_loss", float("nan")),
                metrics.get("val_loss", float("nan")),
                metrics.get("val_accuracy", float("nan")),
            )
            if ctx.stop_training:
                logger.info("Stopping early at epoch %s", epoch)
                break

        duration = time.perf_counter() - started
        end_ctx = CallbackContext(
            epoch=self.start_epoch + epochs_ran - 1,
            metrics=last_metrics,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            history=self.history,
        )
        for callback in self.callbacks:
            callback.on_train_end(end_ctx)

        # Always keep a last.pt for resume convenience
        last_path = Path(self.config.checkpoint_dir) / "last.pt"
        save_checkpoint(
            last_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=end_ctx.epoch,
            metrics=last_metrics,
            config=self.config.to_dict(),
        )

        dataset_version = read_dataset_version(self.config.project_root)
        tracker = ExperimentTracker(self.config.experiment_dir)
        experiment_path = tracker.write(
            profile=self.config.profile,
            dataset_version=dataset_version,
            model_name=self.config.model.model_name,
            hyperparameters={
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "grad_clip_norm": self.config.grad_clip_norm,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "loss": "CrossEntropyLoss",
                "sampling": "ClassBalancedCaseSampler",
            },
            metrics={
                "final": last_metrics,
                "best": self.history.best(
                    self.config.monitor_metric, self.config.monitor_mode
                ),
            },
            duration_seconds=duration,
            notes=self.config.notes,
            extras={"epochs_ran": epochs_ran, "device": str(self.device)},
        )

        return TrainerResult(
            history=self.history,
            best_checkpoint=self.best_checkpoint,
            experiment_path=experiment_path,
            metrics=last_metrics,
            duration_seconds=duration,
            epochs_ran=epochs_ran,
        )
