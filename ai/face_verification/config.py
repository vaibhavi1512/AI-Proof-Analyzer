"""Configuration for MAYA Phase 5 face reference verification."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ai.models.model_config import _project_root


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class FaceVerificationConfig:
    """Tunables for detection, quality, embedding, and MATCH / NO_MATCH / INCONCLUSIVE."""

    engine: str = "opencv"
    model_name: str = "inception_resnet_v1"
    model_version: str = "vggface2"
    pretrained: str = "vggface2"
    device: str = "cpu"

    match_threshold: float = 0.70
    no_match_threshold: float = 0.50
    inconclusive_margin: float = 0.05

    min_face_size: int = 40
    min_sharpness: float = 25.0
    min_brightness: float = 20.0
    max_brightness: float = 235.0
    crop_padding: float = 0.20
    embed_size: int = 160

    project_root: Path = field(default_factory=_project_root)
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        if not 0.0 <= self.match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        if not 0.0 <= self.no_match_threshold <= 1.0:
            raise ValueError("no_match_threshold must be in [0, 1]")
        if self.no_match_threshold > self.match_threshold:
            raise ValueError("no_match_threshold must be <= match_threshold")


def get_face_verification_config() -> FaceVerificationConfig:
    cfg = FaceVerificationConfig(
        engine=_env_str("FACE_VERIFICATION_ENGINE", "opencv"),
        model_name=_env_str("FACE_VERIFICATION_MODEL", "inception_resnet_v1"),
        model_version=_env_str("FACE_VERIFICATION_MODEL_VERSION", "vggface2"),
        pretrained=_env_str("FACE_VERIFICATION_PRETRAINED", "vggface2"),
        device=_env_str("FACE_VERIFICATION_DEVICE", "cpu"),
        match_threshold=_env_float("FACE_VERIFICATION_MATCH_THRESHOLD", 0.70),
        no_match_threshold=_env_float("FACE_VERIFICATION_NO_MATCH_THRESHOLD", 0.50),
        inconclusive_margin=_env_float("FACE_VERIFICATION_INCONCLUSIVE_MARGIN", 0.05),
        min_face_size=_env_int("FACE_VERIFICATION_MIN_FACE_SIZE", 40),
        min_sharpness=_env_float("FACE_VERIFICATION_MIN_SHARPNESS", 25.0),
        min_brightness=_env_float("FACE_VERIFICATION_MIN_BRIGHTNESS", 20.0),
        max_brightness=_env_float("FACE_VERIFICATION_MAX_BRIGHTNESS", 235.0),
    )
    cache_raw = os.getenv("FACE_VERIFICATION_CACHE_DIR")
    if cache_raw and cache_raw.strip():
        cfg.cache_dir = Path(cache_raw.strip())
    return cfg


def config_from_mapping(values: dict | None) -> FaceVerificationConfig:
    """Build config from Flask ``app.config`` without hard-coding filesystem paths."""

    cfg = FaceVerificationConfig()
    if not values:
        return cfg
    engine = values.get("FACE_VERIFICATION_ENGINE")
    if engine:
        cfg.engine = str(engine)
    model = values.get("FACE_VERIFICATION_MODEL")
    if model:
        cfg.model_name = str(model)
    version = values.get("FACE_VERIFICATION_MODEL_VERSION")
    if version:
        cfg.model_version = str(version)
    pretrained = values.get("FACE_VERIFICATION_PRETRAINED")
    if pretrained:
        cfg.pretrained = str(pretrained)
    device = values.get("FACE_VERIFICATION_DEVICE")
    if device:
        cfg.device = str(device)
    if values.get("FACE_VERIFICATION_MATCH_THRESHOLD") is not None:
        cfg.match_threshold = float(values["FACE_VERIFICATION_MATCH_THRESHOLD"])
    if values.get("FACE_VERIFICATION_NO_MATCH_THRESHOLD") is not None:
        cfg.no_match_threshold = float(values["FACE_VERIFICATION_NO_MATCH_THRESHOLD"])
    if values.get("FACE_VERIFICATION_MIN_FACE_SIZE") is not None:
        cfg.min_face_size = int(values["FACE_VERIFICATION_MIN_FACE_SIZE"])
    if values.get("FACE_VERIFICATION_MIN_SHARPNESS") is not None:
        cfg.min_sharpness = float(values["FACE_VERIFICATION_MIN_SHARPNESS"])
    cache = values.get("FACE_VERIFICATION_CACHE_DIR")
    if cache:
        cfg.cache_dir = Path(cache)
    root = values.get("ROOT_DIR")
    if root:
        cfg.project_root = Path(root)
    if cfg.no_match_threshold > cfg.match_threshold:
        cfg.no_match_threshold = cfg.match_threshold
    return cfg
