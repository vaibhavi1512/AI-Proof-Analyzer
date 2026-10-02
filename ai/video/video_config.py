"""Configuration for video metadata reading and frame extraction.

Phase 2 extracts approximately one frame per second. Authenticity
classification of those frames is deferred to Phase 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class VideoConfig:
    """Tunables for OpenCV metadata + ~1 FPS sampling.

    Attributes
    ----------
    target_sample_fps:
        Desired extracted frames per second of video time (~1 FPS).
    jpeg_quality:
        OpenCV JPEG quality for saved frames (1–100).
    max_extracted_frames:
        Safety cap so a misreported FPS cannot dump every decodeable frame.
    frame_image_extension:
        Suffix written for extracted stills (Phase 3 consumes these as images).
    """

    target_sample_fps: float = 1.0
    jpeg_quality: int = 95
    max_extracted_frames: int = 300
    frame_image_extension: str = ".jpg"

    def __post_init__(self) -> None:
        self.target_sample_fps = _env_float(
            "MAYA_VIDEO_SAMPLE_FPS", self.target_sample_fps
        )
        self.jpeg_quality = _env_int("MAYA_VIDEO_JPEG_QUALITY", self.jpeg_quality)
        self.max_extracted_frames = _env_int(
            "MAYA_VIDEO_MAX_FRAMES", self.max_extracted_frames
        )
        if self.target_sample_fps <= 0:
            raise ValueError("target_sample_fps must be > 0")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.max_extracted_frames < 1:
            raise ValueError("max_extracted_frames must be >= 1")
        ext = self.frame_image_extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        self.frame_image_extension = ext


def get_video_config() -> VideoConfig:
    return VideoConfig()
