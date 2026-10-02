"""Classify Phase 2 extracted frames with the existing image inference pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from ai.inference.pipeline import InferencePipeline
from ai.video.frame_extractor import ExtractedFrame, VideoExtractionResult

logger = logging.getLogger("maya.ai.video.frame_analyzer")


def _resolve_frame_path(frame: ExtractedFrame, path_root: Path | None) -> Path:
    stored = Path(frame.frame_path)
    if stored.is_absolute() or path_root is None:
        return stored
    return path_root / stored


def analyze_extracted_frames(
    result: VideoExtractionResult,
    *,
    path_root: Path | str | None = None,
    pipeline: InferencePipeline,
) -> VideoExtractionResult:
    """Fill REAL/FAKE fields on each extracted frame using ``pipeline``.

    Uses the same preprocess / EfficientNet / threshold path as still-image
    analysis. Does not aggregate a video-level verdict.
    """

    root = Path(path_root).resolve() if path_root is not None else None
    loaded = pipeline.loader.load()
    logger.info(
        "Scoring %s video frames checkpoint=%s",
        len(result.frames),
        loaded.checkpoint_path,
    )
    for frame in result.frames:
        image_path = _resolve_frame_path(frame, root)
        if image_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            raise ValueError(
                f"Video container cannot be classified as a frame: {image_path.name}"
            )
        decision = pipeline.classify_image(image_path)
        frame.prediction = decision.predicted_label
        frame.confidence = float(decision.confidence)
        frame.real_probability = float(decision.real_probability)
        frame.fake_probability = float(decision.fake_probability)
    return result
