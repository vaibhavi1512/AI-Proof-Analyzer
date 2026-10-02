"""Select a few extracted stills and reuse image Grad-CAM on those JPEGs.

This is not a video explainer, a temporal CAM, or proof of manipulation.
It only chooses frames that already went through Phase 3 classification.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from ai.video.frame_extractor import ExtractedFrame

logger = logging.getLogger("maya.ai.video.frame_xai")

MAX_VIDEO_XAI_FRAMES = 3
_VIDEO_CONTAINER_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_STILL_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _as_frame(item: ExtractedFrame | dict[str, Any]) -> ExtractedFrame:
    if isinstance(item, ExtractedFrame):
        return item
    return ExtractedFrame(
        frame_number=int(item.get("frame_number", 0)),
        timestamp_seconds=float(item.get("timestamp_seconds") or 0.0),
        frame_path=str(item.get("frame_path") or ""),
        prediction=item.get("prediction"),
        confidence=item.get("confidence"),
        real_probability=item.get("real_probability"),
        fake_probability=item.get("fake_probability"),
    )


def _resolve_path(frame: ExtractedFrame, path_root: Path | None) -> Path:
    stored = Path(frame.frame_path)
    if stored.is_absolute() or path_root is None:
        return stored
    return path_root / stored


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().upper()
    if label in {"REAL", "FAKE"}:
        return label
    return None


def _confidence_key(frame: ExtractedFrame) -> float:
    value = _finite(frame.confidence)
    return value if value is not None else -1.0


def _fake_key(frame: ExtractedFrame) -> float:
    value = _finite(frame.fake_probability)
    return value if value is not None else -1.0


def is_video_container_path(path: Path) -> bool:
    return path.suffix.lower() in _VIDEO_CONTAINER_SUFFIXES


def resolve_still_frame_path(
    frame: ExtractedFrame | dict[str, Any],
    path_root: Path | str | None = None,
) -> Path | None:
    """Return a usable still-image path, or None if this is not a Grad-CAM input."""

    record = _as_frame(frame)
    root = Path(path_root).resolve() if path_root is not None else None
    path = _resolve_path(record, root)
    if is_video_container_path(path):
        logger.warning("Refusing video container for Grad-CAM: %s", path)
        return None
    if path.suffix.lower() not in _STILL_SUFFIXES:
        return None
    if not path.is_file():
        logger.info("Skipping missing video frame for XAI: %s", path)
        return None
    return path


def select_video_xai_frames(
    frames: Sequence[ExtractedFrame | dict[str, Any]],
    *,
    video_prediction: str,
    path_root: Path | str | None = None,
    max_frames: int = MAX_VIDEO_XAI_FRAMES,
) -> list[ExtractedFrame]:
    """Pick at most ``max_frames`` classified stills for Grad-CAM."""

    limit = max(0, int(max_frames))
    if limit == 0:
        return []
    usable: list[ExtractedFrame] = []
    for item in frames:
        record = _as_frame(item)
        if resolve_still_frame_path(record, path_root) is None:
            continue
        usable.append(record)

    fakes = [frame for frame in usable if _label(frame.prediction) == "FAKE"]
    if fakes:
        fakes.sort(key=_fake_key, reverse=True)
        return fakes[:limit]

    wanted = _label(video_prediction)
    matching = [frame for frame in usable if wanted and _label(frame.prediction) == wanted]
    if matching:
        matching.sort(key=_confidence_key, reverse=True)
        return matching[:1]

    usable.sort(key=_confidence_key, reverse=True)
    return usable[:1]


def explain_video_frames(
    frames: Sequence[ExtractedFrame],
    *,
    investigation_id: str,
    explainer: str,
    artifact_root: Path,
    path_root: Path | str | None,
    explain_fn: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run ``explain_fn`` once per selected JPEG into its own artifact directory."""

    results: list[dict[str, Any]] = []
    root = Path(artifact_root)
    for frame in frames:
        image_path = resolve_still_frame_path(frame, path_root)
        if image_path is None:
            continue
        if is_video_container_path(image_path):
            raise ValueError(f"Video container cannot be explained: {image_path}")
        frame_dir = root / "xai" / (explainer or "gradcam") / f"frame_{frame.frame_number:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        meta = explain_fn(
            image_path,
            investigation_id=investigation_id,
            explainer=explainer or "gradcam",
            artifact_dir=frame_dir,
        )
        payload = dict(meta)
        payload["frame_number"] = frame.frame_number
        payload["timestamp_seconds"] = float(frame.timestamp_seconds)
        payload["frame_path"] = str(image_path)
        payload["artifact_dir"] = str(frame_dir)
        results.append(payload)
    return results
