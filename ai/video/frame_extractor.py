"""Sample approximately 1 FPS from a video and write still frames."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai.video.exceptions import VideoPathError, VideoUnreadableError
from ai.video.metadata import (
    VideoMetadata,
    read_video_metadata,
    sample_interval_frames,
    timestamp_for_frame,
)
from ai.video.video_config import VideoConfig, get_video_config

logger = logging.getLogger("maya.ai.video.frames")

METADATA_FILENAME = "metadata.json"


@dataclass
class ExtractedFrame:
    """One sampled frame, with empty prediction slots for Phase 3."""

    frame_number: int
    timestamp_seconds: float
    frame_path: str
    prediction: str | None = None
    confidence: float | None = None
    real_probability: float | None = None
    fake_probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VideoExtractionResult:
    """Metadata plus sampled frames written under ``output_dir``."""

    metadata: VideoMetadata
    frames: list[ExtractedFrame] = field(default_factory=list)
    output_dir: str = ""
    metadata_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "output_dir": self.output_dir,
            "metadata_path": self.metadata_path,
        }


def _safe_frame_path(output_dir: Path, frame_number: int, extension: str) -> Path:
    name = f"frame_{frame_number:06d}{extension}"
    if "/" in name or "\\" in name or name.startswith(".") or ".." in name:
        raise VideoPathError("Invalid frame filename")
    root = output_dir.resolve()
    dest = (root / name).resolve()
    try:
        if not dest.is_relative_to(root):
            raise VideoPathError("Frame path escaped output directory")
    except (OSError, ValueError) as exc:
        raise VideoPathError("Frame path escaped output directory") from exc
    return dest


def _write_metadata_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_frames(
    video_path: Path | str,
    output_dir: Path | str,
    *,
    config: VideoConfig | None = None,
) -> VideoExtractionResult:
    """Decode ``video_path``, sample ~1 FPS, and save JPEGs under ``output_dir``.

    Sampling walks the file sequentially (more reliable than codec seeking) and
    keeps every Nth frame where N approximates one frame per second. Short
    clips always yield at least the first readable frame when one exists.
    """

    cfg = config or get_video_config()
    source = Path(video_path)
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_dir = dest_dir.resolve()

    metadata = read_video_metadata(source, config=cfg)
    interval = sample_interval_frames(metadata.fps, cfg.target_sample_fps)
    metadata.sample_interval_frames = interval
    metadata.sampling_rate_fps = float(cfg.target_sample_fps)

    try:
        import cv2
    except ImportError as exc:
        raise VideoUnreadableError("Invalid or unreadable video") from exc

    capture = None
    written: list[Path] = []
    frames: list[ExtractedFrame] = []
    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoUnreadableError("Invalid or unreadable video")

        index = 0
        decoded = 0
        last_shape: tuple[int, ...] | None = None
        while True:
            ok, image = capture.read()
            if not ok or image is None or getattr(image, "size", 0) == 0:
                break
            decoded += 1
            last_shape = tuple(image.shape)
            if index % interval != 0:
                index += 1
                continue
            if len(frames) >= cfg.max_extracted_frames:
                logger.warning(
                    "Stopping extraction at max_extracted_frames=%s path=%s",
                    cfg.max_extracted_frames,
                    source,
                )
                break

            frame_path = _safe_frame_path(
                dest_dir, index, cfg.frame_image_extension
            )
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
            saved = cv2.imwrite(str(frame_path), image, params)
            if not saved or not frame_path.is_file() or frame_path.stat().st_size <= 0:
                raise VideoUnreadableError("Failed to write extracted frame")
            written.append(frame_path)
            frames.append(
                ExtractedFrame(
                    frame_number=index,
                    timestamp_seconds=timestamp_for_frame(index, metadata.fps),
                    frame_path=str(frame_path),
                )
            )
            index += 1

        if decoded <= 0 or not frames:
            raise VideoUnreadableError("Invalid or unreadable video")

        if metadata.frame_count <= 0:
            metadata.frame_count = decoded
        if metadata.fps > 0 and metadata.duration_seconds <= 0:
            metadata.duration_seconds = decoded / metadata.fps
        if last_shape is not None and (metadata.width <= 0 or metadata.height <= 0):
            metadata.height = int(last_shape[0])
            metadata.width = int(last_shape[1])

        metadata.extracted_frame_count = len(frames)
        result = VideoExtractionResult(
            metadata=metadata,
            frames=frames,
            output_dir=str(dest_dir),
            metadata_path=str(dest_dir / METADATA_FILENAME),
        )
        _write_metadata_json(dest_dir / METADATA_FILENAME, result.to_dict())
        logger.info(
            "Extracted %s frames from %s interval=%s fps=%s",
            len(frames),
            source,
            interval,
            metadata.fps,
        )
        return result
    except (VideoUnreadableError, VideoPathError):
        for leftover in written:
            leftover.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for leftover in written:
            leftover.unlink(missing_ok=True)
        logger.warning("Frame extraction failed path=%s", source, exc_info=True)
        raise VideoUnreadableError("Invalid or unreadable video") from exc
    finally:
        if capture is not None:
            capture.release()
