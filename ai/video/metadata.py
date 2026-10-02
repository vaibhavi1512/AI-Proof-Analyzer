"""Read video duration, FPS, frame count, and resolution via OpenCV."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai.video.exceptions import VideoUnreadableError
from ai.video.video_config import VideoConfig, get_video_config

logger = logging.getLogger("maya.ai.video.metadata")


@dataclass
class VideoMetadata:
    """Container properties plus extraction bookkeeping for Phase 3."""

    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    extracted_frame_count: int = 0
    sampling_rate_fps: float = 1.0
    sample_interval_frames: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_interval_frames(fps: float, target_sample_fps: float) -> int:
    """Return the decode stride that approximates ``target_sample_fps``.

    ``interval = round(video_fps / target_fps)`` so a 30 FPS clip yields every
    30th frame (~1/s) while a 5 FPS clip yields every 5th frame. Interval is
    always at least 1 so short or low-FPS videos still produce frames.
    """

    if target_sample_fps <= 0:
        target_sample_fps = 1.0
    if fps <= 0:
        return 1
    return max(1, int(round(fps / target_sample_fps)))


def duration_from_count(frame_count: int, fps: float) -> float:
    """Duration in seconds from container frame count and FPS."""

    if frame_count <= 0 or fps <= 0:
        return 0.0
    return float(frame_count) / float(fps)


def timestamp_for_frame(frame_number: int, fps: float) -> float:
    """Presentation timestamp for a 0-based frame index."""

    if fps <= 0:
        return float(frame_number)
    return float(frame_number) / float(fps)


def read_video_metadata(
    video_path: Path | str,
    *,
    config: VideoConfig | None = None,
) -> VideoMetadata:
    """Open ``video_path`` with OpenCV and return container metadata.

    Does not extract or write frames. Raises :class:`VideoUnreadableError` when
    the file cannot be opened or reports an unusable size.
    """

    cfg = config or get_video_config()
    path = Path(video_path)
    if not path.is_file():
        raise VideoUnreadableError("Invalid or unreadable video")

    try:
        import cv2
    except ImportError as exc:
        raise VideoUnreadableError("Invalid or unreadable video") from exc

    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoUnreadableError("Invalid or unreadable video")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                raise VideoUnreadableError("Invalid or unreadable video")
            height, width = int(frame.shape[0]), int(frame.shape[1])

        interval = sample_interval_frames(fps, cfg.target_sample_fps)
        duration = duration_from_count(frame_count, fps)
        return VideoMetadata(
            duration_seconds=duration,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            extracted_frame_count=0,
            sampling_rate_fps=float(cfg.target_sample_fps),
            sample_interval_frames=interval,
        )
    except VideoUnreadableError:
        raise
    except Exception as exc:
        logger.warning("Video metadata read failed path=%s", path, exc_info=True)
        raise VideoUnreadableError("Invalid or unreadable video") from exc
    finally:
        if capture is not None:
            capture.release()
