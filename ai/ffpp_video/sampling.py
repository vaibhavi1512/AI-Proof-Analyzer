"""Sample a fixed number of frames spread across a video."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import NUM_FRAMES
from .exceptions import InsufficientFramesError, VideoOpenError


@dataclass(frozen=True)
class SampledFrame:
    """One frame taken from its original position in the video."""

    frame_index: int
    timestamp_seconds: float
    frame: np.ndarray


@dataclass(frozen=True)
class SampledVideo:
    """Frames sampled from one video, with the timing used to choose them."""

    video_path: Path
    fps: float
    total_frames: int
    duration_seconds: float
    frames: tuple[SampledFrame, ...]

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.frame_index for frame in self.frames)

    @property
    def timestamps_seconds(self) -> tuple[float, ...]:
        return tuple(frame.timestamp_seconds for frame in self.frames)


def uniform_frame_indices(frame_count: int, num_frames: int) -> list[int]:
    """Return ``num_frames`` unique indices spread from the first frame to the last.

    The mapping is integer rounding of ``i * (frame_count - 1) / (num_frames - 1)``.
    The same inputs always produce the same indices. The last index is the last
    frame, so the sample can include the end of the video.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise TypeError("frame_count must be an int")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError("num_frames must be an int")
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if frame_count < num_frames:
        raise InsufficientFramesError(
            f"Cannot choose {num_frames} unique frames from {frame_count} frames. "
            "Frames are not duplicated."
        )
    if num_frames == 1:
        return [0]

    last_index = frame_count - 1
    denominator = num_frames - 1
    indices = [
        (position * last_index + denominator // 2) // denominator
        for position in range(num_frames)
    ]
    if len(set(indices)) != num_frames or indices[0] != 0 or indices[-1] != last_index:
        raise RuntimeError(
            f"Uniform index calculation failed for {frame_count} frames and {num_frames} samples."
        )
    return indices


def sample_video_frames(video_path: str | Path, num_frames: int = NUM_FRAMES) -> SampledVideo:
    """Read exactly ``num_frames`` frames spread across ``video_path``.

    Frames are read in order. If fewer than ``num_frames`` frames can be read,
    this raises ``InsufficientFramesError`` instead of repeating frames.
    """
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError("num_frames must be an int")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")

    path = Path(video_path)
    if not path.is_file():
        raise VideoOpenError(f"Could not open video: {path} does not exist.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoOpenError(
                f"Could not open video: {path}. The file is unreadable or is not a supported video."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise VideoOpenError(f"Could not read a valid FPS from video: {path} (fps={fps}).")
        if not math.isfinite(reported_frames):
            raise VideoOpenError(f"Could not read a frame count from video: {path}.")

        total_frames = int(reported_frames)
        if total_frames < num_frames:
            raise InsufficientFramesError(
                f"Video {path.name} does not contain enough readable frames: "
                f"reported frame count is {total_frames}, but {num_frames} are required. "
                "Frames are not duplicated."
            )

        indices = uniform_frame_indices(total_frames, num_frames)
        read_frames = _read_frame_indices(capture, indices)
        if len(read_frames) != num_frames:
            raise InsufficientFramesError(
                f"Video {path.name} does not contain enough readable frames. "
                f"It reports {total_frames} frames and {num_frames} were requested, "
                f"but only {len(read_frames)} selected frame(s) could be read. "
                "Frames are not duplicated."
            )

        sampled = tuple(
            SampledFrame(
                frame_index=index,
                timestamp_seconds=index / fps,
                frame=frame,
            )
            for index, frame in read_frames
        )
        return SampledVideo(
            video_path=path,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=total_frames / fps,
            frames=sampled,
        )
    finally:
        capture.release()


def _read_frame_indices(
    capture: cv2.VideoCapture,
    indices: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Walk the video in order and decode only the requested frame indices."""
    collected: list[tuple[int, np.ndarray]] = []
    target = 0
    frame_index = 0
    while target < len(indices):
        if frame_index == indices[target]:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            collected.append((frame_index, frame.copy()))
            target += 1
        elif not capture.grab():
            break
        frame_index += 1
    return collected
