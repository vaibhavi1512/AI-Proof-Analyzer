"""Errors raised by Phase 2 video metadata / frame extraction."""


class VideoProcessingError(Exception):
    """Base error for unreadable videos or unsafe frame paths."""


class VideoUnreadableError(VideoProcessingError):
    """The file cannot be opened or decoded as a video."""


class VideoPathError(VideoProcessingError):
    """An output path would escape the allowed frame directory."""


class VideoAggregationError(VideoProcessingError):
    """Frame-level results cannot be aggregated into a video-level assessment."""
