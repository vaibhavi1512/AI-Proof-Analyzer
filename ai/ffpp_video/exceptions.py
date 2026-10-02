"""Errors for production 16-frame video inference. Checkpoints are not rewritten."""


class VideoModelError(Exception):
    """The video model cannot produce a result from this input."""


class VideoOpenError(VideoModelError):
    """The video path cannot be opened as a readable video."""


class InsufficientFramesError(VideoModelError):
    """The video does not contain enough readable frames to sample 16 unique frames."""


class FacePreprocessingError(VideoModelError):
    """A sampled frame could not be turned into a model input."""


class VisualModelError(VideoModelError):
    """The frozen EfficientNet-B0 visual encoder cannot be loaded or run."""


class LstmModelError(VideoModelError):
    """The frozen 2-layer LSTM cannot be loaded or run."""


class XaiError(VideoModelError):
    """An optional explanation cannot be produced. The verdict can still stand."""


class CheckpointNotFoundError(VideoModelError):
    """A configured visual or LSTM checkpoint file is missing."""
