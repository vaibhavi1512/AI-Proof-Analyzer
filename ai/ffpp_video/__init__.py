"""Frozen FF++ 16-frame video inference for MAYA.

Visual encoder: EfficientNet-B0 features (1280-d), classifier unused.
Temporal model: 2-layer LSTM over 16 frames. Threshold stays 0.50.
"""

from ai.ffpp_video.analyze import analyze_video, reset_video_runtime

__all__ = ["analyze_video", "reset_video_runtime"]
