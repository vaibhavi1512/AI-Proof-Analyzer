"""Production settings for the frozen 16-frame FF++ video model.

These match the finalized visual encoder and 2-layer LSTM. They are not
training hyperparameters and they are not a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

NUM_FRAMES = 16
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FEATURE_DIM = 1280
CLASSIFIER_DROPOUT = 0.2

LSTM_INPUT_SIZE = 1280
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_SEQUENCE_LENGTH = NUM_FRAMES

DECISION_THRESHOLD = 0.5
REAL_LABEL = "REAL"
FAKE_LABEL = "FAKE"

VISUAL_MODEL_NAME = "ffpp_visual_efficientnet_b0"
LSTM_MODEL_NAME = "ffpp_video_lstm_v2"
MODEL_VERSION = "v2-16frame"

P_FAKE_MEANING = "model-predicted probability for the FAKE class"
SPATIAL_WORDING = "regions contributing to the model prediction"
TEMPORAL_WORDING = "frames with higher relative contribution"

CONTACT_SHEET_FRAMES = 8


@dataclass(frozen=True)
class FacePreprocessConfig:
    """Haar detector and crop settings used by the finalized FF++ pipeline."""

    image_size: int = IMAGE_SIZE
    num_frames: int = NUM_FRAMES
    margin: float = 0.25
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_size: int = 40
    min_confidence: float = 0.0
    cascade_name: str = "haarcascade_frontalface_default.xml"
