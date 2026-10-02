"""Load the frozen 16-frame video model once and score one video.

Normal inference runs under ``model.eval()`` and ``torch.no_grad()``.
Grad-CAM is optional because it needs a backward pass.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .constants import (
    DECISION_THRESHOLD,
    FAKE_LABEL,
    FEATURE_DIM,
    IMAGE_SIZE,
    LSTM_MODEL_NAME,
    MODEL_VERSION,
    NUM_FRAMES,
    P_FAKE_MEANING,
    REAL_LABEL,
    SPATIAL_WORDING,
    TEMPORAL_WORDING,
    VISUAL_MODEL_NAME,
)
from .exceptions import (
    CheckpointNotFoundError,
    FacePreprocessingError,
    InsufficientFramesError,
    LstmModelError,
    VideoModelError,
    VideoOpenError,
    VisualModelError,
    XaiError,
)
from .face_detector import HaarFaceDetector
from .face_preprocess import process_sampled_video
from .gradcam import gradcam_from_logit
from .lstm_model import VideoLstmV2
from .sampling import sample_video_frames
from .visual_model import load_visual_checkpoint
from .visualization import save_contact_sheet, save_frame_gradcam, save_temporal_plot

logger = logging.getLogger("maya.ai.ffpp_video")

_LOCK = threading.Lock()
_RUNTIMES: dict[tuple[str, str], "VideoRuntime"] = {}


class VideoRuntime:
    """One CPU copy of the visual encoder, the LSTM, and the face detector."""

    def __init__(self, visual_path: Path, lstm_path: Path) -> None:
        visual, visual_payload = load_visual_checkpoint(visual_path)
        if visual_payload.get("model_name") != VISUAL_MODEL_NAME:
            raise VisualModelError(
                f"{visual_path.name} is not the FF++ EfficientNet-B0 visual encoder."
            )
        lstm = _load_lstm(lstm_path)
        visual.to("cpu")
        lstm.to("cpu")
        visual.eval()
        lstm.eval()
        for model in (visual, lstm):
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        self.visual = visual
        self.lstm = lstm
        self.detector = HaarFaceDetector()
        self.visual_checkpoint = str(visual_path)
        self.lstm_checkpoint = str(lstm_path)
        self.model_name = LSTM_MODEL_NAME
        self.model_version = MODEL_VERSION
        self.visual_model_name = VISUAL_MODEL_NAME


def reset_video_runtime() -> None:
    """Drop cached networks. Tests use this to prove the next call reloads."""

    with _LOCK:
        _RUNTIMES.clear()


def get_video_runtime(
    visual_checkpoint: str | Path | None = None,
    lstm_checkpoint: str | Path | None = None,
) -> VideoRuntime:
    """Return the process-wide runtime for this checkpoint pair."""

    visual_path = _resolve_checkpoint(
        visual_checkpoint, "VIDEO_VISUAL_CHECKPOINT", "Visual encoder"
    )
    lstm_path = _resolve_checkpoint(
        lstm_checkpoint, "VIDEO_LSTM_CHECKPOINT", "LSTM"
    )
    key = (str(visual_path), str(lstm_path))
    with _LOCK:
        runtime = _RUNTIMES.get(key)
        if runtime is None:
            logger.info("Loading video checkpoints visual=%s lstm=%s", visual_path.name, lstm_path.name)
            runtime = VideoRuntime(visual_path, lstm_path)
            _RUNTIMES[key] = runtime
        return runtime


def analyze_video(
    video_path: str | Path,
    generate_xai: bool = False,
    *,
    artifact_dir: str | Path | None = None,
    visual_checkpoint: str | Path | None = None,
    lstm_checkpoint: str | Path | None = None,
) -> dict:
    """Score one video as REAL or FAKE from exactly 16 face-aware frames.

    A missing face uses the full frame. No-face frames are never discarded,
    and a missing face is not treated as FAKE by itself.
    """

    path = Path(video_path)
    sampled = sample_video_frames(path, num_frames=NUM_FRAMES)
    runtime = get_video_runtime(visual_checkpoint, lstm_checkpoint)
    processed = process_sampled_video(sampled, runtime.detector)
    if len(processed.frames) != NUM_FRAMES:
        raise FacePreprocessingError(
            f"Expected {NUM_FRAMES} frames, got {len(processed.frames)}."
        )
    order = [frame.metadata.frame_index for frame in processed.frames]
    if order != list(sampled.frame_indices):
        raise FacePreprocessingError("Processed frames are not in sampled order.")
    if order != sorted(order) or len(set(order)) != NUM_FRAMES:
        raise FacePreprocessingError("Frame order was not preserved.")

    features = _encode_frames(runtime.visual, processed.arrays)
    logit, p_fake, prediction = _predict(runtime.lstm, features)
    face_crop_frames = int(processed.face_crop_count)
    fallback_frames = int(processed.fallback_count)
    if face_crop_frames + fallback_frames != NUM_FRAMES:
        raise FacePreprocessingError("Every frame must be a face crop or a full-frame fallback.")

    frame_rows = [_frame_row(frame.metadata) for frame in processed.frames]
    result = {
        "prediction": prediction,
        "p_fake": p_fake,
        "raw_fake_logit": logit,
        "threshold": DECISION_THRESHOLD,
        "frames_analyzed": NUM_FRAMES,
        "face_crop_frames": face_crop_frames,
        "full_frame_fallback_frames": fallback_frames,
        "fallback_frames": fallback_frames,
        "model_name": runtime.model_name,
        "model_version": runtime.model_version,
        "visual_model_name": runtime.visual_model_name,
        "xai_available": False,
        "p_fake_meaning": P_FAKE_MEANING,
        "frames": frame_rows,
    }

    if not generate_xai:
        return result

    try:
        explanation = _explain(processed.arrays, frame_rows, runtime.visual, runtime.lstm)
        if explanation["prediction_name"] != prediction:
            logger.warning(
                "XAI label %s differed from inference label %s; keeping the inference label",
                explanation["prediction_name"],
                prediction,
            )
        xai_frames = explanation["frames"]
        if len(xai_frames) != NUM_FRAMES:
            raise XaiError(f"Temporal importance covered {len(xai_frames)} frames.")
        if [item["frame_index"] for item in xai_frames] != order:
            raise XaiError("Temporal importance reordered the frames.")
        names: list[str] = []
        if artifact_dir is not None:
            names = _write_xai_artifacts(Path(artifact_dir), processed.arrays, explanation)
        result["xai_available"] = True
        result["xai"] = {
            "spatial_wording": SPATIAL_WORDING,
            "temporal_wording": TEMPORAL_WORDING,
            "frames": xai_frames,
        }
        result["xai_artifact_names"] = names
    except Exception as exc:  # noqa: BLE001 - explanation must not erase a completed verdict
        logger.exception("Video XAI failed for %s", path.name)
        result["xai_available"] = False
        result["xai_error"] = type(exc).__name__
    return result


def _resolve_checkpoint(
    value: str | Path | None,
    env_name: str,
    label: str,
) -> Path:
    raw = "" if value is None else str(value).strip()
    if not raw:
        raw = os.environ.get(env_name, "").strip()
    path = Path(raw) if raw else None
    if path is None or not path.is_file():
        raise CheckpointNotFoundError(
            f"{label} checkpoint is not available. Set {env_name} to the checkpoint file."
        )
    return path.resolve()


def _load_lstm(path: Path) -> VideoLstmV2:
    if not path.is_file():
        raise CheckpointNotFoundError(f"LSTM checkpoint is not available: {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("model_name") != LSTM_MODEL_NAME:
        raise LstmModelError(f"{path.name} is not the finalized 16-frame LSTM checkpoint.")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise LstmModelError(f"{path.name} is missing LSTM weights or model config.")
    if int(config.get("sequence_length", 0)) != NUM_FRAMES:
        raise LstmModelError("The LSTM checkpoint is not the 16-frame model.")
    model = VideoLstmV2(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        sequence_length=int(config["sequence_length"]),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _encode_frames(visual: nn.Module, arrays: np.ndarray) -> torch.Tensor:
    expected = (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
    if arrays.shape != expected or arrays.dtype != np.float32:
        raise VisualModelError(f"Expected {expected} float32 frames, got {arrays.shape} {arrays.dtype}.")
    batch = torch.from_numpy(np.ascontiguousarray(arrays))
    classifier_calls = 0

    def _count(_module, _inputs, _output) -> None:
        nonlocal classifier_calls
        classifier_calls += 1

    hook = visual.classifier.register_forward_hook(_count)
    try:
        visual.eval()
        with torch.no_grad():
            pooled = visual.avgpool(visual.features(batch))
            features = torch.flatten(pooled, 1).detach().cpu().contiguous()
    finally:
        hook.remove()
    if classifier_calls != 0:
        raise VisualModelError("The visual classifier ran during video feature extraction.")
    if tuple(features.shape) != (NUM_FRAMES, FEATURE_DIM) or features.dtype != torch.float32:
        raise VisualModelError(f"Expected {(NUM_FRAMES, FEATURE_DIM)} features, got {tuple(features.shape)}.")
    if not torch.isfinite(features).all():
        raise VisualModelError("Visual features are not finite.")
    return features


def _predict(lstm: VideoLstmV2, features: torch.Tensor) -> tuple[float, float, str]:
    lstm.eval()
    with torch.no_grad():
        logit_tensor = lstm(features.unsqueeze(0)).reshape(())
    if not torch.isfinite(logit_tensor):
        raise LstmModelError("The LSTM logit is not finite.")
    probability = float(torch.sigmoid(logit_tensor).detach().cpu())
    if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise LstmModelError("P(FAKE) is outside [0, 1].")
    label = FAKE_LABEL if probability >= DECISION_THRESHOLD else REAL_LABEL
    return float(logit_tensor.detach().cpu()), probability, label


def _frame_row(metadata) -> dict:
    if bool(metadata.used_face_crop) == bool(metadata.used_full_frame_fallback):
        raise FacePreprocessingError(
            f"Frame {metadata.frame_index} did not choose exactly one input path."
        )
    return {
        "frame_index": int(metadata.frame_index),
        "timestamp_seconds": float(metadata.timestamp_seconds),
        "face_detected": bool(metadata.face_detected),
        "used_face_crop": bool(metadata.used_face_crop),
        "used_full_frame_fallback": bool(metadata.used_full_frame_fallback),
    }


def _explain(arrays: np.ndarray, frame_rows: list[dict], visual: nn.Module, lstm: VideoLstmV2) -> dict:
    explanation = gradcam_from_logit(
        torch.from_numpy(np.ascontiguousarray(arrays)),
        visual,
        lstm,
    )
    frames = []
    for item, importance, magnitude in zip(
        frame_rows,
        explanation["temporal_importance"],
        explanation["temporal_gradient_magnitude"],
    ):
        frames.append(
            {
                "frame_index": int(item["frame_index"]),
                "timestamp_seconds": float(item["timestamp_seconds"]),
                "timestamp": float(item["timestamp_seconds"]),
                "face_detected": bool(item["face_detected"]),
                "used_face_crop": bool(item["used_face_crop"]),
                "used_full_frame_fallback": bool(item["used_full_frame_fallback"]),
                "temporal_importance": float(importance),
                "temporal_gradient_magnitude": float(magnitude),
            }
        )
    explanation["frames"] = frames
    return explanation


def _write_xai_artifacts(artifact_dir: Path, arrays: np.ndarray, explanation: dict) -> list[str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frames = explanation["frames"]
    save_temporal_plot(artifact_dir / "temporal_contribution.png", frames)
    save_contact_sheet(
        artifact_dir / "gradcam_contact_sheet.png",
        arrays,
        explanation["heatmaps"],
        frames,
    )
    heatmaps = explanation["heatmaps"].detach().cpu().numpy()
    frame_names: list[str] = []
    for index, frame in enumerate(frames):
        name = f"gradcam_frame_{int(frame['frame_index'])}.png"
        save_frame_gradcam(artifact_dir / name, arrays[index], heatmaps[index])
        frame_names.append(name)
    record = {
        "prediction": explanation["prediction_name"],
        "p_fake": explanation["p_fake"],
        "p_fake_meaning": P_FAKE_MEANING,
        "threshold": DECISION_THRESHOLD,
        "spatial_wording": SPATIAL_WORDING,
        "temporal_wording": TEMPORAL_WORDING,
        "spatial_layer": explanation["spatial_layer"],
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "face_detected": frame["face_detected"],
                "used_face_crop": frame["used_face_crop"],
                "used_full_frame_fallback": frame["used_full_frame_fallback"],
                "temporal_importance": frame["temporal_importance"],
            }
            for frame in frames
        ],
        "artifacts": [
            "explanation.json",
            "temporal_contribution.png",
            "gradcam_contact_sheet.png",
            *frame_names,
        ],
    }
    (artifact_dir / "explanation.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    return list(record["artifacts"])


__all__ = [
    "CheckpointNotFoundError",
    "FacePreprocessingError",
    "InsufficientFramesError",
    "VideoModelError",
    "VideoOpenError",
    "VideoRuntime",
    "analyze_video",
    "get_video_runtime",
    "reset_video_runtime",
]
