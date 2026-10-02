"""Video-level aggregation of existing frame-level authenticity scores.

This is a transparent mean over still-frame probabilities. It is not a
temporal detector, video neural network, or motion model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from ai.inference.confidence import ConfidenceDecision, decide_from_probabilities
from ai.inference.inference_config import InferenceConfig, get_inference_config
from ai.video.exceptions import VideoAggregationError
from ai.video.frame_extractor import ExtractedFrame

AGGREGATION_METHOD = "mean_frame_probability"


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


def _finite_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _prediction_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().upper()
    if label in {"REAL", "FAKE"}:
        return label
    return None


@dataclass
class VideoAggregationResult:
    """Mean-frame video assessment plus preserved frame-level evidence."""

    prediction: str
    confidence: float
    real_probability: float
    fake_probability: float
    threshold: float
    confidence_level: str
    threshold_decision: str
    method: str = AGGREGATION_METHOD
    total_frames: int = 0
    total_frames_analyzed: int = 0
    real_frame_count: int = 0
    fake_frame_count: int = 0
    fake_frame_percentage: float = 0.0
    frames: list[dict[str, Any]] = field(default_factory=list)
    suspicious_frames: list[dict[str, Any]] = field(default_factory=list)
    skipped_invalid_probability_count: int = 0

    def aggregation_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "total_frames": self.total_frames,
            "total_frames_analyzed": self.total_frames_analyzed,
            "real_frame_count": self.real_frame_count,
            "fake_frame_count": self.fake_frame_count,
            "fake_frame_percentage": self.fake_frame_percentage,
            "skipped_invalid_probability_count": self.skipped_invalid_probability_count,
            "threshold": self.threshold,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "real_probability": self.real_probability,
            "fake_probability": self.fake_probability,
            "confidence_level": self.confidence_level,
            "threshold_decision": self.threshold_decision,
            "aggregation": self.aggregation_dict(),
            "frames": self.frames,
            "suspicious_frames": self.suspicious_frames,
        }


def aggregate_frame_predictions(
    frames: Iterable[ExtractedFrame | dict[str, Any]],
    *,
    config: InferenceConfig | None = None,
) -> VideoAggregationResult:
    """Mean REAL/FAKE probabilities, then the existing image decision rule."""

    cfg = config or get_inference_config()
    records = [_as_frame(item) for item in frames]
    if not records:
        raise VideoAggregationError("No extracted frames to aggregate")

    real_values: list[float] = []
    fake_values: list[float] = []
    analyzed: list[ExtractedFrame] = []
    skipped = 0
    for frame in records:
        real_p = _finite_probability(frame.real_probability)
        fake_p = _finite_probability(frame.fake_probability)
        if real_p is None or fake_p is None:
            skipped += 1
            continue
        real_values.append(real_p)
        fake_values.append(fake_p)
        analyzed.append(frame)

    if not analyzed:
        raise VideoAggregationError("No valid frame probabilities to aggregate")

    mean_real = sum(real_values) / len(real_values)
    mean_fake = sum(fake_values) / len(fake_values)
    decision: ConfidenceDecision = decide_from_probabilities(
        [mean_real, mean_fake], cfg
    )

    real_count = 0
    fake_count = 0
    suspicious: list[dict[str, Any]] = []
    for frame in analyzed:
        label = _prediction_label(frame.prediction)
        payload = frame.to_dict()
        if label == "FAKE":
            fake_count += 1
            suspicious.append(payload)
        elif label == "REAL":
            real_count += 1

    total_analyzed = len(analyzed)
    fake_pct = (fake_count / total_analyzed) * 100.0 if total_analyzed else 0.0
    return VideoAggregationResult(
        prediction=decision.predicted_label,
        confidence=float(decision.confidence),
        real_probability=float(decision.real_probability),
        fake_probability=float(decision.fake_probability),
        threshold=float(decision.threshold),
        confidence_level=decision.confidence_level,
        threshold_decision=decision.threshold_decision,
        total_frames=len(records),
        total_frames_analyzed=total_analyzed,
        real_frame_count=real_count,
        fake_frame_count=fake_count,
        fake_frame_percentage=round(fake_pct, 4),
        frames=[frame.to_dict() for frame in records],
        suspicious_frames=suspicious,
        skipped_invalid_probability_count=skipped,
    )
