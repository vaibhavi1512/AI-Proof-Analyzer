"""Basic temporal face consistency on ~1 FPS sampled video frames.

Reuses the existing OpenCV Haar detector. This is forensic observation of
face presence and bounding-box continuity, not a temporal deepfake model
and not a REAL/FAKE verdict.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ai.face_verification.config import FaceVerificationConfig
from ai.face_verification.engine import FaceEngine, get_face_engine
from ai.face_verification.images import FaceImageError, load_rgb_image
from ai.face_verification.types import BoundingBox
from ai.video.frame_extractor import ExtractedFrame

logger = logging.getLogger("maya.ai.video.temporal")

TEMPORAL_METHOD = "haar_iou_centroid_tracking"
IOU_MATCH_THRESHOLD = 0.3
CENTROID_DISTANCE_FACTOR = 0.75
# Temporal-only Haar gates (not used by face-reference verification).
MIN_BOX_SIDE_FRACTION = 0.05
MAX_BOX_SIDE_FRACTION = 0.45
MAX_MISSED_SAMPLED_FRAMES = 1


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def bbox_centroid(box: BoundingBox) -> tuple[float, float]:
    return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)


def bbox_diagonal(box: BoundingBox) -> float:
    return math.hypot(float(box.width), float(box.height))


def spatial_match_score(
    previous: BoundingBox,
    current: BoundingBox,
    *,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    centroid_factor: float = CENTROID_DISTANCE_FACTOR,
) -> float:
    """Higher is better. 0 means no association."""

    iou = bbox_iou(previous, current)
    if iou >= iou_threshold:
        return 1.0 + iou
    ax, ay = bbox_centroid(previous)
    bx, by = bbox_centroid(current)
    dist = math.hypot(ax - bx, ay - by)
    limit = centroid_factor * max(
        min(bbox_diagonal(previous), bbox_diagonal(current)),
        1.0,
    )
    if dist <= limit:
        return 0.5 * (1.0 - dist / limit)
    return 0.0


def bbox_to_payload(box: BoundingBox) -> dict[str, Any]:
    payload = box.to_dict()
    payload["x"] = payload["x1"]
    payload["y"] = payload["y1"]
    return payload


@dataclass
class TrackObservation:
    frame_number: int
    timestamp_seconds: float
    bbox: BoundingBox


@dataclass
class FaceTrack:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    last_sampled_index: int = -1

    @property
    def frames_tracked(self) -> int:
        return len(self.observations)

    @property
    def start_timestamp(self) -> float:
        return self.observations[0].timestamp_seconds if self.observations else 0.0

    @property
    def end_timestamp(self) -> float:
        return self.observations[-1].timestamp_seconds if self.observations else 0.0

    @property
    def last_box(self) -> BoundingBox | None:
        return self.observations[-1].bbox if self.observations else None

    def mean_area(self) -> float:
        if not self.observations:
            return 0.0
        return sum(obs.bbox.width * obs.bbox.height for obs in self.observations) / len(
            self.observations
        )


def filter_haar_boxes(
    boxes: Sequence[BoundingBox],
    frame_width: int,
    frame_height: int,
    *,
    min_side_fraction: float = MIN_BOX_SIDE_FRACTION,
    max_side_fraction: float = MAX_BOX_SIDE_FRACTION,
) -> list[BoundingBox]:
    """Drop Haar boxes that are implausibly small or large for the frame."""

    short_side = min(int(frame_width), int(frame_height))
    if short_side <= 0:
        return list(boxes)
    min_side = min_side_fraction * short_side
    max_side = max_side_fraction * short_side
    kept: list[BoundingBox] = []
    for box in boxes:
        side_min = min(box.width, box.height)
        side_max = max(box.width, box.height)
        if side_min < min_side or side_max > max_side:
            continue
        kept.append(box)
    return kept


def associate_detections(
    frames: Sequence[tuple[int, float, list[BoundingBox]]],
    *,
    iou_threshold: float = IOU_MATCH_THRESHOLD,
    centroid_factor: float = CENTROID_DISTANCE_FACTOR,
    max_missed_sampled_frames: int = MAX_MISSED_SAMPLED_FRAMES,
) -> list[FaceTrack]:
    """Greedy IoU/centroid association; expire after one missed sampled frame."""

    tracks: list[FaceTrack] = []
    next_id = 1
    for sampled_index, (frame_number, timestamp, boxes) in enumerate(frames):
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for t_index, track in enumerate(tracks):
            last = track.last_box
            if last is None:
                continue
            missed = sampled_index - track.last_sampled_index - 1
            if missed > max_missed_sampled_frames:
                continue
            for b_index, box in enumerate(boxes):
                score = spatial_match_score(
                    last,
                    box,
                    iou_threshold=iou_threshold,
                    centroid_factor=centroid_factor,
                )
                if score > 0:
                    pairs.append((score, t_index, b_index))
        pairs.sort(key=lambda item: item[0], reverse=True)
        for _score, t_index, b_index in pairs:
            if t_index in used_tracks or b_index in used_boxes:
                continue
            used_tracks.add(t_index)
            used_boxes.add(b_index)
            tracks[t_index].observations.append(
                TrackObservation(frame_number, timestamp, boxes[b_index])
            )
            tracks[t_index].last_sampled_index = sampled_index
        for b_index, box in enumerate(boxes):
            if b_index in used_boxes:
                continue
            tracks.append(
                FaceTrack(
                    track_id=next_id,
                    observations=[TrackObservation(frame_number, timestamp, box)],
                    last_sampled_index=sampled_index,
                )
            )
            next_id += 1
    return tracks


def _primary_track(tracks: list[FaceTrack]) -> FaceTrack | None:
    if not tracks:
        return None
    return max(
        tracks,
        key=lambda track: (
            track.frames_tracked,
            track.end_timestamp - track.start_timestamp,
            track.mean_area(),
        ),
    )


def _gap_segments(
    sampled: Sequence[tuple[int, float]],
    primary: FaceTrack,
) -> list[dict[str, Any]]:
    present = {obs.frame_number for obs in primary.observations}
    start_ts = primary.start_timestamp
    end_ts = primary.end_timestamp
    missing: list[tuple[int, int, float]] = [
        (index, frame_number, timestamp)
        for index, (frame_number, timestamp) in enumerate(sampled)
        if start_ts <= timestamp <= end_ts and frame_number not in present
    ]
    gaps: list[dict[str, Any]] = []
    current: list[tuple[int, int, float]] = []
    for item in missing:
        if not current:
            current = [item]
            continue
        if item[0] == current[-1][0] + 1:
            current.append(item)
        else:
            gaps.append(
                {
                    "start_timestamp": current[0][2],
                    "end_timestamp": current[-1][2],
                    "missing_frame_count": len(current),
                }
            )
            current = [item]
    if current:
        gaps.append(
            {
                "start_timestamp": current[0][2],
                "end_timestamp": current[-1][2],
                "missing_frame_count": len(current),
            }
        )
    return gaps


def _movement_stats(primary: FaceTrack | None) -> dict[str, float | None]:
    if primary is None or len(primary.observations) < 2:
        return {
            "average_centroid_movement": None,
            "average_bbox_area_change": None,
        }
    distances: list[float] = []
    area_changes: list[float] = []
    previous = primary.observations[0]
    for current in primary.observations[1:]:
        ax, ay = bbox_centroid(previous.bbox)
        bx, by = bbox_centroid(current.bbox)
        distances.append(math.hypot(ax - bx, ay - by))
        area_a = max(1.0, float(previous.bbox.width * previous.bbox.height))
        area_b = float(current.bbox.width * current.bbox.height)
        area_changes.append(abs(area_b - area_a) / area_a)
        previous = current
    return {
        "average_centroid_movement": round(sum(distances) / len(distances), 4),
        "average_bbox_area_change": round(sum(area_changes) / len(area_changes), 4),
    }


def build_temporal_result(
    sampled: Sequence[tuple[int, float]],
    detections: Sequence[list[BoundingBox]],
    tracks: list[FaceTrack],
    *,
    detector_name: str = "opencv_haar",
) -> dict[str, Any]:
    total = len(sampled)
    frames_with_face = sum(1 for boxes in detections if boxes)
    rate = (frames_with_face / total) if total else 0.0
    primary = _primary_track(tracks)
    obs_by_frame: dict[int, list[tuple[int, BoundingBox]]] = {}
    for track in tracks:
        for obs in track.observations:
            obs_by_frame.setdefault(obs.frame_number, []).append((track.track_id, obs.bbox))

    frame_tracks: list[dict[str, Any]] = []
    for (frame_number, timestamp), boxes in zip(sampled, detections):
        assigned = obs_by_frame.get(frame_number, [])
        detections_payload = [
            {"track_id": track_id, "bbox": bbox_to_payload(box)}
            for track_id, box in assigned
        ]
        primary_id = None
        primary_bbox = None
        if primary is not None:
            for track_id, box in assigned:
                if track_id == primary.track_id:
                    primary_id = track_id
                    primary_bbox = bbox_to_payload(box)
                    break
        if primary_id is None and assigned:
            primary_id = assigned[0][0]
            primary_bbox = bbox_to_payload(assigned[0][1])
        frame_tracks.append(
            {
                "frame_number": frame_number,
                "timestamp_seconds": timestamp,
                "face_count": len(boxes),
                "track_id": primary_id,
                "bbox": primary_bbox,
                "detections": detections_payload,
            }
        )

    gaps = _gap_segments(sampled, primary) if primary is not None else []
    notes: list[str] = []
    if total:
        notes.append(
            f"Face detected in {frames_with_face}/{total} sampled frames."
        )
    if primary is not None:
        notes.append(
            "Primary face track spans approximately "
            f"{primary.start_timestamp:.1f}–{primary.end_timestamp:.1f} seconds."
        )
        for gap in gaps:
            notes.append(
                "Face tracking was interrupted between "
                f"{gap['start_timestamp']:.1f} and {gap['end_timestamp']:.1f} seconds."
            )

    movement = _movement_stats(primary)
    primary_payload = None
    if primary is not None:
        primary_payload = {
            "track_id": primary.track_id,
            "start_timestamp": primary.start_timestamp,
            "end_timestamp": primary.end_timestamp,
            "frames_tracked": primary.frames_tracked,
            "coverage": round(primary.frames_tracked / total, 4) if total else 0.0,
        }
    return {
        "method": TEMPORAL_METHOD,
        "detector": detector_name,
        "sampling_note": (
            "Association is on ~1 FPS sampled frames, not every video frame."
        ),
        "total_frames": total,
        "frames_with_face": frames_with_face,
        "face_detection_rate": round(rate, 4),
        "track_count": len(tracks),
        "primary_track": primary_payload,
        "tracking_gaps": gaps,
        "average_centroid_movement": movement["average_centroid_movement"],
        "average_bbox_area_change": movement["average_bbox_area_change"],
        "notes": notes,
        "frame_tracks": frame_tracks,
    }


def _as_frames(frames: Sequence[ExtractedFrame | dict[str, Any]]) -> list[ExtractedFrame]:
    records: list[ExtractedFrame] = []
    for item in frames:
        if isinstance(item, ExtractedFrame):
            records.append(item)
            continue
        records.append(
            ExtractedFrame(
                frame_number=int(item.get("frame_number", 0)),
                timestamp_seconds=float(item.get("timestamp_seconds") or 0.0),
                frame_path=str(item.get("frame_path") or ""),
            )
        )
    return records


def _resolve_path(frame: ExtractedFrame, path_root: Path | None) -> Path:
    stored = Path(frame.frame_path)
    if stored.is_absolute() or path_root is None:
        return stored
    return path_root / stored


class _EmptyTemporalEngine:
    """Valid no-op detector when Haar cannot initialize (e.g. OpenCV 5)."""

    name = "opencv_haar"
    haar_unavailable = True

    def detect(self, image: Any) -> list[Any]:
        return []


def _haar_temporal_engine() -> FaceEngine:
    """Load OpenCV Haar for temporal tracking.

    OpenCV 5.x Python wheels removed ``cv2.CascadeClassifier`` and the bundled
    Haar XML files. That is an environment incompatibility, not a missing face
    in the video. Prefer Haar when the API exists; otherwise return an empty
    detector so callers still get a schema-valid temporal result.
    """

    try:
        import cv2
    except ImportError as exc:
        logger.warning("OpenCV is unavailable for temporal Haar detection: %s", exc)
        return _EmptyTemporalEngine()

    if not hasattr(cv2, "CascadeClassifier"):
        logger.warning(
            "cv2.CascadeClassifier is missing (OpenCV %s); temporal analysis "
            "will record zero detections instead of failing",
            getattr(cv2, "__version__", "unknown"),
        )
        return _EmptyTemporalEngine()

    try:
        return get_face_engine(FaceVerificationConfig(engine="opencv"))
    except (AttributeError, RuntimeError, OSError) as exc:
        logger.warning("OpenCV Haar engine failed to initialize: %s", exc)
        return _EmptyTemporalEngine()


def analyze_temporal_consistency(
    frames: Sequence[ExtractedFrame | dict[str, Any]],
    *,
    path_root: Path | str | None = None,
    engine: FaceEngine | None = None,
    detect_fn: Callable[[Path], list[BoundingBox]] | None = None,
) -> dict[str, Any]:
    """Detect faces on sampled JPEGs and associate tracks. Never scores REAL/FAKE."""

    records = _as_frames(frames)
    sampled = [(frame.frame_number, float(frame.timestamp_seconds)) for frame in records]
    root = Path(path_root).resolve() if path_root is not None else None
    detector = engine
    detector_name = getattr(engine, "name", None) or "opencv_haar"

    detections: list[list[BoundingBox]] = []
    for frame in records:
        image_path = _resolve_path(frame, root)
        boxes: list[BoundingBox] = []
        try:
            if detect_fn is not None:
                boxes = list(detect_fn(image_path))
            else:
                if detector is None:
                    detector = _haar_temporal_engine()
                    detector_name = detector.name
                image = load_rgb_image(image_path)
                height, width = image.shape[:2]
                boxes = filter_haar_boxes(
                    [face.bbox for face in detector.detect(image)],
                    width,
                    height,
                )
        except (FaceImageError, OSError, ValueError, AttributeError) as exc:
            logger.info("Temporal face detection skipped frame=%s: %s", image_path, exc)
            boxes = []
        detections.append(boxes)

    sequence = [
        (frame_number, timestamp, boxes)
        for (frame_number, timestamp), boxes in zip(sampled, detections)
    ]
    tracks = associate_detections(sequence)
    result = build_temporal_result(
        sampled, detections, tracks, detector_name=detector_name
    )
    if getattr(detector, "haar_unavailable", False):
        result["notes"].insert(
            0,
            "OpenCV Haar cascade is unavailable in this cv2 build; no faces were scored.",
        )
    return result
