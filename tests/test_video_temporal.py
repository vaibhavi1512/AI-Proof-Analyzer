"""Phase 5 temporal face consistency on sampled video frames."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.face_verification.types import BoundingBox
from ai.inference.inference_config import InferenceConfig
from ai.video.aggregator import aggregate_frame_predictions
from ai.video.frame_extractor import ExtractedFrame
from ai.video.temporal import (
    MAX_BOX_SIDE_FRACTION,
    MAX_MISSED_SAMPLED_FRAMES,
    MIN_BOX_SIDE_FRACTION,
    TEMPORAL_METHOD,
    analyze_temporal_consistency,
    associate_detections,
    bbox_iou,
    build_temporal_result,
    filter_haar_boxes,
    spatial_match_score,
)
from tests.conftest_product import register_and_login
from tests.test_video_frame_analysis import _fake_pipeline, _write_video


def _box(x: int, y: int, w: int = 40, h: int = 40, score: float = 1.0) -> BoundingBox:
    return BoundingBox(x, y, x + w, y + h, score=score)


def test_bbox_iou_match_and_mismatch() -> None:
    a = _box(0, 0, 40, 40)
    overlapping = _box(10, 10, 40, 40)
    far = _box(200, 200, 40, 40)
    assert bbox_iou(a, a) == pytest.approx(1.0)
    assert bbox_iou(a, overlapping) > 0.3
    assert bbox_iou(a, far) == pytest.approx(0.0)
    assert spatial_match_score(a, overlapping) > 0
    assert spatial_match_score(a, far) == pytest.approx(0.0)


def test_no_face_detected() -> None:
    sampled = [(0, 0.0), (30, 1.0), (60, 2.0)]
    detections = [[], [], []]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    result = build_temporal_result(sampled, detections, tracks)
    assert result["frames_with_face"] == 0
    assert result["face_detection_rate"] == pytest.approx(0.0)
    assert result["track_count"] == 0
    assert result["primary_track"] is None
    assert result["tracking_gaps"] == []


def test_one_continuous_face() -> None:
    sampled = [(0, 0.0), (30, 1.0), (60, 2.0)]
    detections = [[_box(10, 10)], [_box(12, 11)], [_box(14, 12)]]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    result = build_temporal_result(sampled, detections, tracks)
    assert result["track_count"] == 1
    assert result["frames_with_face"] == 3
    assert result["face_detection_rate"] == pytest.approx(1.0)
    assert result["primary_track"]["frames_tracked"] == 3
    assert result["primary_track"]["start_timestamp"] == pytest.approx(0.0)
    assert result["primary_track"]["end_timestamp"] == pytest.approx(2.0)
    assert result["tracking_gaps"] == []
    ids = [row["track_id"] for row in result["frame_tracks"]]
    assert ids == [1, 1, 1]


def test_face_disappears_temporarily() -> None:
    sampled = [(0, 0.0), (30, 1.0), (60, 2.0)]
    detections = [[_box(10, 10)], [], [_box(12, 11)]]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    result = build_temporal_result(sampled, detections, tracks)
    assert result["track_count"] == 1
    assert result["frames_with_face"] == 2
    assert result["face_detection_rate"] == pytest.approx(0.6667)
    assert result["tracking_gaps"]
    gap = result["tracking_gaps"][0]
    assert gap["start_timestamp"] == pytest.approx(1.0)
    assert gap["end_timestamp"] == pytest.approx(1.0)
    assert gap["missing_frame_count"] == 1
    assert any("interrupted" in note for note in result["notes"])


def test_giant_box_cannot_match_distant_small_box() -> None:
    small = _box(0, 0, 88, 88)
    giant = BoundingBox(469, 262, 469 + 1861, 262 + 1861, score=1.0)
    assert bbox_iou(small, giant) < 0.3
    assert spatial_match_score(small, giant) == pytest.approx(0.0)
    sampled = [(0, 0.0), (30, 1.0)]
    detections = [[small], [giant]]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    assert len(tracks) == 2


def test_track_expires_after_missed_sampled_frames() -> None:
    assert MAX_MISSED_SAMPLED_FRAMES == 1
    sampled = [
        (0, 0.0),
        (30, 1.0),
        (60, 2.0),
        (90, 3.0),
        (360, 12.0),
    ]
    box = _box(10, 10)
    detections = [[box], [], [], [], [_box(12, 11)]]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    assert len(tracks) == 2
    assert tracks[0].frames_tracked == 1
    assert tracks[1].frames_tracked == 1
    assert tracks[1].start_timestamp == pytest.approx(12.0)


def test_haar_size_gate_drops_tiny_and_huge_boxes() -> None:
    frame_w, frame_h = 3840, 2160
    tiny = _box(0, 0, 50, 50)
    plausible = _box(100, 100, 200, 200)
    huge = BoundingBox(0, 0, 1861, 1861, score=1.0)
    kept = filter_haar_boxes([tiny, plausible, huge], frame_w, frame_h)
    assert kept == [plausible]
    min_side = MIN_BOX_SIDE_FRACTION * min(frame_w, frame_h)
    max_side = MAX_BOX_SIDE_FRACTION * min(frame_w, frame_h)
    assert tiny.width < min_side
    assert huge.width > max_side
    assert min_side <= plausible.width <= max_side


def test_primary_track_after_expiry_is_longest_surviving() -> None:
    sampled = [
        (0, 0.0),
        (30, 1.0),
        (60, 2.0),
        (90, 3.0),
        (120, 4.0),
        (150, 5.0),
    ]
    detections = [
        [_box(10, 10)],
        [_box(12, 11)],
        [_box(14, 12)],
        [],
        [],
        [_box(16, 13)],
    ]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    result = build_temporal_result(sampled, detections, tracks)
    assert result["track_count"] == 2
    assert result["primary_track"]["track_id"] == 1
    assert result["primary_track"]["frames_tracked"] == 3
    assert result["primary_track"]["end_timestamp"] == pytest.approx(2.0)


def test_multiple_faces_primary_is_longest() -> None:
    sampled = [(0, 0.0), (30, 1.0), (60, 2.0)]
    detections = [
        [_box(10, 10), _box(200, 10)],
        [_box(12, 10), _box(202, 10)],
        [_box(14, 10)],
    ]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    result = build_temporal_result(sampled, detections, tracks)
    assert result["track_count"] == 2
    assert result["primary_track"]["frames_tracked"] == 3
    assert result["frame_tracks"][0]["face_count"] == 2
    assert result["frame_tracks"][2]["face_count"] == 1


def test_non_overlap_starts_new_track() -> None:
    sampled = [(0, 0.0), (30, 1.0)]
    detections = [[_box(0, 0)], [_box(300, 300)]]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    assert len(tracks) == 2


def test_analyze_temporal_haar_unavailable_is_valid_empty_result(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame_000000.jpg"
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(jpeg, format="JPEG")
    frames = [
        ExtractedFrame(frame_number=0, timestamp_seconds=0.0, frame_path=str(jpeg)),
        ExtractedFrame(frame_number=30, timestamp_seconds=1.0, frame_path=str(jpeg)),
    ]

    with patch("ai.video.temporal._haar_temporal_engine") as haar:
        from ai.video.temporal import _EmptyTemporalEngine

        haar.return_value = _EmptyTemporalEngine()
        result = analyze_temporal_consistency(frames)

    assert result["method"] == TEMPORAL_METHOD
    assert "error" not in result
    assert result["frames_with_face"] == 0
    assert result["track_count"] == 0
    assert result["primary_track"] is None
    assert any("unavailable" in note.lower() for note in result["notes"])


def test_analyze_temporal_uses_injected_detector(tmp_path: Path) -> None:
    jpeg = tmp_path / "frame_000000.jpg"
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(jpeg, format="JPEG")
    frames = [
        ExtractedFrame(frame_number=0, timestamp_seconds=0.0, frame_path=str(jpeg)),
        ExtractedFrame(frame_number=30, timestamp_seconds=1.0, frame_path=str(jpeg)),
    ]

    def detect_fn(path: Path) -> list[BoundingBox]:
        return [_box(8, 8)]

    result = analyze_temporal_consistency(frames, detect_fn=detect_fn)
    assert result["method"] == TEMPORAL_METHOD
    assert result["frames_with_face"] == 2
    assert result["track_count"] == 1


def test_phase4_aggregation_unchanged_with_temporal() -> None:
    frames = [
        ExtractedFrame(
            frame_number=0,
            timestamp_seconds=0.0,
            frame_path="a.jpg",
            prediction="REAL",
            confidence=80.0,
            real_probability=0.8,
            fake_probability=0.2,
        ),
        ExtractedFrame(
            frame_number=30,
            timestamp_seconds=1.0,
            frame_path="b.jpg",
            prediction="FAKE",
            confidence=60.0,
            real_probability=0.4,
            fake_probability=0.6,
        ),
        ExtractedFrame(
            frame_number=60,
            timestamp_seconds=2.0,
            frame_path="c.jpg",
            prediction="REAL",
            confidence=60.0,
            real_probability=0.6,
            fake_probability=0.4,
        ),
    ]
    cfg = InferenceConfig(threshold=0.5, device_preference="cpu")
    aggregation = aggregate_frame_predictions(frames, config=cfg)
    sampled = [(f.frame_number, f.timestamp_seconds) for f in frames]
    detections = [[_box(1, 1)], [], [_box(2, 2)]]
    tracks = associate_detections(
        [(n, t, boxes) for (n, t), boxes in zip(sampled, detections)]
    )
    temporal = build_temporal_result(sampled, detections, tracks)
    assert aggregation.fake_probability == pytest.approx(0.4)
    assert aggregation.prediction == "REAL"
    assert temporal["method"] == TEMPORAL_METHOD
    assert temporal["frames_with_face"] == 2


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = tmp_path
    application.config["SECRET_KEY"] = "test-secret-key"
    application.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_video_api_uses_lstm_not_frame_mean(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    register_and_login(client, username="temporaluser")
    case_id = client.post("/api/cases", json={"title": "Temporal"}).get_json()["data"]["case_id"]
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, fps=5.0, frames=10, size=(32, 32))
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), "clip.avi", "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    evidence_id = up.get_json()["data"]["evidence_id"]

    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(),
    ) as scored:
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    scored.assert_called_once()
    body = resp.get_json()["data"]
    assert body["prediction"] == "REAL"
    assert body["fake_probability"] == pytest.approx(0.09)
    assert body["video_analysis"]["frames_analyzed"] == 16
    assert "aggregation" not in body
    assert "temporal_analysis" not in body
