"""Phase 4 video-level mean aggregation of frame-level scores."""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.video.aggregator import aggregate_frame_predictions
from ai.video.exceptions import VideoAggregationError
from ai.video.frame_extractor import ExtractedFrame
from tests.conftest_product import register_and_login
from tests.test_video_frame_analysis import PRODUCT_CHECKPOINT, _fake_pipeline, _write_video


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


def _cfg() -> InferenceConfig:
    return InferenceConfig(threshold=0.5, device_preference="cpu")


def _frame(**kwargs) -> ExtractedFrame:
    defaults = dict(
        frame_number=0,
        timestamp_seconds=0.0,
        frame_path="frame_000000.jpg",
        prediction="REAL",
        confidence=80.0,
        real_probability=0.8,
        fake_probability=0.2,
    )
    defaults.update(kwargs)
    return ExtractedFrame(**defaults)


def test_mean_fake_probability_example() -> None:
    frames = [
        _frame(frame_number=0, fake_probability=0.2, real_probability=0.8, prediction="REAL"),
        _frame(frame_number=1, fake_probability=0.4, real_probability=0.6, prediction="REAL"),
        _frame(frame_number=2, fake_probability=0.6, real_probability=0.4, prediction="FAKE"),
    ]
    result = aggregate_frame_predictions(frames, config=_cfg())
    assert result.fake_probability == pytest.approx(0.4)
    assert result.real_probability == pytest.approx(0.6)
    assert result.prediction == "REAL"
    assert result.confidence == pytest.approx(60.0)
    assert result.method == "mean_frame_probability"
    assert result.total_frames_analyzed == 3
    assert result.real_frame_count == 2
    assert result.fake_frame_count == 1
    assert result.fake_frame_percentage == pytest.approx(100.0 / 3, rel=1e-3)
    assert [item["timestamp_seconds"] for item in result.suspicious_frames] == [
        frames[2].timestamp_seconds
    ]
    assert result.suspicious_frames[0]["frame_number"] == 2


def test_decision_matches_image_threshold() -> None:
    cfg = _cfg()
    frames = [
        _frame(fake_probability=0.7, real_probability=0.3, prediction="FAKE"),
        _frame(fake_probability=0.5, real_probability=0.5, prediction="FAKE"),
        _frame(fake_probability=0.6, real_probability=0.4, prediction="FAKE"),
    ]
    result = aggregate_frame_predictions(frames, config=cfg)
    expected = decide_from_probabilities(
        [result.real_probability, result.fake_probability], cfg
    )
    assert result.prediction == expected.predicted_label
    assert result.confidence == pytest.approx(expected.confidence)
    assert result.prediction == "FAKE"
    assert result.fake_frame_count == 3
    assert result.real_frame_count == 0


def test_near_threshold_does_not_inflate_confidence() -> None:
    frames = [
        _frame(fake_probability=0.51, real_probability=0.49, prediction="FAKE", confidence=51.0),
        _frame(fake_probability=0.49, real_probability=0.51, prediction="REAL", confidence=51.0),
    ]
    result = aggregate_frame_predictions(frames, config=_cfg())
    assert result.fake_probability == pytest.approx(0.5)
    assert result.real_probability == pytest.approx(0.5)
    assert result.confidence == pytest.approx(50.0)


def test_skips_missing_probabilities() -> None:
    frames = [
        _frame(fake_probability=0.2, real_probability=0.8, prediction="REAL"),
        _frame(fake_probability=None, real_probability=None, prediction=None),
        _frame(fake_probability=float("nan"), real_probability=0.5, prediction="REAL"),
        _frame(fake_probability=0.6, real_probability=0.4, prediction="FAKE", timestamp_seconds=4.0),
    ]
    result = aggregate_frame_predictions(frames, config=_cfg())
    assert result.total_frames == 4
    assert result.total_frames_analyzed == 2
    assert result.skipped_invalid_probability_count == 2
    assert result.fake_probability == pytest.approx(0.4)
    assert result.real_probability == pytest.approx(0.6)
    assert result.suspicious_frames[0]["timestamp_seconds"] == pytest.approx(4.0)


def test_empty_frame_list_errors() -> None:
    with pytest.raises(VideoAggregationError, match="No extracted frames"):
        aggregate_frame_predictions([], config=_cfg())


def test_all_invalid_probabilities_error() -> None:
    with pytest.raises(VideoAggregationError, match="No valid frame probabilities"):
        aggregate_frame_predictions(
            [_frame(real_probability=None, fake_probability=None)],
            config=_cfg(),
        )


def test_video_api_returns_lstm_prediction(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    register_and_login(client, username="agguser")
    case_id = client.post("/api/cases", json={"title": "Agg"}).get_json()["data"]["case_id"]
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, fps=5.0, frames=10, size=(32, 32))
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), "clip.avi", "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    uploaded = up.get_json()["data"]
    evidence_id = uploaded["evidence_id"]
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
    assert scored.call_args.kwargs["generate_xai"] is False
    body = resp.get_json()["data"]
    assert body["analysis_status"] == "COMPLETED"
    assert body["prediction"] == "REAL"
    assert body["confidence"] is None
    assert body["fake_probability"] == pytest.approx(0.09)
    assert body["real_probability"] == pytest.approx(0.91)
    video = body["video_analysis"]
    assert video["prediction"] == "REAL"
    assert video["p_fake"] == pytest.approx(0.09)
    assert video["threshold"] == pytest.approx(0.5)
    assert video["frames_analyzed"] == 16
    assert video["face_crop_frames"] + video["fallback_frames"] == 16
    assert video["p_fake_meaning"] == "model-predicted probability for the FAKE class"
    assert "aggregation" not in body
    listed = client.get(f"/api/evidence/{evidence_id}").get_json()["data"]
    assert listed["sha256"] == uploaded["sha256"]


def test_second_video_analysis_calls_the_video_service_again(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    register_and_login(client, username="reuseuser")
    case_id = client.post("/api/cases", json={"title": "Reuse"}).get_json()["data"]["case_id"]
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, fps=5.0, frames=10, size=(32, 32))
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), "clip.avi", "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    uploaded = up.get_json()["data"]
    evidence_id = uploaded["evidence_id"]
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(),
    ) as scored:
        first = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
        second = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert first.status_code == 201, first.get_json()
    assert second.status_code == 201, second.get_json()
    assert scored.call_count == 2
    assert second.get_json()["data"]["prediction"] == "REAL"
    listed = client.get(f"/api/evidence/{evidence_id}").get_json()["data"]
    assert listed["sha256"] == uploaded["sha256"]


def test_nan_is_not_a_valid_probability() -> None:
    assert math.isnan(float("nan"))
    result = aggregate_frame_predictions(
        [
            _frame(fake_probability=0.2, real_probability=0.8),
            _frame(fake_probability=float("inf"), real_probability=0.0),
        ],
        config=_cfg(),
    )
    assert result.total_frames_analyzed == 1
    assert result.fake_probability == pytest.approx(0.2)
