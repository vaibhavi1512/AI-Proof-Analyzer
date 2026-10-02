"""Phase 15 backend tests for the frozen 16-frame video model."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ffpp_video.constants import IMAGE_SIZE, NUM_FRAMES
from ai.ffpp_video.exceptions import InsufficientFramesError, VideoOpenError
from ai.ffpp_video.face_detector import DetectedFace, HaarFaceDetector
from ai.ffpp_video.face_preprocess import process_sampled_video
from ai.ffpp_video.sampling import SampledFrame, SampledVideo, uniform_frame_indices
from tests.conftest_product import register_and_login
from tests.ffpp_video_fakes import lstm_result

VISUAL_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "video" / "visual_model.pt"
LSTM_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "video" / "lstm.pt"


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = tmp_path
    application.config["REPORT_DIR"] = tmp_path / "reports"
    application.config["SECRET_KEY"] = "test-secret-key"
    application.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    application.config["VIDEO_VISUAL_CHECKPOINT"] = str(VISUAL_CHECKPOINT)
    application.config["VIDEO_LSTM_CHECKPOINT"] = str(LSTM_CHECKPOINT)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(12, 24, 36)).save(buf, format="PNG")
    return buf.getvalue()


def _write_video(path: Path, *, frames: int, fps: float = 8.0, size: tuple[int, int] = (64, 48)) -> None:
    import cv2

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (index * 11 % 255, 30, 90)
        writer.write(frame)
    writer.release()


def _sampled_blank() -> SampledVideo:
    frames = []
    for index in range(NUM_FRAMES):
        image = np.zeros((80, 96, 3), dtype=np.uint8)
        frames.append(
            SampledFrame(frame_index=index * 4, timestamp_seconds=index * 0.125, frame=image)
        )
    return SampledVideo(
        video_path=Path("blank.mp4"),
        fps=8.0,
        total_frames=64,
        duration_seconds=8.0,
        frames=tuple(frames),
    )


def test_uniform_sample_keeps_sixteen_frames_in_order() -> None:
    indices = uniform_frame_indices(240, NUM_FRAMES)
    assert len(indices) == NUM_FRAMES
    assert indices == sorted(indices)
    assert len(set(indices)) == NUM_FRAMES
    assert indices[0] == 0
    assert indices[-1] == 239


def test_face_crop_keeps_order_and_shape() -> None:
    class _Face:
        def detect(self, _frame):
            return [DetectedFace(x=10, y=12, width=36, height=36, confidence=3.0)]

    processed = process_sampled_video(_sampled_blank(), _Face())
    assert len(processed.frames) == NUM_FRAMES
    assert processed.face_crop_count == NUM_FRAMES
    assert processed.fallback_count == 0
    assert processed.arrays.shape == (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert [frame.metadata.frame_index for frame in processed.frames] == [index * 4 for index in range(NUM_FRAMES)]
    assert all(frame.metadata.used_face_crop for frame in processed.frames)


def test_no_face_uses_full_frame_and_keeps_every_frame() -> None:
    processed = process_sampled_video(_sampled_blank(), HaarFaceDetector())
    assert len(processed.frames) == NUM_FRAMES
    assert processed.fallback_count == NUM_FRAMES
    assert processed.face_crop_count == 0
    assert processed.arrays.shape[0] == NUM_FRAMES
    assert all(frame.metadata.used_full_frame_fallback for frame in processed.frames)
    assert all(not frame.metadata.face_detected for frame in processed.frames)
    assert [frame.metadata.frame_index for frame in processed.frames] == [index * 4 for index in range(NUM_FRAMES)]


def test_temporal_importance_covers_sixteen_frames() -> None:
    import torch

    from ai.ffpp_video.temporal_importance import (
        normalize_temporal_importance,
        temporal_gradient_magnitude,
    )

    magnitude = temporal_gradient_magnitude(torch.ones(NUM_FRAMES, 1280))
    importance = normalize_temporal_importance(magnitude)
    assert tuple(importance.shape) == (NUM_FRAMES,)
    assert float(importance.sum()) == pytest.approx(1.0)


def test_short_video_is_rejected_before_the_model(tmp_path: Path) -> None:
    from ai.ffpp_video.analyze import analyze_video

    video = tmp_path / "short.avi"
    _write_video(video, frames=4)
    with pytest.raises(InsufficientFramesError):
        analyze_video(video, False, visual_checkpoint=VISUAL_CHECKPOINT, lstm_checkpoint=LSTM_CHECKPOINT)


def test_missing_video_is_rejected(tmp_path: Path) -> None:
    from ai.ffpp_video.analyze import analyze_video

    with pytest.raises(VideoOpenError):
        analyze_video(tmp_path / "missing.mp4", False)


def test_image_evidence_still_uses_image_pipeline(client) -> None:
    from unittest.mock import patch

    from ai.inference.result import InvestigationResult

    register_and_login(client, username="imagephase15")
    case_id = client.post("/api/cases", json={"title": "Image"}).get_json()["data"]["case_id"]
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "still.png", "image/png")},
        content_type="multipart/form-data",
    )
    evidence_id = up.get_json()["data"]["evidence_id"]
    fake = InvestigationResult(
        investigation_id="INV-2026-000111",
        prediction="FAKE",
        confidence=77.0,
        confidence_level="Medium",
        real_probability=0.23,
        fake_probability=0.77,
        threshold=0.5,
        model_name="efficientnet_b0",
        model_version="image",
        dataset_version="maya",
        prediction_time_ms=1.0,
        timestamp="t",
        image_name="still.png",
        image_size=(224, 224),
        processing_device="CPU",
        processing_status="success",
    )
    with (
        patch("backend.app.services.analysis_service.run_inference", return_value=fake) as infer,
        patch("backend.app.services.video_model_service.analyze_video") as video,
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    infer.assert_called_once()
    video.assert_not_called()
    body = resp.get_json()["data"]
    assert body["prediction"] == "FAKE"
    assert body["confidence"] == pytest.approx(77.0)
    assert "video_analysis" not in body


def test_video_api_keeps_sha256_and_audit_trail(app, client, tmp_path: Path) -> None:
    from unittest.mock import patch

    from backend.app.models.entities import AuditLog

    register_and_login(client, username="custody15")
    case_id = client.post("/api/cases", json={"title": "Custody"}).get_json()["data"]["case_id"]
    video = tmp_path / "clip.avi"
    _write_video(video, frames=20)
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video.read_bytes()), "clip.avi", "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    uploaded = up.get_json()["data"]
    evidence_id = uploaded["evidence_id"]
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(fallback_frames=1),
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": True},
        )
    assert resp.status_code == 201, resp.get_json()
    analysis_id = resp.get_json()["data"]["analysis_id"]
    report = client.post(f"/api/analysis/{analysis_id}/report", json={})
    assert report.status_code == 201, report.get_json()
    listed = client.get(f"/api/evidence/{evidence_id}").get_json()["data"]
    assert listed["sha256"] == uploaded["sha256"]
    video_body = resp.get_json()["data"]["video_analysis"]
    assert video_body["frames_analyzed"] == 16
    assert video_body["fallback_frames"] == 1
    dumped = str(video_body)
    assert "dataset_vid" not in dumped
    assert "best.pt" not in dumped
    with app.app_context():
        events = {row.event_type for row in AuditLog.query.filter_by(evidence_id=evidence_id).all()}
    assert "ANALYSIS_STARTED" in events
    assert "ANALYSIS_COMPLETED" in events
    assert "EVIDENCE_UPLOADED" in events or "INTEGRITY_VERIFIED" in events


def test_short_uploaded_video_stays_a_validation_error(client, tmp_path: Path) -> None:
    register_and_login(client, username="short15")
    case_id = client.post("/api/cases", json={"title": "Short"}).get_json()["data"]["case_id"]
    video = tmp_path / "short.avi"
    _write_video(video, frames=4)
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video.read_bytes()), "short.avi", "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    assert up.status_code == 201, up.get_json()
    uploaded = up.get_json()["data"]
    resp = client.post(
        f"/api/evidence/{uploaded['evidence_id']}/analyze",
        json={"generate_explanation": False, "verify_before_analyze": False},
    )
    assert resp.status_code == 400, resp.get_json()
    listed = client.get(f"/api/evidence/{uploaded['evidence_id']}").get_json()["data"]
    assert listed["sha256"] == uploaded["sha256"]


def test_default_video_checkpoints_live_in_this_repository() -> None:
    from backend.app.config.config import BaseConfig

    visual = Path(BaseConfig.VIDEO_VISUAL_CHECKPOINT).resolve()
    lstm = Path(BaseConfig.VIDEO_LSTM_CHECKPOINT).resolve()
    assert visual == VISUAL_CHECKPOINT.resolve()
    assert lstm == LSTM_CHECKPOINT.resolve()
    assert visual.is_file() and lstm.is_file()
    assert "dataset_vid" not in str(visual)
    assert "dataset_vid" not in str(lstm)


@pytest.mark.skipif(
    not VISUAL_CHECKPOINT.is_file() or not LSTM_CHECKPOINT.is_file(),
    reason="Repository video checkpoints are not present",
)
def test_checkpoints_load_once_and_are_reused() -> None:
    from ai.ffpp_video.analyze import VideoRuntime, get_video_runtime, reset_video_runtime

    reset_video_runtime()
    calls = {"count": 0}
    original = VideoRuntime.__init__

    def _counted(self, visual_path, lstm_path):
        calls["count"] += 1
        original(self, visual_path, lstm_path)

    VideoRuntime.__init__ = _counted  # type: ignore[method-assign]
    try:
        first = get_video_runtime(VISUAL_CHECKPOINT, LSTM_CHECKPOINT)
        second = get_video_runtime(VISUAL_CHECKPOINT, LSTM_CHECKPOINT)
    finally:
        VideoRuntime.__init__ = original  # type: ignore[method-assign]
        reset_video_runtime()
    assert first is second
    assert calls["count"] == 1
    assert first.visual_model_name == "ffpp_visual_efficientnet_b0"
    assert first.model_name == "ffpp_video_lstm_v2"
    assert first.lstm.sequence_length == NUM_FRAMES
    assert first.lstm.num_layers == 2
    assert Path(first.visual_checkpoint).resolve() == VISUAL_CHECKPOINT.resolve()
    assert Path(first.lstm_checkpoint).resolve() == LSTM_CHECKPOINT.resolve()
    assert "dataset_vid" not in first.visual_checkpoint
    assert "dataset_vid" not in first.lstm_checkpoint
