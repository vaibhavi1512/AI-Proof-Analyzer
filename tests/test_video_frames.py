"""Phase 2 video metadata and ~1 FPS frame extraction tests."""

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

from ai.video.exceptions import VideoPathError, VideoUnreadableError
from ai.video.frame_extractor import _safe_frame_path, extract_frames
from ai.video.metadata import (
    duration_from_count,
    read_video_metadata,
    sample_interval_frames,
    timestamp_for_frame,
)
from ai.video.video_config import VideoConfig
from tests.conftest_product import register_and_login


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


def _write_video(
    path: Path,
    *,
    fps: float,
    frames: int,
    size: tuple[int, int] = (48, 32),
    fourcc: str = "MJPG",
) -> Path:
    import cv2

    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (width, height)
    )
    assert writer.isOpened(), f"VideoWriter failed for {path}"
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (index * 17 % 256, 40, 200)
        writer.write(frame)
    writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _create_case(client) -> int:
    register_and_login(client, username="frameuser")
    resp = client.post("/api/cases", json={"title": "Frame Extraction Case"})
    return resp.get_json()["data"]["case_id"]


def test_sample_interval_and_duration_helpers() -> None:
    assert sample_interval_frames(30.0, 1.0) == 30
    assert sample_interval_frames(24.0, 1.0) == 24
    assert sample_interval_frames(5.0, 1.0) == 5
    assert sample_interval_frames(1.0, 1.0) == 1
    assert sample_interval_frames(0.0, 1.0) == 1
    assert duration_from_count(90, 30.0) == pytest.approx(3.0)
    assert duration_from_count(0, 30.0) == 0.0
    assert timestamp_for_frame(0, 30.0) == pytest.approx(0.0)
    assert timestamp_for_frame(30, 30.0) == pytest.approx(1.0)
    assert timestamp_for_frame(60, 30.0) == pytest.approx(2.0)


def test_valid_video_metadata_fps_count_resolution_duration(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi", fps=10.0, frames=25, size=(64, 48))
    meta = read_video_metadata(video)
    assert meta.fps == pytest.approx(10.0, rel=0.05)
    assert meta.frame_count == pytest.approx(25, abs=1)
    assert meta.width == 64
    assert meta.height == 48
    assert meta.duration_seconds == pytest.approx(meta.frame_count / meta.fps, rel=0.05)
    assert meta.sampling_rate_fps == pytest.approx(1.0)
    assert meta.sample_interval_frames == sample_interval_frames(meta.fps, 1.0)
    assert meta.extracted_frame_count == 0


def test_approximately_one_fps_sampling_and_timestamps(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "tenfps.avi", fps=10.0, frames=30, size=(40, 40))
    out = tmp_path / "frames"
    result = extract_frames(video, out, config=VideoConfig(target_sample_fps=1.0))
    interval = sample_interval_frames(result.metadata.fps, 1.0)
    expected_indexes = list(range(0, 30, interval))
    assert [frame.frame_number for frame in result.frames] == expected_indexes
    assert result.metadata.extracted_frame_count == len(expected_indexes)
    for frame in result.frames:
        assert frame.timestamp_seconds == pytest.approx(
            timestamp_for_frame(frame.frame_number, result.metadata.fps)
        )
        assert frame.prediction is None
        assert frame.confidence is None
        stored = Path(frame.frame_path)
        assert stored.is_file()
        assert stored.stat().st_size > 0
        assert stored.parent == out.resolve()
        with Image.open(stored) as image:
            assert image.size == (40, 40)
    assert (out / "metadata.json").is_file()
    # 30 frames at 10 FPS → ~3s → about 3 samples at 1 FPS
    assert 2 <= len(result.frames) <= 4


def test_short_video_extracts_at_least_one_frame(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "tiny.avi", fps=30.0, frames=3, size=(32, 32))
    result = extract_frames(video, tmp_path / "short_frames")
    assert result.metadata.extracted_frame_count >= 1
    assert result.frames[0].frame_number == 0
    assert result.frames[0].timestamp_seconds == pytest.approx(0.0)
    assert Path(result.frames[0].frame_path).is_file()
    duration = result.metadata.duration_seconds
    assert duration < 1.0 or result.metadata.frame_count <= 3


def test_corrupt_video_metadata_and_extraction_fail_gracefully(tmp_path: Path) -> None:
    junk = tmp_path / "bad.mp4"
    junk.write_bytes(b"not a video file" + b"\x00" * 64)
    with pytest.raises(VideoUnreadableError):
        read_video_metadata(junk)
    with pytest.raises(VideoUnreadableError):
        extract_frames(junk, tmp_path / "out")
    leftover = list((tmp_path / "out").glob("frame_*")) if (tmp_path / "out").exists() else []
    assert leftover == []


def test_missing_file_is_unreadable(tmp_path: Path) -> None:
    with pytest.raises(VideoUnreadableError):
        read_video_metadata(tmp_path / "missing.avi")


def test_frame_path_stays_inside_output_dir(tmp_path: Path) -> None:
    dest = tmp_path / "safe"
    dest.mkdir()
    path = _safe_frame_path(dest, 12, ".jpg")
    assert path.parent == dest.resolve()
    assert path.name == "frame_000012.jpg"
    with pytest.raises(VideoPathError):
        _safe_frame_path(dest, 1, "/../escape.jpg")


def test_prepare_video_evidence_stores_frames_under_case(app, client, tmp_path: Path) -> None:
    case_id = _create_case(client)
    video_path = tmp_path / "upload.avi"
    _write_video(video_path, fps=8.0, frames=16, size=(36, 36))
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), "clip.avi", "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.get_json()
    evidence_id = resp.get_json()["data"]["evidence_id"]

    from backend.app.services.video_service import prepare_video_evidence
    from backend.app.models.entities import User

    with app.app_context():
        user = User.query.filter_by(username="frameuser").one()
        result = prepare_video_evidence(user, evidence_id)

    upload_root = Path(app.config["UPLOAD_DIR"]).resolve()
    expected_dir = upload_root / "cases" / str(case_id) / "frames" / str(evidence_id)
    assert expected_dir.is_dir()
    assert result.output_dir == expected_dir.relative_to(upload_root).as_posix()
    assert ".." not in result.output_dir
    written = list(expected_dir.glob("frame_*.jpg"))
    assert written
    assert len(written) == result.metadata.extracted_frame_count
    sidecar = expected_dir / "metadata.json"
    assert sidecar.is_file()
    for frame in result.frames:
        assert not Path(frame.frame_path).is_absolute()
        assert (upload_root / frame.frame_path).is_file()


def test_prepare_rejects_image_evidence(app, client) -> None:
    case_id = _create_case(client)
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "still.png", "image/png")},
        content_type="multipart/form-data",
    )
    evidence_id = resp.get_json()["data"]["evidence_id"]
    from backend.app.exceptions import ValidationError
    from backend.app.models.entities import User
    from backend.app.services.video_service import prepare_video_evidence

    with app.app_context():
        user = User.query.filter_by(username="frameuser").one()
        with pytest.raises(ValidationError, match="not a video"):
            prepare_video_evidence(user, evidence_id)


def test_analyze_video_reaches_lstm_service(app, client, tmp_path: Path) -> None:
    from unittest.mock import patch

    from tests.ffpp_video_fakes import lstm_result

    case_id = _create_case(client)
    video_path = tmp_path / "analyze.avi"
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
        analyze = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert analyze.status_code == 201, analyze.get_json()
    scored.assert_called_once()
    body = analyze.get_json()["data"]
    assert body["prediction"] == "REAL"
    assert body["video_analysis"]["frames_analyzed"] == 16
    listed = client.get(f"/api/evidence/{evidence_id}").get_json()["data"]
    assert listed["sha256"] == uploaded["sha256"]


def test_image_analyze_path_is_unchanged(client) -> None:
    from unittest.mock import patch

    from ai.inference.result import InvestigationResult

    case_id = _create_case(client)
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "face.png", "image/png")},
        content_type="multipart/form-data",
    )
    evidence_id = up.get_json()["data"]["evidence_id"]
    fake = InvestigationResult(
        investigation_id="INV-2026-000001",
        prediction="REAL",
        confidence=80.0,
        confidence_level="Medium",
        real_probability=0.8,
        fake_probability=0.2,
        threshold=0.5,
        model_name="efficientnet_b0",
        model_version="test",
        dataset_version="test",
        prediction_time_ms=1.0,
        timestamp="t",
        image_name="face.png",
        image_size=(224, 224),
        processing_device="CPU",
        processing_status="success",
    )
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=fake,
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False},
        )
    assert resp.status_code == 201, resp.get_json()
    assert resp.get_json()["data"]["prediction"] == "REAL"
