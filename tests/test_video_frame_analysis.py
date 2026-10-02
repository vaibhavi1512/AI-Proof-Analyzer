"""Phase 3 frame-level analysis using the existing image inference pipeline."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.inference.confidence import ConfidenceDecision
from ai.inference.inference_config import InferenceConfig
from ai.inference.pipeline import InferencePipeline
from ai.video.frame_analyzer import analyze_extracted_frames
from ai.video.frame_extractor import ExtractedFrame, VideoExtractionResult, extract_frames
from ai.video.metadata import VideoMetadata, timestamp_for_frame
from tests.conftest_product import register_and_login

PRODUCT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "processed_final" / "best.pt"


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


def _product_config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig(
        project_root=ROOT,
        checkpoint_path=PRODUCT_CHECKPOINT,
        artifact_dir=tmp_path / "artifacts" / "phase3" / "sprint4",
        id_state_path=tmp_path / "artifacts" / "phase3" / "sprint4" / "investigation_id_state.json",
        device_preference="cpu",
        threshold=0.5,
        image_size=224,
    )


def _jpeg(path: Path, color=(40, 90, 140)) -> Path:
    Image.new("RGB", (64, 48), color=color).save(path, format="JPEG")
    return path


def _write_video(path: Path, *, fps: float, frames: int, size: tuple[int, int] = (48, 32)) -> Path:
    import cv2
    import numpy as np

    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (width, height)
    )
    assert writer.isOpened(), f"VideoWriter failed for {path}"
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (index * 17 % 256, 40, 200)
        writer.write(frame)
    writer.release()
    return path


def _skip_without_checkpoint() -> None:
    if not PRODUCT_CHECKPOINT.is_file():
        pytest.skip("processed_final/best.pt checkpoint required")


def test_image_pipeline_still_uses_product_checkpoint(tmp_path: Path) -> None:
    _skip_without_checkpoint()
    from backend.app.integrations.ai_bridge import reset_inference_pipeline, get_inference_pipeline

    reset_inference_pipeline()
    pipeline = get_inference_pipeline()
    loaded = pipeline.loader.load()
    assert loaded.checkpoint_path.resolve() == PRODUCT_CHECKPOINT.resolve()
    image = _jpeg(tmp_path / "still.jpg")
    result = pipeline.run(image, artifact_dir_override=tmp_path / "inv")
    assert result.prediction in {"REAL", "FAKE"}
    assert 0.0 <= result.real_probability <= 1.0
    assert 0.0 <= result.fake_probability <= 1.0
    assert result.real_probability + result.fake_probability == pytest.approx(1.0, abs=1e-5)
    reset_inference_pipeline()


def test_frame_goes_through_existing_image_model(tmp_path: Path) -> None:
    _skip_without_checkpoint()
    cfg = _product_config(tmp_path)
    pipeline = InferencePipeline(cfg)
    frame_path = _jpeg(tmp_path / "frame_000000.jpg")
    decision = pipeline.classify_image(frame_path)
    assert decision.predicted_label in {"REAL", "FAKE"}
    assert 0.0 <= decision.real_probability <= 1.0
    assert 0.0 <= decision.fake_probability <= 1.0
    assert decision.real_probability + decision.fake_probability == pytest.approx(1.0, abs=1e-5)
    expected_conf = round(
        (decision.fake_probability if decision.predicted_label == "FAKE" else decision.real_probability)
        * 100.0,
        4,
    )
    assert decision.confidence == pytest.approx(expected_conf)
    assert pipeline.loader.load().checkpoint_path.resolve() == PRODUCT_CHECKPOINT.resolve()


def test_frame_probabilities_match_image_pipeline(tmp_path: Path) -> None:
    _skip_without_checkpoint()
    cfg = _product_config(tmp_path)
    pipeline = InferencePipeline(cfg)
    image = _jpeg(tmp_path / "same.jpg")
    investigation = pipeline.run(image, artifact_dir_override=tmp_path / "inv")
    decision = pipeline.classify_image(image)
    assert decision.predicted_label == investigation.prediction
    assert decision.real_probability == pytest.approx(investigation.real_probability, abs=1e-6)
    assert decision.fake_probability == pytest.approx(investigation.fake_probability, abs=1e-6)
    assert decision.confidence == pytest.approx(investigation.confidence, abs=1e-6)


def test_extracted_frames_keep_number_and_timestamp(tmp_path: Path) -> None:
    _skip_without_checkpoint()
    video = _write_video(tmp_path / "clip.avi", fps=10.0, frames=20, size=(40, 40))
    extracted = extract_frames(video, tmp_path / "frames")
    pipeline = InferencePipeline(_product_config(tmp_path))
    first_loader = pipeline.loader.load()
    analyzed = analyze_extracted_frames(extracted, pipeline=pipeline)
    again = pipeline.loader.load()
    assert again is first_loader
    assert str(first_loader.checkpoint_path).replace("\\", "/").endswith(
        "artifacts/checkpoints/processed_final/best.pt"
    )
    assert analyzed.frames
    for frame in analyzed.frames:
        assert frame.prediction in {"REAL", "FAKE"}
        assert frame.timestamp_seconds == pytest.approx(
            timestamp_for_frame(frame.frame_number, analyzed.metadata.fps)
        )
        assert frame.real_probability is not None
        assert frame.fake_probability is not None
        sidecar_path = Path(frame.frame_path)
        assert sidecar_path.is_file()


def test_phase2_extraction_still_leaves_empty_slots_until_analyzer(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "raw.avi", fps=8.0, frames=8, size=(32, 32))
    result = extract_frames(video, tmp_path / "phase2")
    assert result.frames
    for frame in result.frames:
        assert frame.prediction is None
        assert frame.confidence is None
        assert Path(frame.frame_path).is_file()
    assert (tmp_path / "phase2" / "metadata.json").is_file()


def test_analyze_extracted_frames_uses_injected_pipeline(tmp_path: Path) -> None:
    frame_path = _jpeg(tmp_path / "frame_000030.jpg")
    result = VideoExtractionResult(
        metadata=VideoMetadata(
            duration_seconds=2.0,
            fps=30.0,
            frame_count=60,
            width=64,
            height=48,
            extracted_frame_count=1,
        ),
        frames=[
            ExtractedFrame(
                frame_number=30,
                timestamp_seconds=1.0,
                frame_path=str(frame_path),
            )
        ],
        output_dir=str(tmp_path),
    )
    decision = ConfidenceDecision(
        predicted_class=1,
        predicted_label="FAKE",
        confidence=82.0,
        confidence_level="Medium",
        real_probability=0.18,
        fake_probability=0.82,
        threshold=0.5,
        threshold_decision="FAKE (>= threshold 0.5)",
    )

    class _Loader:
        def load(self):
            return type(
                "Loaded",
                (),
                {
                    "checkpoint_path": PRODUCT_CHECKPOINT,
                    "model_name": "efficientnet_b0",
                },
            )()

    class _Pipeline:
        loader = _Loader()

        def classify_image(self, path):
            assert Path(path) == frame_path
            return decision

    analyze_extracted_frames(result, pipeline=_Pipeline())
    frame = result.frames[0]
    assert frame.frame_number == 30
    assert frame.timestamp_seconds == pytest.approx(1.0)
    assert frame.prediction == "FAKE"
    assert frame.real_probability == pytest.approx(0.18)
    assert frame.fake_probability == pytest.approx(0.82)
    assert frame.confidence == pytest.approx(82.0)


def _fake_pipeline() -> object:
    decision = ConfidenceDecision(
        predicted_class=0,
        predicted_label="REAL",
        confidence=91.0,
        confidence_level="High",
        real_probability=0.91,
        fake_probability=0.09,
        threshold=0.5,
        threshold_decision="REAL (< threshold 0.5)",
    )

    class _Loaded:
        checkpoint_path = PRODUCT_CHECKPOINT
        model_name = "efficientnet_b0"
        model_version = "processed_final-best"

    class _Loader:
        def load(self):
            return _Loaded()

    class _Pipeline:
        loader = _Loader()
        config = InferenceConfig(
            project_root=ROOT,
            checkpoint_path=PRODUCT_CHECKPOINT,
            device_preference="cpu",
            threshold=0.5,
        )

        def classify_image(self, path):
            suffix = Path(path).suffix.lower()
            assert suffix not in {".mp4", ".avi", ".mov", ".mkv"}, path
            assert suffix in {".jpg", ".jpeg"}
            return decision

    return _Pipeline()


def test_video_analyze_api_reaches_lstm_service(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    register_and_login(client, username="phase3user")
    case_id = client.post("/api/cases", json={"title": "Phase 3 Case"}).get_json()["data"]["case_id"]
    video_path = tmp_path / "upload.avi"
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
    assert body["confidence"] is None
    assert body["model_name"] == "ffpp_video_lstm_v2"
    video = body["video_analysis"]
    assert video["frames_analyzed"] == 16
    assert video["p_fake"] == pytest.approx(0.09)
    assert [frame["frame_index"] for frame in video["frames"]] == [index * 5 for index in range(16)]


def test_video_analyze_never_sends_container_to_image_preprocess(app, client, tmp_path: Path) -> None:
    """Regression: Evidence 7 sent the original .mp4 into InferencePipeline.run."""

    from tests.ffpp_video_fakes import lstm_result

    register_and_login(client, username="mp4guard")
    case_id = client.post("/api/cases", json={"title": "MP4 Guard"}).get_json()["data"]["case_id"]
    video_path = tmp_path / "test.mp4"
    _write_video(video_path, fps=5.0, frames=10, size=(32, 32))
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), "c037a2757db5498398dfd4f7f7667149.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    evidence_id = up.get_json()["data"]["evidence_id"]

    def _forbid_container(path, **_kwargs):
        raise AssertionError(f"run_inference received video container {path}")

    with (
        patch("backend.app.services.analysis_service.run_inference", side_effect=_forbid_container) as infer,
        patch(
            "backend.app.services.video_model_service.analyze_video",
            return_value=lstm_result(),
        ),
        patch("backend.app.services.analysis_service.run_explanation") as explain,
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    infer.assert_not_called()
    explain.assert_not_called()
    body = resp.get_json()["data"]
    assert body["prediction"] == "REAL"
    assert body["video_analysis"]["frames_analyzed"] == 16


def test_video_container_not_preprocessed_if_media_type_is_image(app, client, tmp_path: Path) -> None:
    """A video container is scored by the video model even if media_type was mistagged."""

    from backend.app.extensions import db
    from backend.app.models.entities import Evidence
    from tests.ffpp_video_fakes import lstm_result

    register_and_login(client, username="mistag")
    case_id = client.post("/api/cases", json={"title": "Mistag"}).get_json()["data"]["case_id"]
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path, fps=5.0, frames=10, size=(32, 32))
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), "clip.mp4", "video/mp4")},
        content_type="multipart/form-data",
    )
    evidence_id = up.get_json()["data"]["evidence_id"]
    with app.app_context():
        row = db.session.get(Evidence, evidence_id)
        assert row is not None
        row.media_type = "image"
        db.session.commit()

    def _forbid_container(path, **_kwargs):
        raise AssertionError(f"run_inference received {path}")

    with (
        patch("backend.app.services.analysis_service.run_inference", side_effect=_forbid_container) as infer,
        patch(
            "backend.app.services.video_model_service.analyze_video",
            return_value=lstm_result(),
        ) as scored,
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    infer.assert_not_called()
    scored.assert_called_once()
    assert resp.get_json()["data"]["video_analysis"]["prediction"] == "REAL"


def test_image_analyze_still_unchanged_with_mocked_inference(client) -> None:
    from ai.inference.result import InvestigationResult

    register_and_login(client, username="imagephase3")
    case_id = client.post("/api/cases", json={"title": "Image Case"}).get_json()["data"]["case_id"]
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(buf, format="PNG")
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(buf.getvalue()), "face.png", "image/png")},
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
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    data = resp.get_json()["data"]
    assert data["prediction"] == "REAL"
    assert "frames" not in data
