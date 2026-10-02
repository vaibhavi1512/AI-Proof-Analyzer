"""Phase 6: reuse still-image Grad-CAM on selected video JPEG frames."""

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

from ai.inference.confidence import ConfidenceDecision
from ai.inference.inference_config import InferenceConfig
from ai.video.frame_extractor import ExtractedFrame
from ai.video.frame_xai import (
    MAX_VIDEO_XAI_FRAMES,
    explain_video_frames,
    select_video_xai_frames,
)
from tests.conftest_product import register_and_login
from tests.test_video_frame_analysis import PRODUCT_CHECKPOINT, _write_video

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


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


def _jpeg(path: Path, color=(40, 90, 140)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path, format="JPEG")
    return path


def _frame(
    tmp_path: Path,
    number: int,
    *,
    prediction: str,
    confidence: float,
    fake_probability: float,
    real_probability: float | None = None,
) -> ExtractedFrame:
    path = _jpeg(tmp_path / f"frame_{number:06d}.jpg")
    real = 1.0 - fake_probability if real_probability is None else real_probability
    return ExtractedFrame(
        frame_number=number,
        timestamp_seconds=float(number),
        frame_path=str(path),
        prediction=prediction,
        confidence=confidence,
        real_probability=real,
        fake_probability=fake_probability,
    )


def _fake_explain(image_path, *, investigation_id, explainer, artifact_dir):
    path = Path(image_path)
    assert path.suffix.lower() not in _VIDEO_SUFFIXES, path
    assert path.suffix.lower() in {".jpg", ".jpeg"}
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    heatmap = out / "heatmap.png"
    overlay = out / "overlay.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(heatmap)
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(overlay)
    json_path = out / "explanation_result.json"
    json_path.write_text("{}", encoding="utf-8")
    return {
        "explainer": explainer,
        "heatmap": str(heatmap),
        "overlay": str(overlay),
        "explanation_json": str(json_path),
        "prediction": "REAL",
        "confidence": 90.0,
        "target_class": 0,
        "investigation_id": investigation_id,
    }


def _decision(
    label: str,
    *,
    confidence: float,
    real_probability: float,
    fake_probability: float,
) -> ConfidenceDecision:
    return ConfidenceDecision(
        predicted_class=1 if label == "FAKE" else 0,
        predicted_label=label,
        confidence=confidence,
        confidence_level="High",
        real_probability=real_probability,
        fake_probability=fake_probability,
        threshold=0.5,
        threshold_decision=(
            "FAKE (>= threshold 0.5)" if label == "FAKE" else "REAL (< threshold 0.5)"
        ),
    )


def _pipeline_from_scores(scores: list[ConfidenceDecision]):
    cursor = {"i": 0}

    class _Loaded:
        checkpoint_path = PRODUCT_CHECKPOINT

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
            assert suffix not in _VIDEO_SUFFIXES, path
            decision = scores[min(cursor["i"], len(scores) - 1)]
            cursor["i"] += 1
            return decision

    return _Pipeline()


def test_select_prefers_fake_frames_and_caps_at_three(tmp_path: Path) -> None:
    frames = [
        _frame(tmp_path, 0, prediction="REAL", confidence=99.0, fake_probability=0.1),
        _frame(tmp_path, 1, prediction="FAKE", confidence=60.0, fake_probability=0.61),
        _frame(tmp_path, 2, prediction="FAKE", confidence=70.0, fake_probability=0.92),
        _frame(tmp_path, 3, prediction="FAKE", confidence=65.0, fake_probability=0.80),
        _frame(tmp_path, 4, prediction="FAKE", confidence=55.0, fake_probability=0.70),
        _frame(tmp_path, 5, prediction="FAKE", confidence=50.0, fake_probability=0.66),
    ]
    selected = select_video_xai_frames(frames, video_prediction="REAL")
    assert MAX_VIDEO_XAI_FRAMES == 3
    assert [frame.frame_number for frame in selected] == [2, 3, 4]
    assert all(_label == "FAKE" for _label in [frame.prediction for frame in selected])


def test_select_all_real_picks_one_matching_video_prediction(tmp_path: Path) -> None:
    frames = [
        _frame(tmp_path, 0, prediction="REAL", confidence=70.0, fake_probability=0.2),
        _frame(tmp_path, 1, prediction="REAL", confidence=95.0, fake_probability=0.05),
        _frame(tmp_path, 2, prediction="REAL", confidence=80.0, fake_probability=0.1),
    ]
    selected = select_video_xai_frames(frames, video_prediction="REAL")
    assert len(selected) == 1
    assert selected[0].frame_number == 1


def test_select_ignores_video_containers(tmp_path: Path) -> None:
    jpeg = _frame(tmp_path, 0, prediction="REAL", confidence=90.0, fake_probability=0.1)
    mp4 = ExtractedFrame(
        frame_number=1,
        timestamp_seconds=1.0,
        frame_path=str(tmp_path / "clip.mp4"),
        prediction="FAKE",
        confidence=99.0,
        real_probability=0.01,
        fake_probability=0.99,
    )
    (tmp_path / "clip.mp4").write_bytes(b"not-a-frame")
    selected = select_video_xai_frames([mp4, jpeg], video_prediction="REAL")
    assert [frame.frame_number for frame in selected] == [0]


def test_explain_writes_separate_frame_directories(tmp_path: Path) -> None:
    frames = [
        _frame(tmp_path, 0, prediction="FAKE", confidence=80.0, fake_probability=0.8),
        _frame(tmp_path, 30, prediction="FAKE", confidence=70.0, fake_probability=0.7),
    ]
    artifact_root = tmp_path / "artifacts" / "investigations" / "INV-TEST"
    results = explain_video_frames(
        frames,
        investigation_id="INV-TEST",
        explainer="gradcam",
        artifact_root=artifact_root,
        path_root=None,
        explain_fn=_fake_explain,
    )
    assert len(results) == 2
    dir_a = artifact_root / "xai" / "gradcam" / "frame_000000"
    dir_b = artifact_root / "xai" / "gradcam" / "frame_000030"
    assert (dir_a / "heatmap.png").is_file()
    assert (dir_b / "heatmap.png").is_file()
    assert dir_a != dir_b
    assert results[0]["heatmap"] != results[1]["heatmap"]


def _upload_video(client, tmp_path: Path, name: str = "clip.avi") -> int:
    register_and_login(client, username=f"xai_{name.split('.')[0]}")
    case_id = client.post("/api/cases", json={"title": "Video XAI"}).get_json()["data"]["case_id"]
    video_path = tmp_path / name
    _write_video(video_path, fps=5.0, frames=30, size=(32, 32))
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(video_path.read_bytes()), name, "video/x-msvideo")},
        content_type="multipart/form-data",
    )
    return up.get_json()["data"]["evidence_id"]


def test_generate_explanation_false_skips_xai(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    evidence_id = _upload_video(client, tmp_path, "skip.avi")
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(with_xai=False),
    ) as scored:
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    assert scored.call_args.kwargs["generate_xai"] is False
    body = resp.get_json()["data"]
    assert body["prediction"] == "REAL"
    assert body["video_analysis"]["xai_available"] is False
    assert "xai" not in body["video_analysis"]


def test_video_xai_returns_sixteen_temporal_scores(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    evidence_id = _upload_video(client, tmp_path, "real.mp4")

    def _forbid_container(path, **_kwargs):
        raise AssertionError(f"image pipeline received {path}")

    with (
        patch("backend.app.services.analysis_service.run_inference", side_effect=_forbid_container),
        patch("ai.inference.preprocessing.preprocess_image") as preprocess,
        patch(
            "backend.app.services.video_model_service.analyze_video",
            return_value=lstm_result(with_xai=True),
        ) as scored,
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    preprocess.assert_not_called()
    assert scored.call_args.kwargs["generate_xai"] is True
    body = resp.get_json()["data"]
    xai = body["video_analysis"]["xai"]
    assert body["video_analysis"]["xai_available"] is True
    assert len(xai["frames"]) == 16
    assert xai["spatial_wording"] == "regions contributing to the model prediction"
    assert xai["temporal_wording"] == "frames with higher relative contribution"
    assert [frame["frame_index"] for frame in xai["frames"]] == [index * 5 for index in range(16)]
    dumped = str(body["video_analysis"])
    assert "proof of manipulation" not in dumped


def test_video_xai_keeps_fallback_frames(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    evidence_id = _upload_video(client, tmp_path, "fakes.avi")
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(prediction="FAKE", p_fake=0.81, fallback_frames=2, with_xai=True),
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    video = body["video_analysis"]
    assert video["prediction"] == "FAKE"
    assert video["fallback_frames"] == 2
    assert video["face_crop_frames"] == 14
    assert len(video["xai"]["frames"]) == 16
    assert sum(1 for frame in video["frames"] if frame["used_full_frame_fallback"]) == 2


def test_xai_failure_does_not_fail_video_analysis(app, client, tmp_path: Path) -> None:
    from tests.ffpp_video_fakes import lstm_result

    evidence_id = _upload_video(client, tmp_path, "failxai.avi")
    failed = lstm_result(with_xai=False)
    failed["xai_error"] = "XaiError"
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=failed,
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    assert body["prediction"] == "REAL"
    assert body["video_analysis"]["xai_available"] is False
    assert body["video_analysis"]["p_fake"] == pytest.approx(0.09)
