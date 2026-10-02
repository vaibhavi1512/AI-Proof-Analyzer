"""Phase 7: video forensic PDF uses stored Phase 1–6 results only."""

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

from tests.conftest_product import register_and_login  # noqa: E402
from tests.test_analysis_api import _fake_investigation  # noqa: E402


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = tmp_path
    application.config["REPORT_DIR"] = tmp_path / "reports"
    application.config["REPORT_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["SECRET_KEY"] = "test-secret-key"
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _png(size: tuple[int, int] = (32, 32), color=(30, 90, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _plain_text(flowable) -> list[str]:
    from reportlab.platypus import KeepTogether, LongTable, Paragraph, Table

    bits: list[str] = []
    if isinstance(flowable, Paragraph):
        bits.append(flowable.getPlainText())
    elif isinstance(flowable, (Table, LongTable)):
        for row in getattr(flowable, "_cellvalues", []) or []:
            for cell in row:
                bits.extend(_plain_text(cell))
    elif isinstance(flowable, KeepTogether):
        for child in getattr(flowable, "_content", []) or []:
            bits.extend(_plain_text(child))
    elif isinstance(flowable, list):
        for child in flowable:
            bits.extend(_plain_text(child))
    return bits


def _story_text(story) -> str:
    return "\n".join(_plain_text(story))


def _video_story_text(analysis_id: int) -> str:
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun, AuditLog, User
    from backend.app.services.report_service import _append_video_report, _extract_raw_payload, _styles

    run = db.session.get(AnalysisRun, analysis_id)
    evidence = run.evidence
    case = evidence.case
    generator = db.session.get(User, run.created_by_user_id)
    styles = _styles()
    story: list = []
    audit_events = (
        AuditLog.query.filter(AuditLog.case_id == run.case_id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )
    _append_video_report(
        story,
        styles,
        report_number="RPT-TEST-000001",
        case=case,
        evidence=evidence,
        run=run,
        audit_events=audit_events,
        generator=generator,
        notes=None,
        face_verifications=[],
        raw=_extract_raw_payload(run),
    )
    return _story_text(story)


def _upload_image(client, title: str = "Report case") -> tuple[int, int]:
    case_id = client.post("/api/cases", json={"title": title}).get_json()["data"]["case_id"]
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "still.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.get_json()
    return case_id, resp.get_json()["data"]["evidence_id"]


def _analyze_image(client, evidence_id: int) -> int:
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["data"]["analysis_id"]


def _write_png(path: Path, color=(200, 10, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(path, format="PNG")


def _video_payload(*, temporal=None, include_xai_meta=True) -> dict:
    frames = [
        {
            "frame_number": 0,
            "timestamp_seconds": 0.0,
            "frame_path": "cases/1/frames/1/frame_000000.jpg",
            "prediction": "REAL",
            "confidence": 91.0,
            "real_probability": 0.91,
            "fake_probability": 0.09,
        },
        {
            "frame_number": 30,
            "timestamp_seconds": 1.0,
            "frame_path": "cases/1/frames/1/frame_000030.jpg",
            "prediction": "FAKE",
            "confidence": 88.0,
            "real_probability": 0.12,
            "fake_probability": 0.88,
        },
        {
            "frame_number": 90,
            "timestamp_seconds": 3.0,
            "frame_path": "cases/1/frames/1/frame_000090.jpg",
            "prediction": "REAL",
            "confidence": 97.0,
            "real_probability": 0.97,
            "fake_probability": 0.03,
        },
    ]
    payload = {
        "media_type": "video",
        "prediction": "REAL",
        "confidence": 92.8,
        "real_probability": 0.928,
        "fake_probability": 0.072,
        "confidence_level": "High",
        "threshold_decision": "REAL (< threshold)",
        "aggregation": {
            "method": "mean_frame_probability",
            "total_frames": 3,
            "total_frames_analyzed": 3,
            "real_frame_count": 2,
            "fake_frame_count": 1,
            "fake_frame_percentage": 33.3333,
            "threshold": 0.5,
        },
        "frames": frames,
        "suspicious_frames": [frames[1]],
        "metadata": {
            "duration_seconds": 4.0,
            "fps": 30.0,
            "frame_count": 120,
            "width": 64,
            "height": 48,
            "extracted_frame_count": 3,
            "sampling_rate_fps": 1.0,
            "sample_interval_frames": 30,
        },
        "model_name": "efficientnet_b0",
        "model_version": "processed_final",
    }
    if temporal is not False:
        payload["temporal_analysis"] = temporal if temporal is not None else {
            "method": "haar_iou_centroid_tracking",
            "detector": "opencv_haar",
            "face_detection_rate": 0.66,
            "frames_with_face": 2,
            "track_count": 1,
            "primary_track": {
                "track_id": 1,
                "coverage": 0.66,
                "frames_tracked": 2,
                "start_timestamp": 0.0,
                "end_timestamp": 3.0,
            },
            "tracking_gaps": [
                {
                    "start_timestamp": 1.0,
                    "end_timestamp": 2.0,
                    "missing_frame_count": 1,
                }
            ],
            "average_centroid_movement": 4.2,
            "average_bbox_area_change": 0.11,
            "sampling_note": "Association is on ~1 FPS sampled frames, not every video frame.",
        }
    if include_xai_meta:
        payload["explanation"] = {"explainer": "gradcam", "frame_number": 90}
    return payload


def _write_jpeg(path: Path, color=(40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), color=color).save(path, format="JPEG")


def _install_video_analysis(
    app,
    analysis_id: int,
    *,
    payload: dict,
    xai_frames: list[int] | None = None,
    missing_jpegs: set[int] | None = None,
) -> None:
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun, Evidence

    run = db.session.get(AnalysisRun, analysis_id)
    evidence = db.session.get(Evidence, run.evidence_id)
    evidence.media_type = "video"
    evidence.original_filename = "clip.mp4"
    evidence.mime_type = "video/mp4"
    upload = Path(app.config["UPLOAD_DIR"])
    missing = set(missing_jpegs or [])
    frames = payload.get("frames") if isinstance(payload.get("frames"), list) else []
    for item in frames:
        if not isinstance(item, dict) or item.get("frame_number") is None:
            continue
        number = int(item["frame_number"])
        rel = (
            Path("cases")
            / str(evidence.case_id)
            / "frames"
            / str(evidence.id)
            / f"frame_{number:06d}.jpg"
        )
        item["frame_path"] = rel.as_posix()
        if number not in missing:
            _write_jpeg(upload / rel)
    run.raw_result_json = json.dumps(payload)
    run.prediction = payload["prediction"]
    run.confidence = payload["confidence"]
    run.model_name = payload.get("model_name")
    run.investigation_id = run.investigation_id or "INV-VIDEO-REPORT"
    artifact_root = Path(app.config["ROOT_DIR"]) / "artifacts" / "investigations" / run.investigation_id
    run.artifact_dir = str(artifact_root)
    if xai_frames:
        for number in xai_frames:
            folder = artifact_root / "xai" / "gradcam" / f"frame_{number:06d}"
            _write_png(folder / "original.png", (20, 20, 20))
            _write_png(folder / "heatmap.png", (255, 0, 0) if number == xai_frames[0] else (0, 0, 255))
            _write_png(folder / "overlay.png", (0, 255, 0) if number == xai_frames[0] else (255, 255, 0))
        primary = artifact_root / "xai" / "gradcam" / f"frame_{xai_frames[0]:06d}"
        run.heatmap_path = str(primary / "heatmap.png")
        run.overlay_path = str(primary / "overlay.png")
        run.explainer_name = "gradcam"
        run.generate_explanation = True
    db.session.commit()


def _generate(client, analysis_id: int):
    resp = client.post(f"/api/analysis/{analysis_id}/report", json={})
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["data"]


def _download_text(client, report_id: int) -> str:
    resp = client.get(f"/api/reports/{report_id}/download")
    assert resp.status_code == 200
    data = resp.get_data()
    assert data.startswith(b"%PDF")
    return data.decode("latin-1", errors="ignore")


def test_image_report_omits_video_sections(client):
    register_and_login(client, username="imgreport")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    with (
        patch("backend.app.services.report_service._append_video_report") as video_spy,
        patch("backend.app.services.report_service._append_video_frame_gallery") as gallery_spy,
    ):
        report = _generate(client, analysis_id)
    video_spy.assert_not_called()
    gallery_spy.assert_not_called()
    text = _download_text(client, report["report_id"])
    assert text.startswith("%PDF")
    assert report["size_bytes"] > 0


def test_video_report_includes_stored_analysis_fields(app, client):
    register_and_login(client, username="vidreport")
    _, evidence_id = _upload_image(client, title="Video report")
    analysis_id = _analyze_image(client, evidence_id)
    payload = _video_payload()
    _install_video_analysis(app, analysis_id, payload=payload, xai_frames=[90])
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun, Evidence

    run = db.session.get(AnalysisRun, analysis_id)
    evidence = db.session.get(Evidence, evidence_id)
    pred = run.prediction
    conf = run.confidence
    raw = run.raw_result_json
    sha = evidence.sha256_hash

    report = _generate(client, analysis_id)
    pdf = _download_text(client, report["report_id"])
    assert pdf.startswith("%PDF")
    text = _video_story_text(analysis_id)

    assert "VIDEO" in text
    assert "Video Metadata" in text
    assert "mean_frame_probability" in text
    assert "video-level aggregation of frame-level evidence" in text
    assert "dedicated temporal deepfake classifier" in text
    assert "Frame Analysis" in text
    assert "Temporal Face Analysis" in text
    assert "opencv_haar" in text
    assert "Selected XAI frame 90" in text
    assert "Video Frame Gallery" in text
    assert "Frame 0" in text and "Frame 30" in text and "Frame 90" in text
    assert "not the MP4" in text
    assert "not independent proof of manipulation" in text
    assert sha[:16] in text
    assert "Investigation Audit Timeline" in text
    assert str(analysis_id) in text
    assert "0.9280" in text or "92.80%" in text

    run_after = db.session.get(AnalysisRun, analysis_id)
    assert run_after.prediction == pred
    assert run_after.confidence == conf
    assert run_after.raw_result_json == raw


def test_video_report_survives_missing_optional_xai_and_failed_temporal(app, client):
    register_and_login(client, username="vidoptional")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    payload = _video_payload(
        temporal={"method": "haar_iou_centroid_tracking", "error": "temporal_analysis_failed"},
        include_xai_meta=False,
    )
    _install_video_analysis(app, analysis_id, payload=payload, xai_frames=None)
    report = _generate(client, analysis_id)
    assert _download_text(client, report["report_id"]).startswith("%PDF")
    text = _video_story_text(analysis_id)
    assert "temporal_analysis_failed" in text
    assert "No Grad-CAM visualization artifacts" in text
    assert report["size_bytes"] > 0


def test_video_report_includes_multiple_xai_frames(app, client):
    register_and_login(client, username="vidmultixai")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    payload = _video_payload()
    _install_video_analysis(app, analysis_id, payload=payload, xai_frames=[30, 90])
    report = _generate(client, analysis_id)
    assert _download_text(client, report["report_id"]).startswith("%PDF")
    text = _video_story_text(analysis_id)
    assert "Selected XAI frame 30" in text
    assert "Selected XAI frame 90" in text


def test_video_report_omits_temporal_section_error_when_absent(app, client):
    register_and_login(client, username="vidnotemp")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    payload = _video_payload(temporal=False)
    _install_video_analysis(app, analysis_id, payload=payload, xai_frames=None)
    text = _video_story_text(analysis_id)
    assert "Temporal face analysis was not recorded" in text
    assert "temporal_analysis_failed" not in text
    assert _download_text(client, _generate(client, analysis_id)["report_id"]).startswith("%PDF")


def test_video_frame_gallery_includes_all_stored_frames(app, client):
    register_and_login(client, username="vidgallery")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    payload = _video_payload()
    extra = [
        {
            "frame_number": 60,
            "timestamp_seconds": 2.0,
            "frame_path": "",
            "prediction": "REAL",
            "confidence": 93.5,
            "real_probability": 0.935,
            "fake_probability": 0.065,
        },
        {
            "frame_number": 120,
            "timestamp_seconds": 4.0,
            "frame_path": "",
            "prediction": "FAKE",
            "confidence": 70.2,
            "real_probability": 0.30,
            "fake_probability": 0.70,
        },
    ]
    payload["frames"] = [
        payload["frames"][0],
        payload["frames"][1],
        extra[0],
        payload["frames"][2],
        extra[1],
    ]
    payload["aggregation"]["total_frames"] = 5
    payload["aggregation"]["total_frames_analyzed"] = 5
    _install_video_analysis(app, analysis_id, payload=payload, xai_frames=[90])
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun

    run = db.session.get(AnalysisRun, analysis_id)
    pred, conf, raw = run.prediction, run.confidence, run.raw_result_json
    with (
        patch("backend.app.services.analysis_service.run_inference") as infer,
        patch("backend.app.services.analysis_service.run_explanation") as explain,
    ):
        report = _generate(client, analysis_id)
    infer.assert_not_called()
    explain.assert_not_called()
    pdf_bytes = client.get(f"/api/reports/{report['report_id']}/download").get_data()
    assert pdf_bytes.startswith(b"%PDF")
    assert report["size_bytes"] < 1_000_000
    text = _video_story_text(analysis_id)
    assert "Video Frame Gallery" in text
    for number, stamp, label, conf_s in (
        (0, "0.00 s", "REAL", "91.0%"),
        (30, "1.00 s", "FAKE", "88.0%"),
        (60, "2.00 s", "REAL", "93.5%"),
        (90, "3.00 s", "REAL", "97.0%"),
        (120, "4.00 s", "FAKE", "70.2%"),
    ):
        assert f"Frame {number}" in text
        assert stamp in text
        assert label in text
        assert conf_s in text
    assert text.find("Frame 0") < text.find("Frame 30") < text.find("Frame 60")
    assert text.find("Frame 60") < text.find("Frame 90") < text.find("Frame 120")
    assert "Selected XAI frame 90" in text
    run_after = db.session.get(AnalysisRun, analysis_id)
    assert run_after.prediction == pred
    assert run_after.confidence == conf
    assert run_after.raw_result_json == raw


def test_video_frame_gallery_missing_jpeg_does_not_fail(app, client):
    register_and_login(client, username="vidmissingjpg")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    payload = _video_payload()
    _install_video_analysis(
        app, analysis_id, payload=payload, xai_frames=None, missing_jpegs={30}
    )
    report = _generate(client, analysis_id)
    assert report["size_bytes"] > 0
    text = _video_story_text(analysis_id)
    assert "Video Frame Gallery" in text
    assert "Frame 30" in text
    assert "Frame image unavailable" in text
    assert "Frame 0" in text
    assert "Frame 90" in text


def _flow_images(flowable, found=None):
    from reportlab.platypus import Image as RLImage, KeepTogether

    if found is None:
        found = []
    if isinstance(flowable, RLImage):
        found.append(str(getattr(flowable, "filename", "") or ""))
    elif isinstance(flowable, KeepTogether):
        for child in getattr(flowable, "_content", []) or []:
            _flow_images(child, found)
    elif isinstance(flowable, (list, tuple)):
        for child in flowable:
            _flow_images(child, found)
    return found


def _capture_story(client, analysis_id: int):
    from reportlab.platypus import SimpleDocTemplate

    captured: dict = {}
    original = SimpleDocTemplate.build

    def wrapped(self, story, **kwargs):
        captured["text"] = _story_text(story)
        captured["images"] = list(_flow_images(story))
        return original(self, story, **kwargs)

    with patch.object(SimpleDocTemplate, "build", wrapped):
        report = _generate(client, analysis_id)
    return report, captured["text"], captured["images"]


def _lstm_video(prediction: str, p_fake: float, *, xai: bool, fallback: int = 0, frames: int = 16) -> dict:
    rows = []
    for index in range(frames):
        is_fallback = index < fallback
        rows.append(
            {
                "frame_index": index * 10,
                "timestamp_seconds": index * 0.4,
                "used_face_crop": not is_fallback,
                "used_full_frame_fallback": is_fallback,
                "temporal_importance": (index + 1) / (frames * (frames + 1) / 2),
            }
        )
    return {
        "prediction": prediction,
        "p_fake": p_fake,
        "threshold": 0.5,
        "frames_analyzed": frames,
        "face_crop_frames": frames - fallback,
        "fallback_frames": fallback,
        "xai_available": xai,
        "model_name": "ffpp_video_lstm_v2",
        "model_version": "v2-16frame",
        "xai": {
            "spatial_wording": "regions contributing to the model prediction",
            "temporal_wording": "frames with higher relative contribution",
            "frames": rows,
        } if xai else None,
    }


def _install_lstm_analysis(app, analysis_id: int, video: dict, *, investigation_id: str, names: list[str], files: dict[str, tuple]) -> None:
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun, Evidence

    run = db.session.get(AnalysisRun, analysis_id)
    evidence = db.session.get(Evidence, run.evidence_id)
    evidence.media_type = "video"
    evidence.original_filename = "clip.mp4"
    evidence.mime_type = "video/mp4"
    run.prediction = video["prediction"]
    run.confidence = None
    run.heatmap_path = None
    run.overlay_path = None
    run.model_name = video["model_name"]
    run.model_version = video["model_version"]
    run.investigation_id = investigation_id
    artifact_root = Path(app.config["ROOT_DIR"]) / "artifacts" / "investigations" / investigation_id
    run.artifact_dir = str(artifact_root)
    for name, color in files.items():
        _write_png(artifact_root / "xai" / "video" / name, color)
    run.raw_result_json = json.dumps(
        {
            "media_type": "video",
            "prediction": video["prediction"],
            "fake_probability": video["p_fake"],
            "real_probability": 1.0 - video["p_fake"],
            "video_analysis": video,
            "xai_artifact_names": names,
            "model_name": video["model_name"],
            "model_version": video["model_version"],
        }
    )
    db.session.commit()


def test_image_report_keeps_model_confidence_and_heatmap(app, client, tmp_path: Path):
    register_and_login(client, username="imgkeep")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun

    heat = tmp_path / "heatmap.png"
    overlay = tmp_path / "overlay.png"
    _write_png(heat, (255, 0, 0))
    _write_png(overlay, (0, 255, 0))
    run = db.session.get(AnalysisRun, analysis_id)
    run.heatmap_path = str(heat)
    run.overlay_path = str(overlay)
    run.explainer_name = "gradcam"
    db.session.commit()
    report, text, paths = _capture_story(client, analysis_id)
    assert "Model Confidence" in text
    assert "68.10%" in text
    assert "Confidence Score" not in text
    assert "Full-frame fallback frames" not in text
    paths = " ".join(paths)
    assert heat.name in paths
    assert overlay.name in paths
    pdf = client.get(f"/api/reports/{report['report_id']}/download")
    assert pdf.status_code == 200
    assert pdf.mimetype == "application/pdf"
    assert pdf.data.startswith(b"%PDF")


def test_lstm_video_report_uses_confidence_score_and_artifacts(app, client):
    register_and_login(client, username="lstmreport")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    p_fake = 0.62
    video = _lstm_video("FAKE", p_fake, xai=True, fallback=1)
    names = ["temporal_contribution.png", "gradcam_contact_sheet.png", "gradcam_frame_0.png"]
    _install_lstm_analysis(
        app,
        analysis_id,
        video,
        investigation_id="INV-LSTM-A",
        names=names,
        files={name: (10, 20, 30) for name in names},
    )
    report, text, images = _capture_story(client, analysis_id)
    assert "FAKE" in text
    assert f"{p_fake * 100:.2f}%" in text
    assert "Confidence Score" in text
    assert "ffpp_video_lstm_v2" in text
    assert "v2-16frame" in text
    assert "Frames analyzed" in text and "16" in text
    assert "Face-crop frames" in text and "15" in text
    assert "Full-frame fallback frames" in text
    assert "not treated as FAKE" in text
    assert "Face missing" not in text
    assert "Relative temporal contribution" in text
    assert "Regions contributing to the model prediction" in text
    assert "Frames are listed in sampled order." not in text
    paths = " ".join(images)
    assert "gradcam_contact_sheet.png" in paths
    assert "temporal_contribution.png" in paths
    pdf = client.get(f"/api/reports/{report['report_id']}/download")
    assert pdf.mimetype == "application/pdf"
    assert pdf.data.startswith(b"%PDF")
    assert pdf.headers["Content-Type"].startswith("application/pdf")


def test_lstm_video_report_real_confidence_and_missing_xai(app, client):
    register_and_login(client, username="lstmreal")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    p_fake = 0.25
    video = _lstm_video("REAL", p_fake, xai=True, fallback=0)
    _install_lstm_analysis(
        app,
        analysis_id,
        video,
        investigation_id="INV-LSTM-REAL",
        names=["temporal_contribution.png"],
        files={},
    )
    _, text, images = _capture_story(client, analysis_id)
    assert "REAL" in text
    assert f"{(1.0 - p_fake) * 100:.2f}%" in text
    assert "Relative temporal contribution" in text
    assert "gradcam_contact_sheet.png" not in " ".join(images)
    assert "Visual explanation was unavailable" in text

    missing = _lstm_video("REAL", p_fake, xai=False)
    _install_lstm_analysis(
        app,
        analysis_id,
        missing,
        investigation_id="INV-LSTM-NOXAI",
        names=[],
        files={"gradcam_contact_sheet.png": (1, 2, 3)},
    )
    report, text, images = _capture_story(client, analysis_id)
    assert "Explainable AI was not available" in text
    assert images == []
    assert report["size_bytes"] > 0
    assert f"{(1.0 - p_fake) * 100:.2f}%" in text


def test_lstm_report_does_not_reuse_a_previous_analysis_artifact(app, client):
    register_and_login(client, username="lstmstale")
    _, evidence_id = _upload_image(client)
    analysis_id = _analyze_image(client, evidence_id)
    first = _lstm_video("FAKE", 0.8, xai=True)
    _install_lstm_analysis(
        app,
        analysis_id,
        first,
        investigation_id="INV-OLD-XAI",
        names=["gradcam_contact_sheet.png"],
        files={"gradcam_contact_sheet.png": (9, 9, 9)},
    )
    _capture_story(client, analysis_id)
    second = _lstm_video("REAL", 0.2, xai=False)
    _install_lstm_analysis(
        app,
        analysis_id,
        second,
        investigation_id="INV-NEW-RESULT",
        names=[],
        files={},
    )
    _, text, images = _capture_story(client, analysis_id)
    paths = " ".join(images)
    assert "REAL" in text
    assert f"{(1.0 - 0.2) * 100:.2f}%" in text
    assert "INV-OLD-XAI" not in paths
    assert "gradcam_contact_sheet.png" not in paths
    assert "Explainable AI was not available" in text


def test_report_audit_timeline_includes_only_the_current_case(client) -> None:
    register_and_login(client, username="casetimeline")
    _, evidence_id = _upload_image(client, title="CURRENT-CASE-TIMELINE")
    analysis_id = _analyze_image(client, evidence_id)
    other = client.post("/api/cases", json={"title": "OTHER-CASE-TIMELINE"})
    assert other.status_code == 201, other.get_json()
    other_id = other.get_json()["data"]["case_id"]
    uploaded = client.post(
        f"/api/evidence/cases/{other_id}",
        data={"file": (io.BytesIO(_png()), "other-case.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201, uploaded.get_json()

    _, text, _images = _capture_story(client, analysis_id)
    timeline = text.split("Investigation Audit Timeline", 1)[1]
    assert "CURRENT-CASE-TIMELINE" in timeline
    assert "OTHER-CASE-TIMELINE" not in timeline
    assert "USER_LOGIN" not in timeline
    assert "USER_REGISTERED" not in timeline
    assert "this case only" in timeline
