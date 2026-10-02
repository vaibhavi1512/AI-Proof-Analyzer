"""Upload → analysis → EVIDEX markup for the frozen 16-frame video model.

FF++ mp4 files are optional test inputs. Checkpoints are the copies in this
repository. The test does not train or rewrite those files.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login

VISUAL = ROOT / "artifacts" / "checkpoints" / "video" / "visual_model.pt"
LSTM = ROOT / "artifacts" / "checkpoints" / "video" / "lstm.pt"
FFPP = ROOT.parent / "dataset_vid" / "FF++"
VIDEOS = {
    "real": FFPP / "original_sequences" / "youtube" / "c23" / "videos" / "033.mp4",
    "fake": FFPP / "manipulated_sequences" / "Deepfakes" / "c23" / "videos" / "033_097.mp4",
    "noface": FFPP / "original_sequences" / "youtube" / "c23" / "videos" / "128.mp4",
}
HAS_VIDEOS = all(path.is_file() for path in VIDEOS.values())
HAS_CHECKPOINTS = VISUAL.is_file() and LSTM.is_file()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module", autouse=True)
def checkpoints_stay_byte_for_byte():
    if not HAS_CHECKPOINTS:
        yield
        return
    before = {path.name: _sha256(path) for path in (VISUAL, LSTM)}
    yield
    after = {path.name: _sha256(path) for path in (VISUAL, LSTM)}
    assert after == before


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
    application.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    application.config["VIDEO_VISUAL_CHECKPOINT"] = str(VISUAL)
    application.config["VIDEO_LSTM_CHECKPOINT"] = str(LSTM)
    application.config["MAIL_ENABLED"] = False
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
    Image.new("RGB", (48, 48), color=(20, 40, 60)).save(buf, format="PNG")
    return buf.getvalue()


def _avi(path: Path, frames: int) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 8 % 255, 40, 90)
        writer.write(frame)
    writer.release()


def _case(client, username: str) -> int:
    register_and_login(client, username=username)
    resp = client.post("/api/cases", json={"title": username})
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["data"]["case_id"]


def _upload(client, case_id: int, data: bytes, filename: str, mime: str) -> dict:
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(data), filename, mime)},
        content_type="multipart/form-data",
    )
    return resp


def _analyze(client, evidence_id: int, *, explain: bool) -> dict:
    resp = client.post(
        f"/api/evidence/{evidence_id}/analyze",
        json={"generate_explanation": explain, "verify_before_analyze": True},
    )
    assert resp.status_code == 201, resp.get_json()
    return resp.get_json()["data"]


def _evidex_html(api_run: dict, screen: str = "analysis") -> str:
    payload = dict(api_run)
    payload["_screen"] = screen
    completed = subprocess.run(
        ["node", str(ROOT / "tests" / "e2e_render_video.js")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        cwd=str(ROOT),
    )
    return completed.stdout


def _assert_video_contract(body: dict, *, explain: bool) -> dict:
    video = body["video_analysis"]
    assert video["prediction"] in {"REAL", "FAKE"}
    assert body["prediction"] == video["prediction"]
    assert isinstance(video["p_fake"], float)
    assert 0.0 <= video["p_fake"] <= 1.0
    assert video["frames_analyzed"] == 16
    assert video["face_crop_frames"] + video["fallback_frames"] == 16
    assert video["face_crop_frames"] >= 0
    assert video["fallback_frames"] >= 0
    assert "dataset_vid" not in json.dumps(video)
    assert "visual_model.pt" not in json.dumps(video)
    if explain:
        assert video["xai_available"] is True
        assert len(video["xai"]["frames"]) == 16
        assert [row["frame_index"] for row in video["xai"]["frames"]] == [
            row["frame_index"] for row in video["frames"]
        ]
    return video


def _confidence_text(video: dict) -> str:
    p_fake = float(video["p_fake"])
    score = p_fake if video["prediction"] == "FAKE" else (1.0 - p_fake)
    return f"{score * 100:.2f}%"


def _assert_screen_matches(body: dict, sha: str) -> str:
    payload = dict(body)
    payload["_sha256"] = sha
    html = _evidex_html(payload, "analysis")
    video = body["video_analysis"]
    assert 'data-video-screen="analysis"' in html
    assert "VIDEO ANALYSIS" in html
    assert f">{video['prediction']}<" in html
    assert "Confidence Score" in html
    assert "Model-predicted FAKE probability" not in html
    assert _confidence_text(video) in html
    assert f'data-field="frames">{video["frames_analyzed"]}<' in html
    assert f'data-field="face-crops">{video["face_crop_frames"]}<' in html
    assert f'data-field="fallback">{video["fallback_frames"]}<' in html
    if video.get("model_name"):
        assert video["model_name"] in html
    if video.get("model_version"):
        assert video["model_version"] in html
    assert "MODEL CONFIDENCE" not in html
    assert "Certainty" not in html
    assert "Manipulated region" not in html
    assert "video-temporal-list" not in html
    assert "Regions contributing to the model prediction" not in html
    assert "video_gradcam?frame=" not in html
    assert "video_contact_sheet" not in html
    if video["xai_available"]:
        assert "Relative temporal contribution" in html
        assert "videoTemporalChart" in html
        if video["fallback_frames"] > 0:
            assert "full-frame fallback" in html
            assert "no detectable face" in html
        _assert_xai_and_tampering(payload, video)
    else:
        assert "XAI unavailable" in html
        assert "videoTemporalChart" not in html
    return html


def _assert_xai_and_tampering(payload: dict, video: dict) -> None:
    """XAI Insights and Tampering Map carry the spatial artifacts. Analysis does not."""

    xai = _evidex_html(payload, "xai")
    assert 'data-video-screen="xai"' in xai
    assert f">{video['prediction']}<" in xai
    assert "Confidence Score" in xai
    assert _confidence_text(video) in xai
    assert "Relative temporal contribution" in xai
    assert "videoTemporalChart" in xai
    assert "Regions contributing to the model prediction" in xai
    assert "Spatial explanation summary" in xai
    indices = list(video.get("gradcam_frame_indices") or [])
    if video.get("gradcam_contact_sheet"):
        assert "video_contact_sheet" in xai
    if indices:
        assert f"video_gradcam?frame={indices[0]}" in xai
    else:
        assert "video_gradcam?frame=" not in xai

    tamper = _evidex_html(payload, "attention")
    assert 'data-video-screen="attention"' in tamper
    assert "Tampering Map" in tamper
    assert "videoTemporalChart" not in tamper
    assert "Regions contributing to the model prediction" in tamper
    if video.get("gradcam_contact_sheet"):
        assert "video_contact_sheet" in tamper
    for index in indices:
        assert f"video_gradcam?frame={index}" in tamper
        assert f'data-video-frame="{index}"' in tamper
    fallback_indexes = {
        row["frame_index"]
        for row in (video.get("xai") or {}).get("frames") or video.get("frames") or []
        if row.get("used_full_frame_fallback") is True
    }
    if fallback_indexes.intersection(indices):
        assert "full-frame fallback" in tamper
    elif indices:
        assert "face crop" in tamper


EXPECTED = {
    "real": {"prediction": "REAL", "p_fake": 0.1500, "fallback_min": 0},
    "fake": {"prediction": "FAKE", "p_fake": 0.7099, "fallback_min": 0},
    "noface": {"prediction": "REAL", "p_fake": 0.1040, "fallback_min": 1},
}


@pytest.mark.skipif(not (HAS_CHECKPOINTS and HAS_VIDEOS), reason="Repository checkpoints or FF++ sample videos are absent")
@pytest.mark.parametrize(
    ("key", "explain"),
    [("real", False), ("fake", False), ("noface", True)],
)
def test_upload_analysis_matches_evidex_screen(client, key: str, explain: bool) -> None:
    case_id = _case(client, f"e2e_{key}")
    source = VIDEOS[key]
    uploaded = _upload(client, case_id, source.read_bytes(), source.name, "video/mp4")
    assert uploaded.status_code == 201, uploaded.get_json()
    evidence = uploaded.get_json()["data"]
    assert len(evidence["sha256"]) == 64
    assert evidence["media_type"] == "video"

    body = _analyze(client, evidence["evidence_id"], explain=explain)
    video = _assert_video_contract(body, explain=explain)
    expected = EXPECTED[key]
    assert video["prediction"] == expected["prediction"]
    assert video["p_fake"] == pytest.approx(expected["p_fake"], abs=1e-3)
    assert video["fallback_frames"] >= expected["fallback_min"]
    _assert_screen_matches(body, evidence["sha256"])

    listed = client.get(f"/api/evidence/{evidence['evidence_id']}").get_json()["data"]
    assert listed["sha256"] == evidence["sha256"]
    custody = client.get(f"/api/evidence/{evidence['evidence_id']}/custody").get_json()["data"]
    events = {row["event_type"] for row in custody["events"]}
    assert "EVIDENCE_UPLOADED" in events
    assert "ANALYSIS_COMPLETED" in events

    if key == "noface":
        assert video["fallback_frames"] >= 1
        assert video["prediction"] in {"REAL", "FAKE"}
        report = client.post(f"/api/analysis/{body['analysis_id']}/report", json={})
        assert report.status_code == 201, report.get_json()
        report_id = report.get_json()["data"]["report_id"]
        pdf = client.get(f"/api/reports/{report_id}/download")
        assert pdf.status_code == 200
        assert pdf.data.startswith(b"%PDF")
        assert report.get_json()["data"]["analysis_id"] == body["analysis_id"]
        contact = client.get(f"/api/analysis/{body['analysis_id']}/artifact/video_contact_sheet")
        assert contact.status_code == 200
        assert contact.data.startswith(b"\x89PNG")
        frame_index = video["gradcam_frame_indices"][0]
        frame = client.get(
            f"/api/analysis/{body['analysis_id']}/artifact/video_gradcam?frame={frame_index}"
        )
        assert frame.status_code == 200
        assert frame.data.startswith(b"\x89PNG")
        missing = client.get(f"/api/analysis/{body['analysis_id']}/artifact/video_gradcam?frame=999999")
        assert missing.status_code == 404


def test_image_analysis_does_not_render_video_ui(client) -> None:
    from unittest.mock import patch

    from tests.test_analysis_api import _fake_investigation

    case_id = _case(client, "e2e_image")
    uploaded = _upload(client, case_id, _png(), "still.png", "image/png")
    assert uploaded.status_code == 201, uploaded.get_json()
    evidence_id = uploaded.get_json()["data"]["evidence_id"]
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        resp = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    assert body["prediction"] == "REAL"
    assert body["confidence"] == pytest.approx(68.1)
    assert "video_analysis" not in body
    assert _evidex_html(body) == ""


def test_invalid_video_never_receives_a_prediction(client, tmp_path: Path) -> None:
    case_id = _case(client, "e2e_invalid")
    corrupt = _upload(client, case_id, b"this is not a video", "bad.mp4", "video/mp4")
    assert corrupt.status_code == 400
    assert "prediction" not in (corrupt.get_json().get("data") or {})

    short = tmp_path / "short.avi"
    _avi(short, 4)
    uploaded = _upload(client, case_id, short.read_bytes(), "short.avi", "video/x-msvideo")
    assert uploaded.status_code == 201, uploaded.get_json()
    evidence_id = uploaded.get_json()["data"]["evidence_id"]
    analyzed = client.post(
        f"/api/evidence/{evidence_id}/analyze",
        json={"generate_explanation": False, "verify_before_analyze": False},
    )
    assert analyzed.status_code == 400
    payload = analyzed.get_json()
    assert payload.get("ok") is False
    assert (payload.get("data") or {}).get("prediction") not in {"REAL", "FAKE"}
    runs = client.get(f"/api/evidence/{evidence_id}/analyses").get_json()["data"]
    assert runs == []


def test_missing_checkpoints_do_not_invent_a_verdict(client, app, tmp_path: Path) -> None:
    case_id = _case(client, "e2e_missing_ckpt")
    video = tmp_path / "long.avi"
    _avi(video, 20)
    uploaded = _upload(client, case_id, video.read_bytes(), "long.avi", "video/x-msvideo")
    assert uploaded.status_code == 201, uploaded.get_json()
    evidence_id = uploaded.get_json()["data"]["evidence_id"]
    app.config["VIDEO_VISUAL_CHECKPOINT"] = str(tmp_path / "missing_visual.pt")
    app.config["VIDEO_LSTM_CHECKPOINT"] = str(tmp_path / "missing_lstm.pt")
    analyzed = client.post(
        f"/api/evidence/{evidence_id}/analyze",
        json={"generate_explanation": False, "verify_before_analyze": False},
    )
    assert analyzed.status_code == 500
    payload = analyzed.get_json()
    assert payload.get("ok") is False
    assert (payload.get("data") or {}).get("prediction") not in {"REAL", "FAKE"}
    runs = client.get(f"/api/evidence/{evidence_id}/analyses").get_json()["data"]
    assert all(row.get("prediction") not in {"REAL", "FAKE"} for row in runs)


def test_analysis_exception_does_not_return_a_prediction(client, tmp_path: Path) -> None:
    from unittest.mock import patch

    case_id = _case(client, "e2e_boom")
    video = tmp_path / "long.avi"
    _avi(video, 20)
    uploaded = _upload(client, case_id, video.read_bytes(), "long.avi", "video/x-msvideo")
    evidence_id = uploaded.get_json()["data"]["evidence_id"]
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        side_effect=RuntimeError("backend unavailable"),
    ):
        analyzed = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "verify_before_analyze": False},
        )
    assert analyzed.status_code == 500
    text = analyzed.get_data(as_text=True)
    assert "Model-predicted FAKE probability" not in text


def test_second_result_replaces_the_first_on_the_evidex_screen(client) -> None:
    first = {
        "analysis_id": 10,
        "prediction": "FAKE",
        "confidence": None,
        "analysis_status": "COMPLETED",
        "video_analysis": {
            "prediction": "FAKE",
            "p_fake": 0.7099,
            "frames_analyzed": 16,
            "face_crop_frames": 16,
            "fallback_frames": 0,
            "xai_available": False,
            "frames": [],
        },
    }
    second = {
        "analysis_id": 11,
        "prediction": "REAL",
        "confidence": None,
        "analysis_status": "COMPLETED",
        "video_analysis": {
            "prediction": "REAL",
            "p_fake": 0.15,
            "frames_analyzed": 16,
            "face_crop_frames": 15,
            "fallback_frames": 1,
            "xai_available": False,
            "frames": [],
        },
    }
    first_html = _evidex_html(first)
    second_html = _evidex_html(second)
    assert "70.99%" in first_html and ">FAKE<" in first_html
    assert ">REAL<" in second_html and "85.00%" in second_html
    assert "15.00%" not in second_html
    assert 'data-analysis-id="11"' in second_html
    assert ">FAKE<" not in second_html
    assert "70.99%" not in second_html
    assert "1 sampled frame had no detectable face" in second_html
