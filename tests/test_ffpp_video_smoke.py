"""Smoke the repository video checkpoints. FF++ videos are test input only."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login

FFPP = ROOT.parent / "dataset_vid" / "FF++"
VISUAL = ROOT / "artifacts" / "checkpoints" / "video" / "visual_model.pt"
LSTM = ROOT / "artifacts" / "checkpoints" / "video" / "lstm.pt"
VIDEOS = {
    "original_033": FFPP / "original_sequences" / "youtube" / "c23" / "videos" / "033.mp4",
    "deepfake_033_097": FFPP / "manipulated_sequences" / "Deepfakes" / "c23" / "videos" / "033_097.mp4",
    "noface_128": FFPP / "original_sequences" / "youtube" / "c23" / "videos" / "128.mp4",
}

pytestmark = pytest.mark.skipif(
    not VISUAL.is_file() or not LSTM.is_file() or any(not path.is_file() for path in VIDEOS.values()),
    reason="Repository video checkpoints or FF++ smoke videos are not present",
)


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = tmp_path
    application.config["SECRET_KEY"] = "test-secret-key"
    application.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    application.config["VIDEO_VISUAL_CHECKPOINT"] = str(VISUAL)
    application.config["VIDEO_LSTM_CHECKPOINT"] = str(LSTM)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.mark.parametrize("name", list(VIDEOS))
def test_smoke_upload_and_analyze(client, name: str) -> None:
    register_and_login(client, username=f"smoke_{name}"[:30])
    case_id = client.post("/api/cases", json={"title": name}).get_json()["data"]["case_id"]
    source = VIDEOS[name]
    before = source.read_bytes()
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(before), source.name, "video/mp4")},
        content_type="multipart/form-data",
    )
    assert up.status_code == 201, up.get_json()
    uploaded = up.get_json()["data"]
    explain = name == "noface_128"
    resp = client.post(
        f"/api/evidence/{uploaded['evidence_id']}/analyze",
        json={"generate_explanation": explain, "verify_before_analyze": True},
    )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    video = body["video_analysis"]
    assert video["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= float(video["p_fake"]) <= 1.0
    assert video["threshold"] == pytest.approx(0.5)
    assert video["frames_analyzed"] == 16
    assert video["face_crop_frames"] + video["fallback_frames"] == 16
    assert [frame["frame_index"] for frame in video["frames"]] == sorted(
        frame["frame_index"] for frame in video["frames"]
    )
    assert len({frame["frame_index"] for frame in video["frames"]}) == 16
    if name == "noface_128":
        assert video["fallback_frames"] >= 1
        assert video["xai_available"] is True
        assert len(video["xai"]["frames"]) == 16
        assert video["xai"]["spatial_wording"] == "regions contributing to the model prediction"
        assert video["xai"]["temporal_wording"] == "frames with higher relative contribution"
        assert "proof of manipulation" not in str(video)
    listed = client.get(f"/api/evidence/{uploaded['evidence_id']}").get_json()["data"]
    assert listed["sha256"] == uploaded["sha256"]
    assert source.read_bytes() == before
    from ai.ffpp_video.analyze import get_video_runtime

    runtime = get_video_runtime(VISUAL, LSTM)
    assert Path(runtime.visual_checkpoint).resolve() == VISUAL.resolve()
    assert Path(runtime.lstm_checkpoint).resolve() == LSTM.resolve()
    assert "dataset_vid" not in runtime.visual_checkpoint
    assert "dataset_vid" not in runtime.lstm_checkpoint
