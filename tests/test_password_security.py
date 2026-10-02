"""Password strength and authentication/analysis rate limits."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import TEST_PASSWORD, register_and_login  # noqa: E402
from tests.ffpp_video_fakes import lstm_result  # noqa: E402
from tests.test_analysis_api import _fake_investigation  # noqa: E402


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = tmp_path
    application.config["SECRET_KEY"] = "test-secret-key"
    application.config["LOGIN_FAILURE_LIMIT"] = 5
    application.config["LOGIN_FAILURE_WINDOW_SECONDS"] = 60
    application.config["ANALYSIS_RATE_LIMIT"] = 5
    application.config["VIDEO_ANALYSIS_RATE_LIMIT"] = 3
    application.config["ANALYSIS_RATE_WINDOW_SECONDS"] = 60
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _clock(app):
    clock = {"t": 1_000.0}
    app.extensions["rate_limiter"].now = lambda: clock["t"]
    return clock


def _register(client, username: str, password: str):
    return client.post(
        "/api/auth/register",
        json={
            "email": f"{username}@maya.test",
            "username": username,
            "password": password,
            "full_name": "Policy User",
        },
    )


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client) -> int:
    case_id = client.post("/api/cases", json={"title": "Limit"}).get_json()["data"]["case_id"]
    uploaded = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "still.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201, uploaded.get_json()
    return uploaded.get_json()["data"]["evidence_id"]


def test_valid_password_is_accepted(client, app) -> None:
    response = _register(client, "stronguser", TEST_PASSWORD)
    assert response.status_code == 201, response.get_json()
    body = response.get_json()
    assert TEST_PASSWORD not in str(body)
    with app.app_context():
        from backend.app.models import User

        user = User.query.filter_by(username="stronguser").one()
        assert user.password_hash != TEST_PASSWORD
        assert TEST_PASSWORD not in user.password_hash
        assert user.password_hash.startswith(("scrypt:", "pbkdf2:"))


@pytest.mark.parametrize(
    "password",
    ["Ab@1", "abcd@123", "ABCD@123", "Abcdefg@", "Abcdefg1"],
)
def test_weak_passwords_are_rejected(client, password: str) -> None:
    response = _register(client, "weakuser", password)
    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"] == "validation_error"
    assert "uppercase" in body["message"] or "8 characters" in body["message"]
    assert password not in body["message"]


def test_correct_and_incorrect_login(client) -> None:
    register_and_login(client, username="loginuser")
    client.post("/api/auth/logout")
    wrong = client.post(
        "/api/auth/login",
        json={"login": "loginuser", "password": "Wrong@123"},
    )
    assert wrong.status_code == 401
    assert "Wrong@123" not in str(wrong.get_json())
    right = client.post(
        "/api/auth/login",
        json={"login": "loginuser", "password": TEST_PASSWORD},
    )
    assert right.status_code == 200
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.get_json()["data"]["username"] == "loginuser"


def test_failed_login_rate_limit_expires(client, app) -> None:
    register_and_login(client, username="limited")
    client.post("/api/auth/logout")
    clock = _clock(app)
    for _ in range(5):
        failed = client.post(
            "/api/auth/login",
            json={"login": "limited", "password": "Wrong@123"},
        )
        assert failed.status_code == 401
    blocked = client.post(
        "/api/auth/login",
        json={"login": "limited", "password": TEST_PASSWORD},
    )
    assert blocked.status_code == 429
    body = blocked.get_json()
    assert body["error"] == "rate_limited"
    assert body["message"] == "Too many login attempts. Please try again later."
    assert TEST_PASSWORD not in str(body)
    clock["t"] += 61
    restored = client.post(
        "/api/auth/login",
        json={"login": "limited", "password": TEST_PASSWORD},
    )
    assert restored.status_code == 200, restored.get_json()


def test_password_is_not_logged(client, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    _register(client, "quietuser", TEST_PASSWORD)
    client.post(
        "/api/auth/login",
        json={"login": "quietuser", "password": "Wrong@123"},
    )
    assert TEST_PASSWORD not in caplog.text
    assert "Wrong@123" not in caplog.text


def test_image_analysis_rate_limit_and_reads_still_work(client, app) -> None:
    register_and_login(client, username="imagelimit")
    evidence_id = _upload(client)
    clock = _clock(app)
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        first = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
        assert first.status_code == 201, first.get_json()
        analysis_id = first.get_json()["data"]["analysis_id"]
        for _ in range(4):
            again = client.post(
                f"/api/evidence/{evidence_id}/analyze",
                json={"generate_explanation": False, "verify_before_analyze": False},
            )
            assert again.status_code == 201, again.get_json()
        blocked = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert blocked.status_code == 429
    assert blocked.get_json()["message"] == "Too many analysis requests. Please try again later."
    viewed = client.get(f"/api/analysis/{analysis_id}")
    assert viewed.status_code == 200
    clock["t"] += 61
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        restored = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert restored.status_code == 201, restored.get_json()


def test_video_analysis_uses_its_own_limit(client, app) -> None:
    from backend.app.extensions import db
    from backend.app.models import Evidence

    register_and_login(client, username="videolimit")
    evidence_id = _upload(client)
    with app.app_context():
        row = db.session.get(Evidence, evidence_id)
        row.media_type = "video"
        db.session.commit()
    clock = _clock(app)
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(),
    ):
        for _ in range(3):
            ok = client.post(
                f"/api/evidence/{evidence_id}/analyze",
                json={"generate_explanation": False, "verify_before_analyze": False},
            )
            assert ok.status_code == 201, ok.get_json()
            assert ok.get_json()["data"]["video_analysis"]["frames_analyzed"] == 16
        blocked = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert blocked.status_code == 429
    assert blocked.get_json()["error"] == "rate_limited"
    clock["t"] += 61
    with patch(
        "backend.app.services.video_model_service.analyze_video",
        return_value=lstm_result(),
    ):
        restored = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert restored.status_code == 201, restored.get_json()
