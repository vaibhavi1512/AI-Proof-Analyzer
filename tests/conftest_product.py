"""Shared pytest fixtures for MAYA Phase 3 product API tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = ROOT
    application.config["WTF_CSRF_ENABLED"] = False
    application.config["SECRET_KEY"] = "test-secret-key"

    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(20, 80, 140)).save(buf, format="PNG")
    return buf.getvalue()


TEST_PASSWORD = "Abcd@123"


def register_and_login(client, *, username: str = "investigator1", email: str | None = None):
    email = email or f"{username}@maya.test"
    reg = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "username": username,
            "password": TEST_PASSWORD,
            "full_name": "Test User",
        },
    )
    assert reg.status_code == 201, reg.get_json()
    login = client.post(
        "/api/auth/login",
        json={"login": username, "password": TEST_PASSWORD},
    )
    assert login.status_code == 200, login.get_json()
    return login.get_json()["data"]
