"""Authentication API tests (Flask-Login)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login  # noqa: E402


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["SECRET_KEY"] = "test-secret-key"
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_register_and_me(client) -> None:
    user = register_and_login(client)
    assert user["username"] == "investigator1"
    assert user["role"] == "INVESTIGATOR"
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.get_json()["data"]["email"] == "investigator1@maya.test"


def test_duplicate_registration(client) -> None:
    register_and_login(client, username="dupuser")
    again = client.post(
        "/api/auth/register",
        json={
            "email": "dupuser@maya.test",
            "username": "dupuser",
            "password": "Abcd@123",
        },
    )
    assert again.status_code == 409


def test_invalid_password(client) -> None:
    register_and_login(client, username="badpass")
    client.post("/api/auth/logout")
    bad = client.post(
        "/api/auth/login",
        json={"login": "badpass", "password": "wrong-password"},
    )
    assert bad.status_code == 401


def test_protected_endpoint_requires_auth(client) -> None:
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_logout(client) -> None:
    register_and_login(client, username="logoutuser")
    out = client.post("/api/auth/logout")
    assert out.status_code == 200
    me = client.get("/api/auth/me")
    assert me.status_code == 401


def test_password_not_returned(client) -> None:
    data = register_and_login(client, username="nopw")
    assert "password" not in data
    assert "password_hash" not in data
