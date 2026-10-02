"""Evidence upload and integrity API tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
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


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _create_case(client) -> int:
    register_and_login(client, username="evduser")
    resp = client.post("/api/cases", json={"title": "Evidence Case"})
    return resp.get_json()["data"]["case_id"]


def test_upload_valid_evidence_and_sha256(client) -> None:
    case_id = _create_case(client)
    data = {
        "file": (io.BytesIO(_png()), "sample.png", "image/png"),
    }
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data=data,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    assert body["sha256"]
    assert len(body["sha256"]) == 64
    assert body["status"] == "UPLOADED"
    assert body["original_filename"] == "sample.png"
    assert body["media_type"] == "image"
    assert ".." not in body["stored_filename"]

    listed = client.get(f"/api/evidence/cases/{case_id}")
    assert len(listed.get_json()["data"]) == 1

    verify = client.post(f"/api/evidence/{body['evidence_id']}/verify-integrity")
    assert verify.status_code == 200
    assert verify.get_json()["data"]["integrity_status"] == "VALID"


def test_reject_unsupported_file(client) -> None:
    case_id = _create_case(client)
    data = {"file": (io.BytesIO(b"not-an-image"), "notes.txt", "text/plain")}
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data=data,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_unauthorized_evidence_access(client) -> None:
    case_id = _create_case(client)
    data = {"file": (io.BytesIO(_png()), "a.png", "image/png")}
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data=data,
        content_type="multipart/form-data",
    )
    evidence_id = up.get_json()["data"]["evidence_id"]
    client.post("/api/auth/logout")
    register_and_login(client, username="other", email="other@maya.test")
    denied = client.get(f"/api/evidence/{evidence_id}")
    assert denied.status_code == 403
