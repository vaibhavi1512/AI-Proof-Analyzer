"""Case management API tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def test_case_crud_and_close(client) -> None:
    register_and_login(client, username="caseowner")
    created = client.post(
        "/api/cases",
        json={"title": "Case Alpha", "description": "desc", "priority": "HIGH"},
    )
    assert created.status_code == 201
    case = created.get_json()["data"]
    case_id = case["case_id"]
    assert case["case_number"].startswith("CASE-")
    assert case["status"] == "OPEN"

    listed = client.get("/api/cases")
    assert listed.status_code == 200
    assert len(listed.get_json()["data"]) == 1

    got = client.get(f"/api/cases/{case_id}")
    assert got.status_code == 200
    assert got.get_json()["data"]["title"] == "Case Alpha"

    patched = client.patch(
        f"/api/cases/{case_id}",
        json={"title": "Case Alpha Updated", "status": "IN_PROGRESS"},
    )
    assert patched.status_code == 200
    assert patched.get_json()["data"]["title"] == "Case Alpha Updated"
    assert patched.get_json()["data"]["status"] == "IN_PROGRESS"

    closed = client.post(f"/api/cases/{case_id}/close")
    assert closed.status_code == 200
    assert closed.get_json()["data"]["status"] == "CLOSED"
    assert closed.get_json()["data"]["closed_at"] is not None


def test_unauthorized_case_access(client) -> None:
    register_and_login(client, username="owner_a")
    created = client.post("/api/cases", json={"title": "Private"})
    case_id = created.get_json()["data"]["case_id"]
    client.post("/api/auth/logout")

    register_and_login(client, username="owner_b", email="owner_b@maya.test")
    denied = client.get(f"/api/cases/{case_id}")
    assert denied.status_code == 403
    listed = client.get("/api/cases")
    assert listed.get_json()["data"] == []


def test_delete_case_removes_evidence_and_keeps_audit(client) -> None:
    import io

    from PIL import Image

    from backend.app.models.entities import AuditLog, Case, Evidence

    register_and_login(client, username="deleter")
    case_id = client.post("/api/cases", json={"title": "Disposable"}).get_json()["data"]["case_id"]
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), (10, 20, 30)).save(buf, format="PNG")
    buf.seek(0)
    upload = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (buf, "e.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert upload.status_code == 201
    evidence_id = upload.get_json()["data"]["evidence_id"]
    audit_before = AuditLog.query.count()

    deleted = client.delete(f"/api/cases/{case_id}")
    assert deleted.status_code == 200
    payload = deleted.get_json()["data"]
    assert payload["case_id"] == case_id
    assert payload["deleted_evidence"] == 1

    assert Case.query.count() == 0
    assert Evidence.query.count() == 0
    assert client.get(f"/api/cases/{case_id}").status_code == 404
    assert client.get(f"/api/evidence/{evidence_id}").status_code == 404
    assert client.get("/api/cases").get_json()["data"] == []
    # A second delete must not resurrect or 500.
    assert client.delete(f"/api/cases/{case_id}").status_code == 404

    # The custody trail survives the record it describes, plus a CASE_DELETED row.
    assert AuditLog.query.count() == audit_before + 1
    events = {row.event_type for row in AuditLog.query.all()}
    assert "CASE_DELETED" in events
    assert "EVIDENCE_UPLOADED" in events


def test_delete_case_rejects_non_owner(client) -> None:
    from backend.app.models.entities import Case

    register_and_login(client, username="keeper")
    case_id = client.post("/api/cases", json={"title": "Keep"}).get_json()["data"]["case_id"]
    client.post("/api/auth/logout")

    register_and_login(client, username="intruder", email="intruder@maya.test")
    assert client.delete(f"/api/cases/{case_id}").status_code == 403
    client.post("/api/auth/logout")

    assert client.delete(f"/api/cases/{case_id}").status_code == 401
    assert Case.query.count() == 1


def test_admin_users_endpoint_requires_admin(client) -> None:
    from backend.app.database.init_db import _ensure_default_admin

    assert client.get("/api/admin/users").status_code == 401

    register_and_login(client, username="plain_investigator")
    client.post("/api/cases", json={"title": "Owned"})
    assert client.get("/api/admin/users").status_code == 403
    client.post("/api/auth/logout")

    _ensure_default_admin()
    assert client.post(
        "/api/auth/login", json={"login": "admin", "password": "admin@123"}
    ).status_code == 200
    listed = client.get("/api/admin/users")
    assert listed.status_code == 200
    users = listed.get_json()["data"]
    by_name = {u["username"]: u for u in users}
    assert {"admin", "plain_investigator"} <= set(by_name)
    assert by_name["admin"]["role"] == "ADMIN"
    assert by_name["plain_investigator"]["case_count"] == 1
    assert "password_hash" not in by_name["admin"]
