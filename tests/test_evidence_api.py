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


def _upload(client, case_id: int, filename: str = "sample.png", data: bytes | None = None) -> dict:
    response = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(data if data is not None else _png()), filename, "image/png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201, response.get_json()
    return response.get_json()["data"]


def _different_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(220, 4, 19)).save(buf, format="PNG")
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


def test_compare_integrity_same_case_match_and_audit(client):
    from backend.app.models.entities import AuditLog
    from backend.app.extensions import db

    case_id = _create_case(client)
    first = _upload(client, case_id, "one.png")
    second = _upload(client, case_id, "two.png")
    response = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], second["evidence_id"]],
        "case_id": 987654,
        "sha256": ["0" * 64, "f" * 64],
        "storage_path": "../../private/file",
    })
    assert response.status_code == 200, response.get_json()
    data = response.get_json()["data"]
    assert data["case_id"] == case_id
    assert data["outcome"] == "MATCH"
    assert [item["sha256"] for item in data["evidence"]] == [first["sha256"], second["sha256"]]
    assert [item["integrity_status"] for item in data["evidence"]] == ["VALID", "VALID"]
    audit = AuditLog.query.filter_by(case_id=case_id, event_type="EVIDENCE_ACCESSED").all()
    assert any('"action": "compare_integrity"' in row.details_json for row in audit)


def test_compare_integrity_mismatch_cross_case_and_duplicate_ids(client):
    case_id = _create_case(client)
    first = _upload(client, case_id, "one.png")
    second = _upload(client, case_id, "different.png", _different_png())
    mismatch = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], second["evidence_id"]]
    })
    assert mismatch.status_code == 200
    assert mismatch.get_json()["data"]["outcome"] == "MISMATCH"
    duplicate = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], first["evidence_id"]]
    })
    assert duplicate.status_code == 400
    other_case = _create_case_for_same_user(client, "Other Case")
    third = _upload(client, other_case, "same.png")
    cross_case = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], third["evidence_id"]]
    })
    assert cross_case.status_code == 400


def _create_case_for_same_user(client, title: str) -> int:
    response = client.post("/api/cases", json={"title": title})
    assert response.status_code == 201
    return response.get_json()["data"]["case_id"]


def test_compare_integrity_missing_malformed_and_unauthorized(client):
    from backend.app.extensions import db
    from backend.app.models.entities import Evidence

    case_id = _create_case(client)
    first = _upload(client, case_id, "one.png")
    second = _upload(client, case_id, "two.png")
    missing = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], 99999999]
    })
    assert missing.status_code == 404
    with client.application.app_context():
        row = db.session.get(Evidence, second["evidence_id"])
        row.sha256_hash = "not-a-sha256"
        db.session.commit()
    unavailable = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], second["evidence_id"]]
    })
    assert unavailable.status_code == 200
    assert unavailable.get_json()["data"]["outcome"] == "UNAVAILABLE"
    client.post("/api/auth/logout")
    register_and_login(client, username="integrityoutsider", email="integrityoutsider@maya.test")
    denied = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], second["evidence_id"]]
    })
    assert denied.status_code == 404


def test_compare_integrity_keeps_ingestion_hash_and_reports_modified_file(client, app):
    from backend.app.extensions import db
    from backend.app.models.entities import Evidence
    from backend.app.services.evidence_service import absolute_evidence_path

    case_id = _create_case(client)
    first = _upload(client, case_id, "one.png")
    second = _upload(client, case_id, "two.png")
    original_hash = first["sha256"]
    with app.app_context():
        row = db.session.get(Evidence, first["evidence_id"])
        path = absolute_evidence_path(row)
        original_bytes = path.read_bytes()
        path.write_bytes(b"changed after ingestion")
    response = client.post("/api/evidence/compare-integrity", json={
        "evidence_ids": [first["evidence_id"], second["evidence_id"]]
    })
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["outcome"] == "MATCH"
    assert data["evidence"][0]["integrity_status"] == "MODIFIED"
    assert data["evidence"][1]["integrity_status"] == "VALID"
    assert data["evidence"][0]["sha256"] == original_hash
    with app.app_context():
        row = db.session.get(Evidence, first["evidence_id"])
        assert row.sha256_hash == original_hash
        assert absolute_evidence_path(row).read_bytes() == b"changed after ingestion"
    verify = client.post(f"/api/evidence/{first['evidence_id']}/verify-integrity")
    assert verify.status_code == 200
    assert verify.get_json()["data"]["integrity_status"] == "MODIFIED"
    assert original_bytes != b"changed after ingestion"
