"""Phase 5 face reference verification — pipeline, API, audit, artefacts, reports."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.face_verification.compare import cosine_similarity, decide_verification
from ai.face_verification.config import FaceVerificationConfig
from ai.face_verification.pipeline import run_face_verification
from ai.face_verification.quality import assess_face_crop
from ai.face_verification.types import (
    BoundingBox,
    DECISION_INCONCLUSIVE,
    DECISION_MATCH,
    DECISION_NO_MATCH,
    DetectedFace,
    REASON_CORRUPTED_IMAGE,
    REASON_LOW_QUALITY_REFERENCE,
    REASON_MULTIPLE_FACES_EVIDENCE,
    REASON_NO_FACE_EVIDENCE,
    REASON_NO_FACE_REFERENCE,
)
from tests.conftest_product import register_and_login
from tests.test_analysis_api import _fake_investigation


class ScriptedEngine:
    name = "scripted"
    model_name = "scripted_embed"
    model_version = "test-1"

    def __init__(self, ref_faces, evd_faces, ref_emb, evd_emb, embed_error: bool = False):
        self.ref_faces = ref_faces
        self.evd_faces = evd_faces
        self.ref_emb = np.asarray(ref_emb, dtype=np.float64)
        self.evd_emb = np.asarray(evd_emb, dtype=np.float64)
        self.embed_error = embed_error
        self._detect_calls = 0
        self._embed_calls = 0

    def detect(self, image):  # noqa: ARG002
        self._detect_calls += 1
        return self.ref_faces if self._detect_calls == 1 else self.evd_faces

    def embed(self, crop):  # noqa: ARG002
        if self.embed_error:
            raise RuntimeError("embed boom")
        self._embed_calls += 1
        return self.ref_emb if self._embed_calls == 1 else self.evd_emb


def _noisy_crop(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, size=(size, size, 3), dtype=np.uint8)


def _face(seed: int, size: int = 96) -> DetectedFace:
    crop = _noisy_crop(seed, size=size)
    quality = assess_face_crop(crop, FaceVerificationConfig(min_sharpness=1.0, min_face_size=20))
    return DetectedFace(bbox=BoundingBox(2, 2, 2 + size, 2 + size), crop=crop, quality=quality)


def _png_bytes(color=(20, 80, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _write_png(path: Path, color=(20, 80, 140)) -> Path:
    Image.new("RGB", (64, 64), color=color).save(path)
    return path


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
    application.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _setup_evidence(client) -> int:
    register_and_login(client, username="faceuser")
    case_id = client.post("/api/cases", json={"title": "Face Case"}).get_json()["data"]["case_id"]
    up = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png_bytes()), "evidence.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert up.status_code == 201, up.get_json()
    return up.get_json()["data"]["evidence_id"]


def _post_verify(client, evidence_id: int, engine: ScriptedEngine, extra_form: dict | None = None, raw=None):
    form = extra_form or {}
    data = {"file": (io.BytesIO(raw if raw is not None else _png_bytes((9, 9, 9))), "ref.png", "image/png")}
    data.update(form)
    with patch(
        "backend.app.services.face_verification_service.get_face_engine",
        return_value=engine,
    ):
        return client.post(
            f"/api/evidence/{evidence_id}/face-verification",
            data=data,
            content_type="multipart/form-data",
        )


def test_threshold_decision_unit() -> None:
    assert decide_verification(0.90, match_threshold=0.70, no_match_threshold=0.50)[0] == DECISION_MATCH
    assert decide_verification(0.20, match_threshold=0.70, no_match_threshold=0.50)[0] == DECISION_NO_MATCH
    assert decide_verification(0.60, match_threshold=0.70, no_match_threshold=0.50)[0] == DECISION_INCONCLUSIVE
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_pipeline_match_no_match_and_counts(tmp_path: Path) -> None:
    ref = _write_png(tmp_path / "r.png")
    evd = _write_png(tmp_path / "e.png", color=(1, 2, 3))
    cfg = FaceVerificationConfig()
    cfg.match_threshold = 0.70
    cfg.no_match_threshold = 0.50

    match = run_face_verification(
        ref,
        evd,
        config=cfg,
        engine=ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [1.0, 0.0]),
    )
    assert match.decision == DECISION_MATCH
    assert match.verification_status == "COMPLETED"
    assert match.similarity_score == pytest.approx(1.0)

    nomatch = run_face_verification(
        ref,
        evd,
        config=cfg,
        engine=ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [0.0, 1.0]),
    )
    assert nomatch.decision == DECISION_NO_MATCH

    none_ref = run_face_verification(
        ref, evd, config=cfg, engine=ScriptedEngine([], [_face(2)], [1.0], [1.0])
    )
    assert none_ref.decision == DECISION_INCONCLUSIVE
    assert none_ref.reason_code == REASON_NO_FACE_REFERENCE

    none_evd = run_face_verification(
        ref, evd, config=cfg, engine=ScriptedEngine([_face(1)], [], [1.0], [1.0])
    )
    assert none_evd.reason_code == REASON_NO_FACE_EVIDENCE

    multi = run_face_verification(
        ref,
        evd,
        config=cfg,
        engine=ScriptedEngine([_face(1)], [_face(2), _face(3)], [1.0], [1.0]),
    )
    assert multi.reason_code == REASON_MULTIPLE_FACES_EVIDENCE
    assert multi.evidence_face_count == 2

    tiny_crop = _noisy_crop(9, size=8)
    tiny = DetectedFace(
        bbox=BoundingBox(0, 0, 8, 8),
        crop=tiny_crop,
        quality=assess_face_crop(tiny_crop, FaceVerificationConfig(min_face_size=40)),
    )
    low_q = run_face_verification(
        ref, evd, config=cfg, engine=ScriptedEngine([tiny], [_face(2)], [1.0], [1.0])
    )
    assert low_q.reason_code == REASON_LOW_QUALITY_REFERENCE


def test_pipeline_corrupted_image(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not-an-image")
    evd = _write_png(tmp_path / "e.png")
    result = run_face_verification(
        bad,
        evd,
        engine=ScriptedEngine([_face(1)], [_face(2)], [1.0], [1.0]),
    )
    assert result.decision == DECISION_INCONCLUSIVE
    assert result.reason_code == REASON_CORRUPTED_IMAGE


def test_api_match_persist_artifacts_audit(client, tmp_path: Path) -> None:
    evidence_id = _setup_evidence(client)
    engine = ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [0.99, 0.01])
    resp = _post_verify(client, evidence_id, engine)
    assert resp.status_code == 201, resp.get_json()
    data = resp.get_json()["data"]
    assert data["decision"] == DECISION_MATCH
    assert data["verification_status"] == "COMPLETED"
    assert data["similarity_score"] is not None
    assert data["distance_score"] is not None
    assert data["artifact_dir"]
    assert str(data["artifact_dir"]).replace("\\", "/").startswith("artifacts/")
    vid = data["verification_id"]

    got = client.get(f"/api/face-verifications/{vid}")
    assert got.status_code == 200
    assert got.get_json()["data"]["decision"] == DECISION_MATCH

    from backend.app.models import FaceVerification

    row = FaceVerification.query.filter_by(id=vid).first()
    assert row is not None
    assert row.similarity_score is not None
    abs_dir = tmp_path / row.artifact_dir
    assert (abs_dir / "comparison_result.json").is_file()
    assert (abs_dir / "verification_report.json").is_file()
    payload = json.loads((abs_dir / "verification_report.json").read_text(encoding="utf-8"))
    assert payload["decision"] == DECISION_MATCH
    assert payload["model_name"]

    audit = client.get("/api/audit").get_json()["data"]
    events = {e["event_type"] for e in audit}
    assert "FACE_VERIFICATION_STARTED" in events
    assert "FACE_VERIFICATION_COMPLETED" in events


def test_api_no_match(client) -> None:
    evidence_id = _setup_evidence(client)
    resp = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [0.0, 1.0]),
    )
    assert resp.get_json()["data"]["decision"] == DECISION_NO_MATCH


def test_api_no_face_reference_and_evidence(client) -> None:
    evidence_id = _setup_evidence(client)
    r1 = _post_verify(client, evidence_id, ScriptedEngine([], [_face(2)], [1.0], [1.0]))
    assert r1.get_json()["data"]["reason_code"] == REASON_NO_FACE_REFERENCE
    r2 = _post_verify(client, evidence_id, ScriptedEngine([_face(1)], [], [1.0], [1.0]))
    assert r2.get_json()["data"]["reason_code"] == REASON_NO_FACE_EVIDENCE


def test_api_multiple_faces(client) -> None:
    evidence_id = _setup_evidence(client)
    resp = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1), _face(4)], [_face(2)], [1.0], [1.0]),
    )
    body = resp.get_json()["data"]
    assert body["decision"] == DECISION_INCONCLUSIVE
    assert body["reference_face_count"] == 2


def test_api_invalid_image(client) -> None:
    evidence_id = _setup_evidence(client)
    resp = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0], [1.0]),
        raw=b"\x00\x01notpng",
    )
    assert resp.status_code == 201
    assert resp.get_json()["data"]["reason_code"] == REASON_CORRUPTED_IMAGE


def test_api_unsupported_file(client) -> None:
    evidence_id = _setup_evidence(client)
    resp = client.post(
        f"/api/evidence/{evidence_id}/face-verification",
        data={"file": (io.BytesIO(b"hello"), "notes.txt", "text/plain")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_api_auth_required(client) -> None:
    resp = client.post(
        "/api/evidence/1/face-verification",
        data={"file": (io.BytesIO(_png_bytes()), "r.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401
    assert client.get("/api/face-verifications/1").status_code == 401


def test_api_unauthorized_evidence_and_case(client) -> None:
    evidence_id = _setup_evidence(client)
    client.post("/api/auth/logout")
    register_and_login(client, username="otherinv", email="other@maya.test")
    resp = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0], [1.0]),
    )
    assert resp.status_code in {403, 404}

    client.post("/api/auth/logout")
    login_owner = client.post(
        "/api/auth/login", json={"login": "faceuser", "password": "securepass1"}
    )
    assert login_owner.status_code == 200, login_owner.get_json()
    created = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [1.0, 0.0]),
    )
    vid = created.get_json()["data"]["verification_id"]
    client.post("/api/auth/logout")
    login_other = client.post(
        "/api/auth/login", json={"login": "otherinv", "password": "securepass1"}
    )
    assert login_other.status_code == 200
    denied = client.get(f"/api/face-verifications/{vid}")
    assert denied.status_code in {403, 404}


def test_api_threshold_behavior(client) -> None:
    evidence_id = _setup_evidence(client)
    mid = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [0.6, 0.8]),
    )
    assert mid.get_json()["data"]["decision"] == DECISION_INCONCLUSIVE
    matchish = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [0.6, 0.8]),
        extra_form={"threshold": "0.55", "no_match_threshold": "0.40"},
    )
    assert matchish.get_json()["data"]["decision"] == DECISION_MATCH
    nomatch = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [0.6, 0.8]),
        extra_form={"threshold": "0.95", "no_match_threshold": "0.70"},
    )
    assert nomatch.get_json()["data"]["decision"] == DECISION_NO_MATCH


def test_report_integration_and_existing_report_path(client) -> None:
    evidence_id = _setup_evidence(client)
    expl = {
        "explainer": "gradcam",
        "heatmap": None,
        "overlay": None,
        "explanation_json": None,
        "prediction": "REAL",
        "confidence": 68.1,
        "target_class": 0,
    }
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ), patch(
        "backend.app.services.analysis_service.run_explanation",
        return_value=expl,
    ):
        analyzed = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": True, "explainer": "gradcam", "verify_before_analyze": False},
        )
    assert analyzed.status_code == 201, analyzed.get_json()
    analysis_id = analyzed.get_json()["data"]["analysis_id"]

    without_fv = client.post(f"/api/analysis/{analysis_id}/report", json={})
    assert without_fv.status_code == 201, without_fv.get_json()

    fv = _post_verify(
        client,
        evidence_id,
        ScriptedEngine([_face(1)], [_face(2)], [1.0, 0.0], [1.0, 0.0]),
        extra_form={"investigation_id": "INV-2026-009999"},
    )
    assert fv.status_code == 201, fv.get_json()

    from backend.app.services import report_service as rs

    with patch.object(
        rs, "_append_face_verification_section", wraps=rs._append_face_verification_section
    ) as spy:
        with_fv = client.post(f"/api/analysis/{analysis_id}/report", json={})
    assert with_fv.status_code == 201, with_fv.get_json()
    spy.assert_called()
    rows = spy.call_args.args[2]
    assert rows
    assert rows[0].decision == DECISION_MATCH
    assert rows[0].verification_status == "COMPLETED"

    from backend.app.models import InvestigationReport

    report = InvestigationReport.query.filter_by(
        id=with_fv.get_json()["data"]["report_id"]
    ).first()
    assert report is not None
    assert report.investigation_id == "INV-2026-009999"


def test_existing_api_regression_cases_still_work(client) -> None:
    register_and_login(client, username="regression")
    listed = client.get("/api/cases")
    assert listed.status_code == 200
    assert listed.get_json()["ok"] is True
    created = client.post("/api/cases", json={"title": "Still works"})
    assert created.status_code == 201
