"""Phase 1 video validation tests for evidence upload/storage.

Covers image compatibility plus MP4/AVI/MOV/MKV acceptance, MIME/size/empty
rejection, and rejection of corrupt or renamed non-video files.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage, Headers

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.exceptions import InvalidEvidenceError
from backend.app.storage.evidence_storage import (
    ALLOWED_EXTENSIONS,
    IMAGE_EXTENSIONS,
    store_evidence_file,
    validate_upload_file,
)
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


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color=(40, 50, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _video_bytes(suffix: str, fourcc: str) -> bytes:
    import cv2

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 5.0, (32, 32))
        assert writer.isOpened(), f"OpenCV VideoWriter failed for {suffix} ({fourcc})"
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:] = (0, 128, 255)
        for _ in range(6):
            writer.write(frame)
        writer.release()
        data = Path(path).read_bytes()
        assert data, f"OpenCV produced an empty {suffix} fixture"
        return data
    finally:
        Path(path).unlink(missing_ok=True)


def _file_storage(
    data: bytes,
    filename: str,
    content_type: str,
    *,
    content_length: int | None = None,
) -> FileStorage:
    headers = Headers()
    headers["Content-Type"] = content_type
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return FileStorage(
        stream=io.BytesIO(data),
        filename=filename,
        content_type=content_type,
        headers=headers,
    )


def _create_case(client) -> int:
    register_and_login(client, username="videouser")
    resp = client.post("/api/cases", json={"title": "Video Validation Case"})
    return resp.get_json()["data"]["case_id"]


def _upload(client, case_id: int, data: bytes, filename: str, mime: str):
    return client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(data), filename, mime)},
        content_type="multipart/form-data",
    )


def test_image_upload_still_works_with_media_type_image(client) -> None:
    case_id = _create_case(client)
    resp = _upload(client, case_id, _png(), "still.png", "image/png")
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    assert body["media_type"] == "image"
    assert body["mime_type"] == "image/png"
    assert body["sha256"]
    assert len(body["sha256"]) == 64


@pytest.mark.parametrize(
    ("suffix", "fourcc", "filename", "mime"),
    [
        (".mp4", "mp4v", "clip.mp4", "video/mp4"),
        (".avi", "MJPG", "clip.avi", "video/x-msvideo"),
        (".mov", "mp4v", "clip.mov", "video/quicktime"),
        (".mkv", "mp4v", "clip.mkv", "video/x-matroska"),
    ],
)
def test_valid_video_upload_succeeds(
    client, suffix: str, fourcc: str, filename: str, mime: str
) -> None:
    case_id = _create_case(client)
    payload = _video_bytes(suffix, fourcc)
    resp = _upload(client, case_id, payload, filename, mime)
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    assert body["media_type"] == "video"
    assert body["mime_type"] == mime
    assert body["original_filename"] == filename
    assert body["sha256"]
    assert len(body["sha256"]) == 64
    assert body["file_size"] == len(payload)
    assert body["status"] == "UPLOADED"


def test_unsupported_extension_is_rejected(client) -> None:
    case_id = _create_case(client)
    resp = _upload(client, case_id, b"not-media", "notes.txt", "text/plain")
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "unsupported file type" in body["message"].lower()


def test_invalid_video_mime_type_is_rejected(client) -> None:
    case_id = _create_case(client)
    payload = _video_bytes(".mp4", "mp4v")
    resp = _upload(client, case_id, payload, "clip.mp4", "text/plain")
    assert resp.status_code == 400
    assert "unsupported mime type" in resp.get_json()["message"].lower()


def test_image_mime_on_video_extension_is_rejected(client) -> None:
    case_id = _create_case(client)
    payload = _video_bytes(".mp4", "mp4v")
    resp = _upload(client, case_id, payload, "clip.mp4", "image/png")
    assert resp.status_code == 400
    assert "unsupported mime type" in resp.get_json()["message"].lower()


def test_empty_video_is_rejected(client) -> None:
    case_id = _create_case(client)
    resp = _upload(client, case_id, b"", "empty.mp4", "video/mp4")
    assert resp.status_code == 400
    assert "empty file" in resp.get_json()["message"].lower()


def test_oversized_video_is_rejected_by_storage(tmp_path: Path) -> None:
    payload = _video_bytes(".mp4", "mp4v")
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    stored = FileStorage(
        stream=io.BytesIO(payload),
        filename="big.mp4",
        content_type="video/mp4",
    )
    with pytest.raises(InvalidEvidenceError, match="maximum upload size"):
        store_evidence_file(
            stored,
            upload_root=upload_root,
            case_id=1,
            max_bytes=max(1, len(payload) - 1),
        )
    leftover = list(upload_root.rglob("*"))
    leftover_files = [p for p in leftover if p.is_file()]
    assert leftover_files == []


def test_oversized_declared_content_length_is_rejected() -> None:
    stored = _file_storage(
        _video_bytes(".mp4", "mp4v"),
        "clip.mp4",
        "video/mp4",
        content_length=999999,
    )
    with pytest.raises(InvalidEvidenceError, match="maximum upload size"):
        validate_upload_file(
            stored,
            max_bytes=1024,
            allowed_extensions=ALLOWED_EXTENSIONS,
        )


def test_corrupt_video_is_rejected(client) -> None:
    case_id = _create_case(client)
    resp = _upload(client, case_id, b"not a video file" + b"\x00" * 64, "bad.mp4", "video/mp4")
    assert resp.status_code == 400
    assert "unreadable video" in resp.get_json()["message"].lower()


def test_renamed_image_with_mp4_extension_is_rejected(client) -> None:
    case_id = _create_case(client)
    resp = _upload(client, case_id, _jpeg(), "fake.mp4", "video/mp4")
    assert resp.status_code == 400
    assert "unreadable video" in resp.get_json()["message"].lower()


def test_renamed_png_with_mp4_extension_is_rejected(client, app) -> None:
    case_id = _create_case(client)
    resp = _upload(client, case_id, _png(), "fake.mp4", "video/mp4")
    assert resp.status_code == 400
    leftover = list(Path(app.config["UPLOAD_DIR"]).rglob("*.mp4"))
    assert leftover == []


def test_validate_upload_file_default_remains_image_only() -> None:
    stored = _file_storage(_video_bytes(".mp4", "mp4v"), "clip.mp4", "video/mp4")
    with pytest.raises(InvalidEvidenceError, match="Unsupported file type"):
        validate_upload_file(stored, max_bytes=1024 * 1024)
    assert IMAGE_EXTENSIONS == {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def test_path_traversal_filename_is_sanitized(client) -> None:
    case_id = _create_case(client)
    payload = _video_bytes(".mp4", "mp4v")
    resp = _upload(client, case_id, payload, "../escape.mp4", "video/mp4")
    assert resp.status_code == 201, resp.get_json()
    body = resp.get_json()["data"]
    assert ".." not in body["stored_filename"]
    assert ".." not in body["storage_path"]
    assert body["storage_path"].startswith("cases/")
