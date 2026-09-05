"""Safe evidence file storage under UPLOAD_DIR."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from backend.app.exceptions import InvalidEvidenceError
from backend.app.utils.paths import is_within

logger = logging.getLogger("maya.backend.storage")

# Match inference-supported image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_MIME_PREFIXES = ("image/",)


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_upload_file(file: FileStorage, *, max_bytes: int) -> tuple[str, str]:
    if file is None or not file.filename:
        raise InvalidEvidenceError("No file uploaded")
    original = file.filename
    ext = extension_of(original)
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidEvidenceError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    mime = (file.mimetype or "").lower()
    if mime and not any(mime.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise InvalidEvidenceError(f"Unsupported MIME type '{mime}'")
    # Size: stream to temp measure if content_length missing — caller enforces MAX
    if file.content_length is not None and file.content_length > max_bytes:
        raise InvalidEvidenceError("File exceeds maximum upload size")
    display_name = secure_filename(original) or f"evidence{ext}"
    return display_name, mime or "application/octet-stream"


def store_evidence_file(
    file: FileStorage,
    *,
    upload_root: Path,
    case_id: int,
    max_bytes: int,
) -> tuple[str, str, Path, int]:
    """Save upload under uploads/cases/{case_id}/ with a UUID filename.

    Returns:
        (original_display_name, stored_filename, absolute_path, size_bytes)
    """

    display_name, mime = validate_upload_file(file, max_bytes=max_bytes)
    ext = extension_of(display_name)
    stored = f"{uuid.uuid4().hex}{ext}"
    case_dir = Path(upload_root) / "cases" / str(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    dest = case_dir / stored

    # Prevent path escape
    if not is_within(upload_root, dest):
        raise InvalidEvidenceError("Invalid storage path")
    dest = dest.resolve()

    file.save(dest)
    size = dest.stat().st_size
    if size <= 0:
        dest.unlink(missing_ok=True)
        raise InvalidEvidenceError("Empty file rejected")
    if size > max_bytes:
        dest.unlink(missing_ok=True)
        raise InvalidEvidenceError("File exceeds maximum upload size")

    logger.info("Stored evidence case=%s file=%s size=%s", case_id, stored, size)
    return display_name, stored, dest, size
