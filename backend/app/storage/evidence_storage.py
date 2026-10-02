"""Safe evidence file storage under UPLOAD_DIR."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Collection
from pathlib import Path

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from backend.app.exceptions import InvalidEvidenceError
from backend.app.utils.paths import is_within

logger = logging.getLogger("maya.backend.storage")

# Match inference-supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
ALLOWED_MIME_PREFIXES = ("image/",)
VIDEO_MIME_PREFIXES = ("video/",)
# Extra client MIME values that do not use the video/ prefix
VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/avi",
    "video/x-matroska",
    "video/mkv",
    "application/x-matroska",
}


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower()


def media_type_for_extension(ext: str) -> str:
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise InvalidEvidenceError(
        f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
    )


def _mime_is_compatible(mime: str, media_type: str) -> bool:
    if media_type == "image":
        return any(mime.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES)
    if media_type == "video":
        return any(mime.startswith(prefix) for prefix in VIDEO_MIME_PREFIXES) or mime in VIDEO_MIME_TYPES
    return False


def validate_upload_file(
    file: FileStorage,
    *,
    max_bytes: int,
    allowed_extensions: Collection[str] | None = None,
) -> tuple[str, str]:
    # Default remains image-only so existing callers (face reference uploads)
    # do not start accepting videos.
    allowed = IMAGE_EXTENSIONS if allowed_extensions is None else set(allowed_extensions)
    if file is None or not file.filename:
        raise InvalidEvidenceError("No file uploaded")
    original = file.filename
    ext = extension_of(original)
    if ext not in allowed:
        raise InvalidEvidenceError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(allowed)}"
        )
    mime = (file.mimetype or "").lower()
    media_type = media_type_for_extension(ext)
    if mime and not _mime_is_compatible(mime, media_type):
        raise InvalidEvidenceError(f"Unsupported MIME type '{mime}'")
    # Size: stream to temp measure if content_length missing — caller enforces MAX
    if file.content_length is not None and file.content_length > max_bytes:
        raise InvalidEvidenceError("File exceeds maximum upload size")
    display_name = secure_filename(original) or f"evidence{ext}"
    return display_name, mime or "application/octet-stream"


def _file_is_still_image(path: Path) -> bool:
    """Return True when the bytes decode as a still image (e.g. renamed JPEG/PNG)."""

    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _validate_readable_video(path: Path) -> None:
    """Prove the stored file is a decodeable video. Does not extract or save frames."""

    if _file_is_still_image(path):
        raise InvalidEvidenceError("Invalid or unreadable video")

    try:
        import cv2
    except ImportError as exc:
        raise InvalidEvidenceError("Invalid or unreadable video") from exc

    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise InvalidEvidenceError("Invalid or unreadable video")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise InvalidEvidenceError("Invalid or unreadable video")
        if getattr(frame, "size", 0) == 0:
            raise InvalidEvidenceError("Invalid or unreadable video")
    except InvalidEvidenceError:
        raise
    except Exception:
        logger.warning("Video readability check failed", exc_info=True)
        raise InvalidEvidenceError("Invalid or unreadable video")
    finally:
        if capture is not None:
            capture.release()


def store_evidence_file(
    file: FileStorage,
    *,
    upload_root: Path,
    case_id: int,
    max_bytes: int,
) -> tuple[str, str, Path, int, str]:
    """Save upload under uploads/cases/{case_id}/ with a UUID filename.

    Returns:
        (original_display_name, stored_filename, absolute_path, size_bytes, media_type)
    """

    display_name, _mime = validate_upload_file(
        file,
        max_bytes=max_bytes,
        allowed_extensions=ALLOWED_EXTENSIONS,
    )
    ext = extension_of(display_name)
    media_type = media_type_for_extension(ext)
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

    if media_type == "video":
        try:
            _validate_readable_video(dest)
        except Exception:
            dest.unlink(missing_ok=True)
            raise

    logger.info("Stored evidence case=%s file=%s size=%s", case_id, stored, size)
    return display_name, stored, dest, size, media_type
