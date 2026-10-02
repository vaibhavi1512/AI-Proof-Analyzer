"""Environment-based configuration for MAYA.

Why this module exists:
    Centralizes settings so paths, database URLs, and secrets are not
    scattered across the codebase or hardcoded into routes.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# MAYA/backend/app/config/config.py → parents[3] == repository root
ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
INSTANCE_DIR = BACKEND_DIR / "instance"
LOG_DIR = ROOT_DIR / "logs"
UPLOAD_DIR = ROOT_DIR / "uploads"
REPORT_DIR = ROOT_DIR / "reports"

load_dotenv(ROOT_DIR / ".env")


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean setting. Blank values keep the default."""

    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    """Read a non-negative integer setting. Invalid values keep the default."""

    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def _repo_video_checkpoint(filename: str) -> str:
    """Repository-relative path for a frozen video checkpoint."""

    return str(ROOT_DIR / "artifacts" / "checkpoints" / "video" / filename)


class BaseConfig:
    """Shared configuration for all environments."""

    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(INSTANCE_DIR / 'maya.db').as_posix()}",
    )

    ROOT_DIR: Path = ROOT_DIR
    FRONTEND_DIR: Path = FRONTEND_DIR
    LOG_DIR: Path = LOG_DIR
    UPLOAD_DIR: Path = UPLOAD_DIR
    REPORT_DIR: Path = REPORT_DIR

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
    # Session cookie settings for Flask-Login
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    REMEMBER_COOKIE_HTTPONLY: bool = True
    ALLOW_PUBLIC_REGISTRATION: bool = os.getenv(
        "ALLOW_PUBLIC_REGISTRATION", "true"
    ).lower() in {"1", "true", "yes", "on"}

    # Phase 5 — Face reference verification (no hard-coded model filesystem paths)
    FACE_VERIFICATION_ENGINE: str = os.getenv("FACE_VERIFICATION_ENGINE", "opencv")
    FACE_VERIFICATION_MODEL: str = os.getenv("FACE_VERIFICATION_MODEL", "inception_resnet_v1")
    FACE_VERIFICATION_MODEL_VERSION: str = os.getenv(
        "FACE_VERIFICATION_MODEL_VERSION", "vggface2"
    )
    FACE_VERIFICATION_PRETRAINED: str = os.getenv("FACE_VERIFICATION_PRETRAINED", "vggface2")
    FACE_VERIFICATION_DEVICE: str = os.getenv("FACE_VERIFICATION_DEVICE", "cpu")
    FACE_VERIFICATION_MATCH_THRESHOLD: float = float(
        os.getenv("FACE_VERIFICATION_MATCH_THRESHOLD", "0.70")
    )
    FACE_VERIFICATION_NO_MATCH_THRESHOLD: float = float(
        os.getenv("FACE_VERIFICATION_NO_MATCH_THRESHOLD", "0.50")
    )
    FACE_VERIFICATION_MIN_FACE_SIZE: int = int(
        os.getenv("FACE_VERIFICATION_MIN_FACE_SIZE", "40")
    )
    FACE_VERIFICATION_MIN_SHARPNESS: float = float(
        os.getenv("FACE_VERIFICATION_MIN_SHARPNESS", "25.0")
    )
    FACE_VERIFICATION_CACHE_DIR: str | None = os.getenv("FACE_VERIFICATION_CACHE_DIR") or None

    # Frozen 16-frame video model. Defaults live in this repository.
    # VIDEO_VISUAL_CHECKPOINT and VIDEO_LSTM_CHECKPOINT can still override them.
    VIDEO_VISUAL_CHECKPOINT: str = os.getenv("VIDEO_VISUAL_CHECKPOINT") or _repo_video_checkpoint(
        "visual_model.pt"
    )
    VIDEO_LSTM_CHECKPOINT: str = os.getenv("VIDEO_LSTM_CHECKPOINT") or _repo_video_checkpoint(
        "lstm.pt"
    )
    VIDEO_DECISION_THRESHOLD: float = 0.5
    VIDEO_NUM_FRAMES: int = 16

    # Authentication and analysis abuse limits. Zero disables that limit.
    LOGIN_FAILURE_LIMIT: int = _int_env("LOGIN_FAILURE_LIMIT", 5)
    LOGIN_FAILURE_WINDOW_SECONDS: int = _int_env("LOGIN_FAILURE_WINDOW_SECONDS", 60)
    REGISTER_RATE_LIMIT: int = _int_env("REGISTER_RATE_LIMIT", 10)
    REGISTER_RATE_WINDOW_SECONDS: int = _int_env("REGISTER_RATE_WINDOW_SECONDS", 60)
    ANALYSIS_RATE_LIMIT: int = _int_env("ANALYSIS_RATE_LIMIT", 5)
    VIDEO_ANALYSIS_RATE_LIMIT: int = _int_env("VIDEO_ANALYSIS_RATE_LIMIT", 3)
    ANALYSIS_RATE_WINDOW_SECONDS: int = _int_env("ANALYSIS_RATE_WINDOW_SECONDS", 60)
    # Report email is a separate bucket so a mail outage cannot consume the
    # analysis limits, and analysis traffic cannot be used to spam SMTP.
    REPORT_EMAIL_LIMIT: int = _int_env("REPORT_EMAIL_LIMIT", 5)
    REPORT_EMAIL_WINDOW_SECONDS: int = _int_env("REPORT_EMAIL_WINDOW_SECONDS", 60)

    # Outbound forensic-report mail. Credentials come only from the environment.
    MAIL_ENABLED: bool = _bool_env("MAIL_ENABLED", False)
    MAIL_HOST: str = os.getenv("MAIL_HOST", "")
    MAIL_PORT: int = _int_env("MAIL_PORT", 587)
    MAIL_USERNAME: str = os.getenv("MAIL_USERNAME", "")
    MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD", "")
    MAIL_FROM: str = os.getenv("MAIL_FROM", "")
    MAIL_USE_TLS: bool = _bool_env("MAIL_USE_TLS", True)


class DevelopmentConfig(BaseConfig):
    """Local development settings."""

    DEBUG: bool = True


class TestingConfig(BaseConfig):
    """Isolated in-memory settings for automated tests."""

    TESTING: bool = True
    DEBUG: bool = True
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    WTF_CSRF_ENABLED: bool = False
    # Isolated upload root set by tests via app.config override when needed


class ProductionConfig(BaseConfig):
    """Production defaults (DEBUG off, cookies locked down)."""

    DEBUG: bool = False
    # Cookies must not travel over plaintext HTTP in production. Deployments
    # that terminate TLS elsewhere can opt out with SESSION_COOKIE_SECURE=false.
    SESSION_COOKIE_SECURE: bool = os.getenv(
        "SESSION_COOKIE_SECURE", "true"
    ).lower() in {"1", "true", "yes", "on"}
    REMEMBER_COOKIE_SECURE: bool = SESSION_COOKIE_SECURE


# Placeholder values that must never protect a production session cookie.
INSECURE_SECRET_KEYS = {
    "dev-only-change-me",
    "change-me-via-env",
    "please-change-this-secret-key",
    "changeme",
    "secret",
}


class InsecureConfigurationError(RuntimeError):
    """Raised when production is asked to boot with development defaults."""


def validate_production_config(config: type[BaseConfig]) -> None:
    """Refuse to start production with a placeholder or missing SECRET_KEY.

    A development default that silently carries into production is worse than
    a failed boot: it makes every session cookie forgeable.
    """

    secret = str(getattr(config, "SECRET_KEY", "") or "")
    if secret.strip().lower() in INSECURE_SECRET_KEYS or len(secret) < 16:
        raise InsecureConfigurationError(
            "SECRET_KEY is unset, too short, or still a placeholder. Set a "
            "random value of at least 16 characters (e.g. "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"`) "
            "before starting MAYA in production."
        )


CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> type[BaseConfig]:
    """Resolve a configuration class from FLASK_ENV or an explicit name."""

    env_name = (name or os.getenv("FLASK_ENV", "development")).lower()
    return CONFIG_MAP.get(env_name, DevelopmentConfig)
