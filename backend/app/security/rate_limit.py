"""In-process sliding-window limits for authentication and analysis starts.

Limits are read from Flask config on each check. A zero limit disables that
bucket. State lives on the application object, so it is not shared across
processes and it resets when the process restarts.
"""

from __future__ import annotations

import threading
import time

from flask import current_app, request
from flask_login import current_user

from backend.app.exceptions import RateLimitError

LOGIN_MESSAGE = "Too many login attempts. Please try again later."
REGISTER_MESSAGE = "Too many registration attempts. Please try again later."
ANALYSIS_MESSAGE = "Too many analysis requests. Please try again later."
REPORT_EMAIL_MESSAGE = "Too many report emails. Please try again later."


class SlidingWindowLimiter:
    """Count events inside a window without extending the window while blocked."""

    def __init__(self) -> None:
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self.now = time.monotonic

    def _prune(self, key: str, window_seconds: float, now: float) -> list[float]:
        events = [stamp for stamp in self._events.get(key, []) if now - stamp < window_seconds]
        if events:
            self._events[key] = events
        else:
            self._events.pop(key, None)
        return events

    def is_limited(self, key: str, limit: int, window_seconds: float) -> bool:
        if limit <= 0:
            return False
        now = float(self.now())
        with self._lock:
            return len(self._prune(key, window_seconds, now)) >= limit

    def try_consume(self, key: str, limit: int, window_seconds: float) -> bool:
        """Record one event. Return False when the bucket is already full."""

        if limit <= 0:
            return True
        now = float(self.now())
        with self._lock:
            events = self._prune(key, window_seconds, now)
            if len(events) >= limit:
                return False
            events.append(now)
            self._events[key] = events
            return True


def get_limiter() -> SlidingWindowLimiter:
    limiter = current_app.extensions.get("rate_limiter")
    if not isinstance(limiter, SlidingWindowLimiter):
        limiter = SlidingWindowLimiter()
        current_app.extensions["rate_limiter"] = limiter
    return limiter


def client_address() -> str:
    """Use the direct client address. Forwarded headers are not trusted."""

    return request.remote_addr or "unknown"


def _setting(name: str, default: int) -> int:
    try:
        return int(current_app.config.get(name, default))
    except (TypeError, ValueError):
        return default


def enforce_login_allowed() -> None:
    limiter = get_limiter()
    if limiter.is_limited(
        f"login:{client_address()}",
        _setting("LOGIN_FAILURE_LIMIT", 5),
        _setting("LOGIN_FAILURE_WINDOW_SECONDS", 60),
    ):
        raise RateLimitError(LOGIN_MESSAGE)


def record_login_failure() -> None:
    get_limiter().try_consume(
        f"login:{client_address()}",
        _setting("LOGIN_FAILURE_LIMIT", 5),
        _setting("LOGIN_FAILURE_WINDOW_SECONDS", 60),
    )


def enforce_register_rate_limit() -> None:
    allowed = get_limiter().try_consume(
        f"register:{client_address()}",
        _setting("REGISTER_RATE_LIMIT", 10),
        _setting("REGISTER_RATE_WINDOW_SECONDS", 60),
    )
    if not allowed:
        raise RateLimitError(REGISTER_MESSAGE)


def enforce_analysis_rate_limit(*, video: bool) -> None:
    """Limit analysis starts for the signed-in user. Viewing stays unlimited."""

    user_id = int(current_user.id)
    kind = "video" if video else "image"
    limit_name = "VIDEO_ANALYSIS_RATE_LIMIT" if video else "ANALYSIS_RATE_LIMIT"
    default = 3 if video else 5
    allowed = get_limiter().try_consume(
        f"analysis:{kind}:{user_id}",
        _setting(limit_name, default),
        _setting("ANALYSIS_RATE_WINDOW_SECONDS", 60),
    )
    if not allowed:
        raise RateLimitError(ANALYSIS_MESSAGE)


def enforce_report_email_rate_limit() -> None:
    """Limit report emails for the signed-in user. Does not touch analysis buckets."""

    user_id = int(current_user.id)
    allowed = get_limiter().try_consume(
        f"report-email:{user_id}",
        _setting("REPORT_EMAIL_LIMIT", 5),
        _setting("REPORT_EMAIL_WINDOW_SECONDS", 60),
    )
    if not allowed:
        raise RateLimitError(REPORT_EMAIL_MESSAGE)
