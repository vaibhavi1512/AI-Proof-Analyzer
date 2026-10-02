"""Password hashing and authentication helpers (Werkzeug)."""

from __future__ import annotations

import re
from functools import wraps
from typing import Any, Callable, TypeVar

from flask_login import current_user
from werkzeug.security import check_password_hash, generate_password_hash

from backend.app.exceptions import AuthenticationError, AuthorizationError, ValidationError
from backend.app.models.enums import UserRole

F = TypeVar("F", bound=Callable[..., Any])


_UPPER = re.compile(r"[A-Z]")
_LOWER = re.compile(r"[a-z]")
_DIGIT = re.compile(r"[0-9]")
_SPECIAL = re.compile(r"[^A-Za-z0-9]")

PASSWORD_REQUIREMENTS_MESSAGE = (
    "Password must be at least 8 characters and include an uppercase letter, "
    "a lowercase letter, a number, and a special character."
)


def password_requirement_status(password: str) -> dict[str, bool]:
    """Public strength checks. The password itself is never returned."""

    value = password or ""
    return {
        "length": len(value) >= 8,
        "uppercase": _UPPER.search(value) is not None,
        "lowercase": _LOWER.search(value) is not None,
        "number": _DIGIT.search(value) is not None,
        "special": _SPECIAL.search(value) is not None,
    }


def validate_password_strength(password: str) -> None:
    """Reject a new password that misses any strength requirement."""

    if not all(password_requirement_status(password).values()):
        raise ValidationError(PASSWORD_REQUIREMENTS_MESSAGE)


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def login_required_api(view: F) -> F:
    """Require an authenticated Flask-Login user for JSON APIs."""

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        if not current_user.is_authenticated:
            raise AuthenticationError("Authentication required")
        return view(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def roles_required(*roles: UserRole | str) -> Callable[[F], F]:
    allowed = {
        r.value if isinstance(r, UserRole) else str(r).upper() for r in roles
    }

    def decorator(view: F) -> F:
        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any):
            if not current_user.is_authenticated:
                raise AuthenticationError("Authentication required")
            if current_user.role not in allowed:
                raise AuthorizationError("Insufficient role for this operation")
            return view(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


def get_current_user_id() -> int:
    if not current_user.is_authenticated:
        raise AuthenticationError("Authentication required")
    return int(current_user.id)
