"""Authentication service (registration, login, current user)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from flask_login import login_user, logout_user
from sqlalchemy import func, or_

from backend.app.audit import record_audit
from backend.app.exceptions import AuthenticationError, ConflictError, ValidationError
from backend.app.extensions import db
from backend.app.models.entities import User
from backend.app.models.enums import AuditEventType, UserRole
from backend.app.security import hash_password, verify_password
from backend.app.security.passwords import validate_password_strength

logger = logging.getLogger("maya.backend.auth")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def register_user(
    *,
    email: str,
    username: str,
    password: str,
    full_name: str = "",
    role: UserRole = UserRole.INVESTIGATOR,
) -> User:
    email = (email or "").strip().lower()
    username = (username or "").strip()
    full_name = (full_name or "").strip()
    if not email or not _EMAIL_RE.match(email):
        raise ValidationError("Valid email is required")
    if not username or len(username) < 3:
        raise ValidationError("Username must be at least 3 characters")
    validate_password_strength(password)
    if role not in {UserRole.INVESTIGATOR, UserRole.ADMIN}:
        raise ValidationError("Invalid role")

    existing = User.query.filter(
        or_(func.lower(User.email) == email, User.username == username)
    ).first()
    if existing:
        raise ConflictError("Email or username already registered")

    user = User(
        email=email,
        username=username,
        password_hash=hash_password(password),
        full_name=full_name or username,
        role=role.value,
        is_active=True,
    )
    db.session.add(user)
    db.session.flush()
    record_audit(
        AuditEventType.USER_REGISTERED,
        user_id=user.id,
        details={"email": email, "username": username, "role": role.value},
    )
    db.session.commit()
    logger.info("Registered user id=%s username=%s", user.id, username)
    return user


def authenticate_user(*, login: str, password: str) -> User:
    login_value = (login or "").strip()
    if not login_value or not password:
        raise AuthenticationError("Invalid credentials")

    user = User.query.filter(
        or_(
            func.lower(User.email) == login_value.lower(),
            User.username == login_value,
        )
    ).first()
    if user is None or not user.is_active:
        raise AuthenticationError("Invalid credentials")
    if not verify_password(user.password_hash, password):
        raise AuthenticationError("Invalid credentials")

    login_user(user, remember=False)
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    record_audit(
        AuditEventType.USER_LOGIN,
        user_id=user.id,
        details={"username": user.username},
    )
    db.session.commit()
    logger.info("User login id=%s", user.id)
    return user


def logout_current_user() -> None:
    from flask_login import current_user as _cu

    user_id = None
    username = None
    if _cu.is_authenticated:
        user_id = int(_cu.id)
        username = getattr(_cu, "username", None)
    logout_user()
    if user_id is not None:
        record_audit(
            AuditEventType.USER_LOGOUT,
            user_id=user_id,
            details={"username": username} if username else {},
        )
        db.session.commit()
