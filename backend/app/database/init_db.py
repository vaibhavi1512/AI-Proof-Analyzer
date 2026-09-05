"""Database initialization helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask
from sqlalchemy import inspect, text
from sqlalchemy.sql.sqltypes import Boolean, DateTime, Float, Integer, String, Text

from backend.app.extensions import db

logger = logging.getLogger("maya.database")


def init_database(app: Flask) -> None:
    """Bind SQLAlchemy and create tables for registered models."""

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        # Ensure models are registered on metadata
        import backend.app.models  # noqa: F401

        db.create_all()
        _ensure_sqlite_columns()
        logger.info(
            "Database initialized (%s)",
            app.config.get("SQLALCHEMY_DATABASE_URI"),
        )


def _sqlite_ddl_type(column) -> str:
    """Map a SQLAlchemy column type to a conservative SQLite DDL fragment."""

    col_type = column.type
    if isinstance(col_type, Integer):
        return "INTEGER"
    if isinstance(col_type, Float):
        return "FLOAT"
    if isinstance(col_type, Boolean):
        return "BOOLEAN"
    if isinstance(col_type, DateTime):
        return "DATETIME"
    if isinstance(col_type, Text):
        return "TEXT"
    if isinstance(col_type, String):
        length = getattr(col_type, "length", None) or 255
        return f"VARCHAR({length})"
    return "TEXT"


def _ensure_sqlite_columns() -> None:
    """Add missing columns on existing SQLite tables (create_all does not ALTER).

    Never drops columns or tables. New columns are added as nullable so older
    rows remain valid. No-op for non-SQLite engines and in-memory DBs that
    were just created empty.
    """

    uri = str(db.engine.url)
    if not uri.startswith("sqlite"):
        return

    inspector = inspect(db.engine)
    existing_tables = set(inspector.get_table_names())
    for table in db.metadata.sorted_tables:
        if table.name not in existing_tables:
            continue
        existing_cols = {col["name"] for col in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing_cols:
                continue
            ddl_type = _sqlite_ddl_type(column)
            stmt = text(
                f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl_type}'
            )
            db.session.execute(stmt)
            logger.warning(
                "Added missing SQLite column %s.%s (%s)",
                table.name,
                column.name,
                ddl_type,
            )
        db.session.commit()
