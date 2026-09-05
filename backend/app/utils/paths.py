"""Path containment helpers.

Why this exists:
    ``str(path).startswith(str(root))`` treats ``/data/uploads-evil`` as living
    inside ``/data/uploads``. Every filesystem boundary check in the product
    layer goes through :func:`resolve_within` so the comparison is done on path
    components instead of raw string prefixes.
"""

from __future__ import annotations

from pathlib import Path


def is_within(root: Path | str, candidate: Path | str) -> bool:
    """Return True when ``candidate`` resolves inside ``root``."""

    root_path = Path(root).resolve()
    candidate_path = Path(candidate).resolve()
    try:
        return candidate_path == root_path or candidate_path.is_relative_to(root_path)
    except (OSError, ValueError):
        return False


def resolve_within(root: Path | str, relative: Path | str) -> Path | None:
    """Join ``relative`` onto ``root`` and return it only if it stays inside.

    An absolute ``relative`` (or one containing ``..``) that escapes ``root``
    yields ``None`` rather than a path outside the storage boundary.
    """

    root_path = Path(root).resolve()
    candidate = (root_path / relative).resolve()
    return candidate if is_within(root_path, candidate) else None
