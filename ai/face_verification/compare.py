"""Similarity / distance and MATCH / NO_MATCH / INCONCLUSIVE decision."""

from __future__ import annotations

import numpy as np

from ai.face_verification.types import (
    DECISION_INCONCLUSIVE,
    DECISION_MATCH,
    DECISION_NO_MATCH,
    REASON_INCONCLUSIVE_SCORE,
    REASON_OK,
)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = l2_normalize(left)
    b = l2_normalize(right)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = l2_normalize(left)
    b = l2_normalize(right)
    return float(np.linalg.norm(a - b))


def decide_verification(
    similarity: float,
    *,
    match_threshold: float,
    no_match_threshold: float,
) -> tuple[str, str]:
    """Return (decision, reason_code) from cosine similarity."""

    if similarity >= match_threshold:
        return DECISION_MATCH, REASON_OK
    if similarity < no_match_threshold:
        return DECISION_NO_MATCH, REASON_OK
    return DECISION_INCONCLUSIVE, REASON_INCONCLUSIVE_SCORE
