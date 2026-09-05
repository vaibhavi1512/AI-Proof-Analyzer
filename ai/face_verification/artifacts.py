"""Write face-verification investigation artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ai.face_verification.types import FaceVerificationResult
from ai.inference.utils import write_text


def _save_crop(path: Path, crop: np.ndarray | None) -> str | None:
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return None
    array = crop
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).convert("RGB").save(path)
    return path.name


def write_face_verification_artifacts(
    result: FaceVerificationResult,
    artifact_dir: Path,
    *,
    investigation_id: str,
    extra: dict | None = None,
) -> dict[str, str]:
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_name = _save_crop(out / "reference_face.png", result.reference_crop)
    evd_name = _save_crop(out / "evidence_face.png", result.evidence_crop)

    comparison = result.to_public_dict()
    comparison["investigation_id"] = investigation_id
    if extra:
        comparison.update(extra)
    if ref_name:
        comparison["reference_face_artifact"] = ref_name
    if evd_name:
        comparison["evidence_face_artifact"] = evd_name

    write_text(out / "comparison_result.json", json.dumps(comparison, indent=2, default=str))

    report = {
        "verification_status": result.verification_status,
        "decision": result.decision,
        "similarity_score": result.similarity_score,
        "distance_score": result.distance_score,
        "threshold": result.threshold,
        "model_name": result.model_name,
        "model_version": result.model_version,
        "reference_face_count": result.reference_face_count,
        "evidence_face_count": result.evidence_face_count,
        "reason_code": result.reason_code,
        "investigation_id": investigation_id,
        "engine_name": result.engine_name,
        "no_match_threshold": result.no_match_threshold,
    }
    write_text(out / "verification_report.json", json.dumps(report, indent=2, default=str))

    names = {
        "comparison_result": "comparison_result.json",
        "verification_report": "verification_report.json",
    }
    if ref_name:
        names["reference_face"] = ref_name
    if evd_name:
        names["evidence_face"] = evd_name
    return names
