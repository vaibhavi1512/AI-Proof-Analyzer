"""Pytest suite for MAYA Phase 3 Sprint 4 investigation inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.inference.confidence import confidence_level_from_score, decide_from_probabilities
from ai.inference.inference_config import ConfidenceBands, InferenceConfig
from ai.inference.investigation_id import InvestigationIDGenerator
from ai.inference.model_loader import ModelLoader
from ai.inference.pipeline import InferencePipeline
from ai.inference.predictor import predict_probabilities
from ai.inference.preprocessing import ImagePreprocessError, preprocess_image
from ai.inference.result import InvestigationResult
from ai.inference.utils import timing_ms


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.jpg"
    Image.new("RGB", (128, 96), color=(40, 90, 140)).save(path)
    return path


@pytest.fixture()
def infer_config(tmp_path: Path) -> InferenceConfig:
    product = ROOT / "artifacts" / "checkpoints" / "processed_final" / "best.pt"
    legacy = ROOT / "artifacts" / "checkpoints" / "best.pt"
    ckpt = product if product.exists() else legacy
    return InferenceConfig(
        project_root=ROOT,
        checkpoint_path=ckpt if ckpt.exists() else ROOT / "missing.pt",
        artifact_dir=tmp_path / "artifacts" / "phase3" / "sprint4",
        id_state_path=tmp_path / "artifacts" / "phase3" / "sprint4" / "investigation_id_state.json",
        device_preference="cpu",
        threshold=0.5,
        image_size=224,
    )


def test_image_preprocessing(sample_image: Path, infer_config: InferenceConfig) -> None:
    prepared = preprocess_image(sample_image, infer_config)
    assert prepared.tensor.shape == (1, 3, 224, 224)
    assert prepared.image_name == "evidence.jpg"


def test_preprocess_rejects_bad_extension(tmp_path: Path, infer_config: InferenceConfig) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ImagePreprocessError):
        preprocess_image(bad, infer_config)


def test_model_loading(infer_config: InferenceConfig) -> None:
    loaded = ModelLoader(infer_config).load()
    assert loaded.model is not None
    assert loaded.device.type == "cpu"
    # Reuse cache
    again = ModelLoader(infer_config)
    again._cached = loaded
    assert again.load() is loaded


def test_predictor_output(infer_config: InferenceConfig, sample_image: Path) -> None:
    loaded = ModelLoader(infer_config).load()
    prepared = preprocess_image(sample_image, infer_config)
    probs = predict_probabilities(loaded.model, prepared.tensor, loaded.device)
    assert probs.shape[-1] == 2
    assert torch.allclose(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-5)


def test_confidence_and_threshold_logic() -> None:
    bands = ConfidenceBands(very_high=0.95, high=0.85, medium=0.70)
    assert confidence_level_from_score(0.96, bands) == "Very High"
    assert confidence_level_from_score(0.90, bands) == "High"
    assert confidence_level_from_score(0.75, bands) == "Medium"
    assert confidence_level_from_score(0.50, bands) == "Low"

    cfg = InferenceConfig(threshold=0.5, device_preference="cpu")
    # High FAKE probability → FAKE
    decision = decide_from_probabilities([0.2, 0.8], cfg)
    assert decision.predicted_label == "FAKE"
    assert decision.confidence == pytest.approx(80.0)
    # Low FAKE probability → REAL
    decision2 = decide_from_probabilities([0.7, 0.3], cfg)
    assert decision2.predicted_label == "REAL"


def test_investigation_id_generation(tmp_path: Path) -> None:
    gen = InvestigationIDGenerator(tmp_path / "id_state.json", year=2026)
    first = gen.next_id()
    second = gen.next_id()
    assert first == "INV-2026-000001"
    assert second == "INV-2026-000002"


def test_result_serialization(tmp_path: Path) -> None:
    result = InvestigationResult(
        investigation_id="INV-2026-000001",
        prediction="FAKE",
        confidence=88.5,
        confidence_level="High",
        real_probability=0.115,
        fake_probability=0.885,
        threshold=0.5,
        model_name="efficientnet_b0",
        model_version="test",
        dataset_version="v1",
        prediction_time_ms=12.5,
        timestamp=InvestigationResult.utc_now(),
        image_name="x.jpg",
        image_size=(224, 224),
        processing_device="cpu",
        processing_status="success",
    )
    path = result.save_json(tmp_path / "investigation_result.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["investigation_id"] == "INV-2026-000001"
    assert data["prediction"] == "FAKE"


def test_processing_timer() -> None:
    with timing_ms() as timer:
        _ = sum(range(1000))
    assert timer["ms"] >= 0.0


def test_pipeline_json_and_markdown(infer_config: InferenceConfig, sample_image: Path) -> None:
    pipeline = InferencePipeline(infer_config)
    result = pipeline.run(sample_image)
    assert result.processing_status == "success"
    assert result.investigation_id.startswith("INV-")
    out = Path(infer_config.artifact_dir)
    assert (out / "prediction.json").exists()
    assert (out / "investigation_result.json").exists()
    assert (out / "prediction_report.md").exists()
    assert (out / "pipeline_summary.md").exists()
    assert (out / "prediction_log.txt").exists()


def test_classify_image_matches_run_without_new_investigation(infer_config: InferenceConfig, sample_image: Path) -> None:
    pipeline = InferencePipeline(infer_config)
    scored = pipeline.classify_image(sample_image)
    result = pipeline.run(sample_image)
    assert scored.predicted_label == result.prediction
    assert scored.real_probability == pytest.approx(result.real_probability, abs=1e-6)
    assert scored.fake_probability == pytest.approx(result.fake_probability, abs=1e-6)
    assert scored.confidence == pytest.approx(result.confidence, abs=1e-6)
    assert pipeline.loader._cached is not None
    assert pipeline.loader.load() is pipeline.loader._cached
