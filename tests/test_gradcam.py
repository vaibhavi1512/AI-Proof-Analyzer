"""Pytest suite for MAYA Phase 4 Sprint 4.1 Grad-CAM explainability."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.base import Explainer
from ai.explainability.config import ColorMap, ExplainabilityConfig
from ai.explainability.engine import ExplainabilityEngine
from ai.explainability.explanation import (
    ExplanationResult,
    write_explanation_json,
    write_explanation_report,
)
from ai.explainability.gradcam import GradCAM, compute_gradcam_map
from ai.explainability.heatmap import normalize_heatmap, prepare_heatmap, upsample_heatmap
from ai.explainability.hooks import ActivationGradientHooks
from ai.explainability.overlay import apply_colormap, overlay_heatmap
from ai.explainability.registry import ExplainerRegistry, register_default_explainers
from ai.explainability.target_layer import (
    TargetLayerResolver,
    UnsupportedModelForExplainability,
    register_default_target_layers,
)
from ai.inference.inference_config import InferenceConfig
from ai.inference.model_loader import ModelLoader
from ai.inference.preprocessing import preprocess_image


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.jpg"
    Image.new("RGB", (160, 120), color=(60, 100, 150)).save(path)
    return path


@pytest.fixture()
def xai_config(tmp_path: Path) -> ExplainabilityConfig:
    ckpt = ROOT / "artifacts" / "checkpoints" / "best.pt"
    return ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ckpt if ckpt.exists() else ROOT / "missing.pt",
        artifact_dir=tmp_path / "artifacts" / "phase4" / "sprint1",
        id_state_path=tmp_path / "artifacts" / "phase4" / "sprint1" / "id_state.json",
        device_preference="cpu",
        opacity=0.45,
        color_map=ColorMap.JET,
        output_resolution=224,
    )


@pytest.fixture()
def loaded_model(xai_config: ExplainabilityConfig):
    infer = InferenceConfig(
        project_root=xai_config.project_root,
        checkpoint_path=xai_config.checkpoint_path,
        device_preference="cpu",
        image_size=224,
    )
    return ModelLoader(infer).load()


def test_registry_works() -> None:
    ExplainerRegistry.clear()
    register_default_explainers()
    assert "gradcam" in ExplainerRegistry.list_available()
    factory = ExplainerRegistry.get("gradcam")
    assert callable(factory)
    with pytest.raises(KeyError):
        ExplainerRegistry.get("not_a_real_explainer")


def test_target_layer_detection(loaded_model) -> None:
    register_default_target_layers()
    layer, name = TargetLayerResolver.resolve(
        loaded_model.model, "efficientnet_b0"
    )
    assert isinstance(layer, nn.Module)
    assert name.startswith("features.")
    with pytest.raises(UnsupportedModelForExplainability):
        TargetLayerResolver.resolve(loaded_model.model, "unknown_cnn_xyz")


def test_hook_registration_and_cleanup(loaded_model, sample_image: Path, xai_config) -> None:
    infer = InferenceConfig(
        project_root=xai_config.project_root,
        checkpoint_path=xai_config.checkpoint_path,
        device_preference="cpu",
    )
    prepared = preprocess_image(sample_image, infer)
    layer, _ = TargetLayerResolver.resolve(loaded_model.model, "efficientnet_b0")
    tensor = prepared.tensor.to(loaded_model.device).requires_grad_(True)

    hooks = ActivationGradientHooks(layer)
    hooks.register()
    assert len(hooks._handles) == 2
    logits = loaded_model.model(tensor)
    logits[0, 1].backward()
    assert hooks.activations is not None
    assert hooks.gradients is not None
    hooks.remove()
    assert hooks._handles == []
    assert hooks.activations is None
    assert hooks.gradients is None


def test_hook_context_manager_cleanup(loaded_model) -> None:
    layer, _ = TargetLayerResolver.resolve(loaded_model.model, "efficientnet_b0")
    with ActivationGradientHooks(layer) as hooks:
        assert len(hooks._handles) == 2
    assert hooks._handles == []


def test_heatmap_generation() -> None:
    raw = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    # Upsample → normalize (paper visualization order)
    prepared = prepare_heatmap(raw, (8, 8))
    assert prepared.shape == (8, 8)
    assert prepared.min() == pytest.approx(0.0, abs=1e-5)
    assert prepared.max() == pytest.approx(1.0, abs=1e-5)
    norm = normalize_heatmap(raw)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_overlay_generation(sample_image: Path) -> None:
    img = Image.open(sample_image).convert("RGB").resize((64, 64))
    heat = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    overlay = overlay_heatmap(img, heat, opacity=0.4, color_map=ColorMap.JET)
    assert overlay.size == (64, 64)
    assert overlay.mode == "RGB"


def test_explanation_result_serialization(tmp_path: Path) -> None:
    result = ExplanationResult(
        investigation_id="INV-2026-000001",
        explainer_name="gradcam",
        model_name="efficientnet_b0",
        model_version="test",
        dataset_version="v1",
        prediction="FAKE",
        confidence=97.61,
        target_class=1,
        target_layer="features.8",
        generation_time_ms=118.0,
        device="CPU",
        heatmap_path="heatmap.png",
        overlay_path="overlay.png",
        timestamp=ExplanationResult.utc_now(),
    )
    payload = result.to_dict()
    assert payload["prediction"] == "FAKE"
    path = result.save_json(tmp_path / "full.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["explainer_name"] == "gradcam"


def test_json_and_markdown_generation(tmp_path: Path) -> None:
    result = ExplanationResult(
        investigation_id="INV-2026-000002",
        explainer_name="gradcam",
        model_name="efficientnet_b0",
        model_version="test",
        dataset_version="v1",
        prediction="REAL",
        confidence=88.0,
        target_class=0,
        target_layer="features.8",
        generation_time_ms=50.0,
        device="CPU",
        heatmap_path="h.png",
        overlay_path="o.png",
        timestamp=ExplanationResult.utc_now(),
        image_name="x.jpg",
    )
    jpath = write_explanation_json(result, tmp_path / "explanation.json")
    mpath = write_explanation_report(result, tmp_path / "explanation_report.md")
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert data["explainer"] == "gradcam"
    assert "prediction" in data
    md = mpath.read_text(encoding="utf-8")
    assert "Investigator Notes" in md
    assert "Target Layer" in md


def test_gradcam_paper_equations() -> None:
    """α = GAP(∂y/∂A); L = ReLU(Σ α_k A^k) — Selvaraju et al. ICCV 2017."""

    # Synthetic A and ∂y/∂A with known closed form
    activations = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[0.5, 0.5], [0.5, 0.5]]]],
        dtype=torch.float32,
    )  # (1, 2, 2, 2)
    gradients = torch.tensor(
        [[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]],
        dtype=torch.float32,
    )
    # α0 = 1.0, α1 = 2.0
    # L_ij = ReLU(1*A0 + 2*A1) → [[2, 3], [4, 5]]
    cam = compute_gradcam_map(activations, gradients)
    expected = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(cam, expected, rtol=1e-5)

    # Negative combination must be zeroed by ReLU
    neg_grads = torch.tensor(
        [[[[-1.0, -1.0], [-1.0, -1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]],
        dtype=torch.float32,
    )
    cam_neg = compute_gradcam_map(activations, neg_grads)
    assert float(cam_neg.max()) == 0.0


def test_gradcam_cpu_execution(
    loaded_model, sample_image: Path, xai_config: ExplainabilityConfig
) -> None:
    infer = InferenceConfig(
        project_root=xai_config.project_root,
        checkpoint_path=xai_config.checkpoint_path,
        device_preference="cpu",
    )
    prepared = preprocess_image(sample_image, infer)
    layer, name = TargetLayerResolver.resolve(loaded_model.model, "efficientnet_b0")
    explainer = GradCAM(loaded_model.model, loaded_model.device)
    assert isinstance(explainer, Explainer)
    out = explainer.generate(
        prepared.tensor,
        target_layer=layer,
        target_layer_name=name,
    )
    assert out.raw_heatmap.ndim == 2
    assert out.probabilities.shape == (2,)
    assert out.target_layer_name == name


def test_engine_end_to_end(
    sample_image: Path, xai_config: ExplainabilityConfig
) -> None:
    engine = ExplainabilityEngine(xai_config)
    result = engine.explain(sample_image)
    assert result.prediction in {"REAL", "FAKE"}
    assert result.device == "CPU"
    artifact_dir = Path(xai_config.artifact_dir)
    assert (artifact_dir / "heatmap.png").exists()
    assert (artifact_dir / "overlay.png").exists()
    assert (artifact_dir / "comparison.png").exists()
    assert (artifact_dir / "explanation.json").exists()
    assert (artifact_dir / "explanation_report.md").exists()
    assert (artifact_dir / "pipeline_summary.md").exists()
    expected_size = Image.open(sample_image).size
    for artifact in ("original.png", "heatmap.png", "overlay.png"):
        with Image.open(artifact_dir / artifact) as image:
            assert image.size == expected_size


def test_engine_preserves_portrait_coordinates(
    tmp_path: Path, xai_config: ExplainabilityConfig
) -> None:
    evidence = tmp_path / "portrait.png"
    Image.new("RGB", (90, 170), color=(30, 80, 120)).save(evidence)
    out_dir = tmp_path / "portrait-xai"

    ExplainabilityEngine(xai_config).explain(evidence, artifact_dir=out_dir)

    for artifact in ("original.png", "heatmap.png", "overlay.png"):
        with Image.open(out_dir / artifact) as image:
            assert image.size == (90, 170)


def test_selected_imd2020_example_keeps_prediction_and_full_image_cam(
    tmp_path: Path,
) -> None:
    example_dir = ROOT / "artifacts" / "MAYA_GradCAM_Best_Examples" / "FAKE_TOP_25" / "01"
    evidence = example_dir / "fake_image.png"
    expected = json.loads((example_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = ExplainabilityConfig(
        project_root=ROOT,
        checkpoint_path=ROOT / "artifacts" / "checkpoints" / "processed_final" / "best.pt",
        artifact_dir=tmp_path / "selected-example",
        id_state_path=tmp_path / "selected-example" / "unused-id-state.json",
        device_preference="cpu",
    )

    result = ExplainabilityEngine(cfg).explain(evidence, artifact_dir=cfg.artifact_dir)

    with Image.open(evidence) as source:
        assert source.size == (1600, 1067)
        expected_size = source.size
    assert result.prediction == expected["prediction"]
    assert result.confidence == pytest.approx(expected["confidence"], abs=0.001)
    selected_csv = (
        ROOT / "artifacts" / "MAYA_GradCAM_Best_Examples" / "reports" / "selected_fake.csv"
    )
    with selected_csv.open(newline="", encoding="utf-8-sig") as handle:
        selected_row = next(
            row for row in csv.DictReader(handle)
            if Path(row["export_folder"]) == example_dir
        )
    evaluator_map = np.load(selected_row["heatmap_npy"])
    expected_full_map = upsample_heatmap(evaluator_map, (expected_size[1], expected_size[0]))
    with Image.open(Path(cfg.artifact_dir) / "heatmap.png") as image:
        np.testing.assert_array_equal(np.asarray(image), apply_colormap(expected_full_map))
    for artifact in ("original.png", "heatmap.png", "overlay.png"):
        with Image.open(Path(cfg.artifact_dir) / artifact) as image:
            assert image.size == expected_size


def test_memory_cleanup_after_gradcam(
    loaded_model, sample_image: Path, xai_config: ExplainabilityConfig
) -> None:
    infer = InferenceConfig(
        project_root=xai_config.project_root,
        checkpoint_path=xai_config.checkpoint_path,
        device_preference="cpu",
    )
    prepared = preprocess_image(sample_image, infer)
    layer, name = TargetLayerResolver.resolve(loaded_model.model, "efficientnet_b0")
    hooks = ActivationGradientHooks(layer)
    explainer = GradCAM(loaded_model.model, loaded_model.device)
    explainer.generate(prepared.tensor, target_layer=layer, target_layer_name=name)
    # After GradCAM.generate, hooks context has exited — no leftover handles
    assert hooks._handles == []
    assert hooks.activations is None
    assert hooks.gradients is None
