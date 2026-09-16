"""AI bridge — orchestrates existing inference / XAI without duplicating logic."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai.explainability import (
    AdvancedExplainabilityEngine,
    AdvancedXAIConfig,
    DEFAULT_EXPLAINER_NAMES,
    ExplainabilityConfig,
    ExplainabilityEngine,
    ShapSettings,
    FaithfulnessSettings,
    CounterfactualSettings,
    TrustScoreWeights,
    fuse_explanations,
)
from ai.explainability.comparison import run_multi_explainer_comparison
from ai.explainability.config import get_explainability_config
from ai.explainability.registry import ExplainerRegistry
from ai.explainability.target_layer import TargetLayerResolver
from ai.inference import InferenceConfig, InferencePipeline
from ai.inference.model_loader import ModelLoader
from ai.inference.preprocessing import preprocess_image
from ai.inference.result import InvestigationResult
from ai.models.model_config import _project_root

logger = logging.getLogger("maya.backend.integrations.ai")


@dataclass
class AiAnalysisBundle:
    investigation: InvestigationResult
    explanation: dict[str, Any] | None = None


_pipeline: InferencePipeline | None = None


def _checkpoint_path() -> Path:
    return _project_root() / "artifacts" / "checkpoints" / "processed_final" / "best.pt"


def get_inference_pipeline() -> InferencePipeline:
    """Process-level cached pipeline (reuses ModelLoader cache)."""

    global _pipeline
    if _pipeline is None:
        _pipeline = InferencePipeline(
            InferenceConfig(
                project_root=_project_root(),
                checkpoint_path=_checkpoint_path(),
                device_preference="cpu",
            )
        )
    return _pipeline


def reset_inference_pipeline() -> None:
    """Test helper."""

    global _pipeline
    _pipeline = None


def run_inference(
    image_path: Path,
    *,
    artifact_dir: Path | None = None,
) -> InvestigationResult:
    pipeline = get_inference_pipeline()
    logger.info("Calling InferencePipeline.run path=%s", image_path)
    return pipeline.run(image_path, artifact_dir_override=artifact_dir)


def run_explanation(
    image_path: Path,
    *,
    investigation_id: str,
    explainer: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    cfg = ExplainabilityConfig(
        project_root=_project_root(),
        device_preference="cpu",
        explainer_name=explainer,
        artifact_dir=artifact_dir,
        checkpoint_path=_checkpoint_path(),
    )
    engine = ExplainabilityEngine(cfg)
    # Reuse investigation ID from inference; isolate artefacts
    result = engine.explain(
        image_path,
        investigation_id=investigation_id,
        artifact_dir=artifact_dir,
    )
    return {
        "explainer": result.explainer_name,
        "heatmap": result.heatmap_path,
        "overlay": result.overlay_path,
        "comparison": result.comparison_path,
        "explanation_json": str(Path(artifact_dir) / "explanation_result.json"),
        "prediction": result.prediction,
        "confidence": result.confidence,
        "target_class": result.target_class,
    }


def run_advanced_xai(
    image_path: Path,
    investigation_id: str,
    artifact_dir: Path,
    *,
    xai_config: dict[str, Any],
) -> dict[str, Any]:
    """Run advanced XAI pipeline with configurable methods.

    xai_config options:
      explainer: str (default "gradcam") - single explainer: gradcam/gradcam_plus_plus/layercam/scorecam/eigencam
      multi_explainer: bool (default False) - run all 5 explainers
      shap: bool (default False)
      faithfulness: bool (default False)
      counterfactual: bool (default False)
      fusion: bool (default False) - requires at least 2 methods
      trust: bool (default False)

    Returns structured dict with results + artifact paths from each method.
    """
    explainer_name = xai_config.get("explainer", "gradcam")
    multi_explainer = bool(xai_config.get("multi_explainer", False))
    enable_shap = bool(xai_config.get("shap", False))
    enable_faithfulness = bool(xai_config.get("faithfulness", False))
    enable_counterfactual = bool(xai_config.get("counterfactual", False))
    enable_fusion = bool(xai_config.get("fusion", False))
    enable_trust = bool(xai_config.get("trust", False))

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "investigation_id": investigation_id,
        "methods_run": [],
        "artifact_paths": {},
        "explainers": {},
        "trust_score": None,
        "quality_score": None,
    }

    root = _project_root()
    cp_path = _checkpoint_path()

    methods_count = 0
    explainer_results: dict[str, Any] = {}
    cam_maps: dict[str, Any] = {}

    explainer_list = list(DEFAULT_EXPLAINER_NAMES) if multi_explainer else [explainer_name]

    try:
        infer_cfg = InferenceConfig(
            project_root=root,
            checkpoint_path=cp_path,
            artifact_dir=artifact_dir,
            device_preference="cpu",
        )
        loader = ModelLoader(infer_cfg)
        loaded = loader.load()
        prepared = preprocess_image(Path(image_path), infer_cfg)

        out_h = 224
        with __import__("PIL").Image.open(Path(image_path)) as img:
            original = img.convert("RGB").resize((out_h, out_h), __import__("PIL").Image.Resampling.BILINEAR)

        for exp_name in explainer_list:
            try:
                exp_dir = artifact_dir / "explainers" / exp_name
                exp_dir.mkdir(parents=True, exist_ok=True)
                cfg = ExplainabilityConfig(
                    project_root=root,
                    device_preference="cpu",
                    explainer_name=exp_name,
                    artifact_dir=exp_dir,
                    checkpoint_path=cp_path,
                )
                engine = ExplainabilityEngine(cfg)
                exp_result = engine.explain(
                    image_path,
                    investigation_id=investigation_id,
                    artifact_dir=exp_dir,
                )
                explainer_results[exp_name] = {
                    "explainer": exp_result.explainer_name,
                    "heatmap": exp_result.heatmap_path,
                    "overlay": exp_result.overlay_path,
                    "comparison": exp_result.comparison_path,
                    "explanation_json": str(exp_dir / "explanation_result.json"),
                    "prediction": exp_result.prediction,
                    "confidence": exp_result.confidence,
                    "target_class": exp_result.target_class,
                }
                results["artifact_paths"][f"{exp_name}_heatmap"] = exp_result.heatmap_path
                results["artifact_paths"][f"{exp_name}_overlay"] = exp_result.overlay_path
                results["artifact_paths"][f"{exp_name}_comparison"] = exp_result.comparison_path
                methods_count += 1
            except Exception as e:
                logger.warning("Explainer %s failed: %s", exp_name, e)
                explainer_results[exp_name] = {"error": str(e)}

        results["explainers"] = explainer_results
        if multi_explainer or len(explainer_list) > 1:
            results["methods_run"].extend(explainer_list)
        else:
            results["methods_run"].append(explainer_name)

        advanced_cfg = AdvancedXAIConfig(
            project_root=root,
            checkpoint_path=cp_path,
            artifact_dir=artifact_dir,
            primary_cam_explainer=explainer_name,
            shap=ShapSettings(enabled=enable_shap),
            faithfulness=FaithfulnessSettings(enabled=enable_faithfulness),
            counterfactual=CounterfactualSettings(enabled=enable_counterfactual),
            trust=TrustScoreWeights(),
        )

        if enable_shap or enable_faithfulness or enable_counterfactual or enable_fusion or enable_trust:
            try:
                adv_engine = AdvancedExplainabilityEngine(config=advanced_cfg)
                adv_result = adv_engine.analyze(image_path)
                results["artifact_paths"].update(adv_result.artifact_paths)

                if enable_shap:
                    results["shap"] = {
                        "enabled": True,
                        "heatmap": adv_result.artifact_paths.get("shap_heatmap"),
                        "overlay": adv_result.artifact_paths.get("shap_overlay"),
                        "comparison": adv_result.artifact_paths.get("shap_comparison"),
                    }
                    results["methods_run"].append("shap")
                    methods_count += 1

                if enable_faithfulness:
                    results["faithfulness"] = {
                        "enabled": True,
                        "comparison": adv_result.artifact_paths.get("faithfulness_comparison"),
                        "report": adv_result.artifact_paths.get("faithfulness_report"),
                    }
                    results["methods_run"].append("faithfulness")
                    methods_count += 1

                if enable_counterfactual:
                    results["counterfactual"] = {
                        "enabled": True,
                        "comparison": adv_result.artifact_paths.get("counterfactual_comparison"),
                        "report": adv_result.artifact_paths.get("counterfactual_report"),
                    }
                    results["methods_run"].append("counterfactual")
                    methods_count += 1

                if enable_fusion and methods_count >= 2:
                    results["fusion"] = {
                        "enabled": True,
                        "fused_explanation": adv_result.artifact_paths.get("fused_explanation"),
                        "report": adv_result.artifact_paths.get("advanced_xai_summary"),
                    }
                    results["methods_run"].append("fusion")

                if enable_trust:
                    results["trust_score"] = adv_result.trust.trust_score
                    results["quality_score"] = (
                        adv_result.trust.components.get("quality")
                        if hasattr(adv_result.trust, "components")
                        else None
                    )
                    results["trust"] = {
                        "enabled": True,
                        "trust_score": adv_result.trust.trust_score,
                        "grade": adv_result.trust.grade if hasattr(adv_result.trust, "grade") else None,
                        "components": (
                            asdict(adv_result.trust.components)
                            if hasattr(adv_result.trust, "components")
                            else {}
                        ),
                        "summary_chart": adv_result.artifact_paths.get("trust_summary"),
                        "report": adv_result.artifact_paths.get("explanation_trust_report"),
                    }
                    results["methods_run"].append("trust")

                results["artifact_paths"]["advanced_xai_result"] = adv_result.artifact_paths.get(
                    "advanced_xai_result"
                )
                results["artifact_paths"]["xai_audit"] = adv_result.artifact_paths.get("xai_audit")

                del loaded, prepared, original
            except Exception as e:
                logger.exception("Advanced XAI stages failed: %s", e)
                results["advanced_error"] = str(e)
    except Exception as e:
        logger.exception("Advanced XAI setup failed: %s", e)
        results["setup_error"] = str(e)

    return results
