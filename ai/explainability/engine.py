"""Explainability Engine — orchestrates registry explainers and artefacts.

Purpose
-------
Coordinate explainability without embedding CAM math. Explainer selection is
configuration + registry only — adding a plugin never requires engine edits.

Independence
------------
Does not import training, evaluation, or benchmark packages. Reuses inference
``ModelLoader`` / preprocessing / confidence via composition only.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ai.explainability.comparison import ComparisonResult, run_multi_explainer_comparison
from ai.explainability.config import ExplainabilityConfig, get_explainability_config
from ai.explainability.explanation import (
    ExplanationResult,
    write_explanation_json,
    write_explanation_report,
    write_pipeline_summary,
)
from ai.explainability.heatmap import prepare_heatmap, upsample_heatmap
from ai.explainability.overlay import overlay_heatmap
from ai.explainability.registry import ExplainerRegistry, register_default_explainers
from ai.explainability.target_layer import TargetLayerResolver
from ai.explainability.visualization import write_visualization_bundle
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.investigation_id import InvestigationIDGenerator
from ai.inference.model_loader import ModelLoader
from ai.inference.preprocessing import preprocess_image
from ai.inference.utils import ensure_file, timing_ms

logger = logging.getLogger("maya.ai.explainability.engine")


def _device_label(device: object) -> str:
    name = str(device).upper()
    if name.startswith("CUDA"):
        return "CUDA"
    if name.startswith("CPU"):
        return "CPU"
    return name


class ExplainabilityEngine:
    """High-level coordinator for investigator-facing explanations."""

    def __init__(
        self,
        config: ExplainabilityConfig | None = None,
        *,
        model_loader: ModelLoader | None = None,
        id_generator: InvestigationIDGenerator | None = None,
    ) -> None:
        self.config = config or get_explainability_config()
        register_default_explainers()

        infer_cfg = InferenceConfig(
            project_root=self.config.project_root,
            checkpoint_path=self.config.checkpoint_path,
            artifact_dir=self.config.artifact_dir,
            id_state_path=self.config.id_state_path,
            device_preference=self.config.device_preference,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            image_size=self.config.image_size,
        )
        self._infer_cfg = infer_cfg
        self._loader = model_loader or ModelLoader(infer_cfg)
        self._id_gen = id_generator or InvestigationIDGenerator(
            self.config.id_state_path
        )

    def explain(
        self,
        image_path: Path | str,
        *,
        investigation_id: str | None = None,
        artifact_dir: Path | str | None = None,
    ) -> ExplanationResult:
        """Run a single configured explainer (Sprint 4.1-compatible).

        Args:
            image_path: Evidence image path (does **not** determine output dir).
            investigation_id: Reuse an existing investigation ID (e.g. from
                ``scripts/predict.py``). When ``None``, allocate a new ID.
            artifact_dir: Override output directory for this run. When ``None``,
                uses ``config.artifact_dir`` (default Sprint 4.1 path). Outputs
                are never derived from the input image location.

        Target class: ``config.target_class`` when set; otherwise the predicted
        class from the same loaded checkpoint + preprocessing as inference.
        """

        cfg = self.config
        path = ensure_file(Path(image_path))
        out_dir = Path(artifact_dir) if artifact_dir is not None else Path(cfg.artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with timing_ms() as timer:
            loaded = self._loader.load()
            prepared = preprocess_image(path, self._infer_cfg)

            layer, layer_name = TargetLayerResolver.resolve(
                loaded.model,
                cfg.model_name,
                override=cfg.target_layer_override,
            )

            explainer = ExplainerRegistry.create(
                cfg.explainer_name,
                loaded.model,
                loaded.device,
            )
            if cfg.explainer_name.lower() == "scorecam" and hasattr(
                explainer, "batch_size"
            ):
                explainer.batch_size = cfg.scorecam_batch_size  # type: ignore[attr-defined]

            # None → predicted class inside explainer (select_target_class)
            cam = explainer.generate(
                prepared.tensor,
                target_class=cfg.target_class,
                target_layer=layer,
                target_layer_name=layer_name,
            )

            decision = decide_from_probabilities(
                cam.probabilities.tolist(),
                self._infer_cfg,
            )

            # The classifier still receives the configured square model input.
            # Render the CAM in source-image coordinates so its artifacts align
            # with the complete evidence image and preserve its aspect ratio.
            with Image.open(path) as img:
                original = img.convert("RGB")
            output_size = (original.height, original.width)
            # Match the evaluator's normalization grid first, then map those
            # normalized coordinates back onto the full source image.
            model_heatmap = prepare_heatmap(
                cam.raw_heatmap,
                (cfg.output_resolution, cfg.output_resolution),
                eps=cfg.heatmap_eps,
                interpolation=cfg.interpolation,
            )
            heatmap = upsample_heatmap(
                model_heatmap,
                output_size,
                interpolation=cfg.interpolation,
            )
            overlay = overlay_heatmap(
                original,
                heatmap,
                opacity=cfg.opacity,
                color_map=cfg.color_map,
            )

            viz_paths = write_visualization_bundle(
                source_image=original,
                heatmap=heatmap,
                overlay=overlay,
                artifact_dir=out_dir,
                color_map=cfg.color_map,
            )

            inv_id = (
                investigation_id.strip()
                if investigation_id and investigation_id.strip()
                else self._id_gen.next_id()
            )
            result = ExplanationResult(
                investigation_id=inv_id,
                explainer_name=explainer.name,
                model_name=loaded.model_name or cfg.model_name,
                model_version=loaded.model_version or cfg.model_version,
                dataset_version=cfg.dataset_version,
                prediction=decision.predicted_label,
                confidence=decision.confidence,
                target_class=cam.target_class,
                target_layer=cam.target_layer_name,
                generation_time_ms=0.0,
                device=_device_label(loaded.device),
                heatmap_path=str(viz_paths["heatmap"]),
                overlay_path=str(viz_paths["overlay"]),
                timestamp=ExplanationResult.utc_now(),
                comparison_path=str(viz_paths["comparison"]),
                original_path=str(viz_paths["original"]),
                image_name=path.name,
                real_probability=decision.real_probability,
                fake_probability=decision.fake_probability,
            )

            del cam, heatmap, overlay, original, prepared

        result.generation_time_ms = round(timer["ms"], 2)

        write_explanation_json(result, out_dir / "explanation.json")
        write_explanation_report(result, out_dir / "explanation_report.md")
        write_pipeline_summary(out_dir / "pipeline_summary.md")
        result.save_json(out_dir / "explanation_result.json")

        logger.info(
            "Explanation complete id=%s explainer=%s pred=%s target_class=%s "
            "artifact_dir=%s (%.1f ms)",
            result.investigation_id,
            result.explainer_name,
            result.prediction,
            result.target_class,
            out_dir,
            result.generation_time_ms,
        )
        return result

    def compare(self, image_path: Path | str) -> ComparisonResult:
        """Run all configured comparison explainers (Sprint 4.2).

        Algorithm names come from ``config.comparison_explainers`` + registry.
        """

        cfg = self.config
        path = ensure_file(Path(image_path))
        artifact_dir = Path(cfg.comparison_artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        loaded = self._loader.load()
        prepared = preprocess_image(path, self._infer_cfg)

        # Shared prediction metadata without running a CAM algorithm twice.
        from ai.inference.predictor import predict_probabilities

        probs = predict_probabilities(
            loaded.model, prepared.tensor, loaded.device
        )
        decision = decide_from_probabilities(probs[0], self._infer_cfg)

        with Image.open(path) as img:
            original = img.convert("RGB")

        result = run_multi_explainer_comparison(
            model=loaded.model,
            device=loaded.device,
            input_tensor=prepared.tensor,
            original_image=original,
            image_name=path.name,
            config=cfg,
            prediction=decision.predicted_label,
            confidence=decision.confidence,
            model_name=loaded.model_name or cfg.model_name,
            model_version=loaded.model_version or cfg.model_version,
            artifact_dir=artifact_dir,
            explainer_names=list(cfg.comparison_explainers),
        )
        del prepared, original
        return result
