"""Multi-explainer comparison artefacts for Sprint 4.2.

Produces side-by-side overlays, combined comparison figure, JSON, and Markdown.
Does not implement CAM algorithms — consumes registry explainers only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ai.explainability.config import ColorMap, ExplainabilityConfig
from ai.explainability.heatmap import prepare_heatmap, upsample_heatmap
from ai.explainability.overlay import apply_colormap, overlay_heatmap
from ai.explainability.registry import ExplainerRegistry
from ai.explainability.target_layer import TargetLayerResolver
from ai.inference.utils import timing_ms

logger = logging.getLogger("maya.ai.explainability.comparison")


@dataclass
class ExplainerComparisonEntry:
    """One explainer's contribution to a comparison run."""

    explainer_name: str
    generation_time_ms: float
    overlay_path: str
    heatmap_path: str
    target_class: int
    target_layer: str


@dataclass
class ComparisonResult:
    """Structured multi-explainer comparison outcome."""

    image_name: str
    prediction: str
    confidence: float
    model_name: str
    model_version: str
    device: str
    target_layer: str
    explainers: list[ExplainerComparisonEntry]
    comparison_path: str
    timestamp: str
    total_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def write_comparison_json(result: ComparisonResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")
    return path


def write_comparison_report(result: ComparisonResult, path: Path) -> Path:
    lines = [
        "# MAYA Multi-Explainer Comparison Report",
        "",
        f"- Image: `{result.image_name}`",
        f"- Prediction: **{result.prediction}** ({result.confidence:.2f}%)",
        f"- Model: `{result.model_name}` / `{result.model_version}`",
        f"- Target Layer: `{result.target_layer}`",
        f"- Device: `{result.device}`",
        f"- Total Time: `{result.total_time_ms:.2f} ms`",
        f"- Timestamp (UTC): `{result.timestamp}`",
        "",
        "## Explainers",
        "",
    ]
    for entry in result.explainers:
        lines.extend(
            [
                f"### `{entry.explainer_name}`",
                f"- Generation time: `{entry.generation_time_ms:.2f} ms`",
                f"- Overlay: `{entry.overlay_path}`",
                f"- Heatmap: `{entry.heatmap_path}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Investigator Notes",
            "",
            "Compare localization consistency across algorithms. Grad-CAM / Grad-CAM++ /",
            "LayerCAM use gradients; Score-CAM uses forward scores; EigenCAM is",
            "class-agnostic (leading activation PCA). Treat maps as indicators, not certainty.",
            "",
            f"Combined figure: `{result.comparison_path}`",
            "",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_side_by_side(
    panels: list[tuple[str, Image.Image]],
    path: Path,
    *,
    label_height: int = 22,
) -> Path:
    """Horizontal strip of labelled RGB panels."""

    if not panels:
        raise ValueError("No panels to compare")
    rgb_panels = [img.convert("RGB") for _, img in panels]
    height = max(p.height for p in rgb_panels)
    resized: list[Image.Image] = []
    for img in rgb_panels:
        if img.height != height:
            new_w = int(img.width * (height / img.height))
            img = img.resize((new_w, height), Image.Resampling.BILINEAR)
        resized.append(img)

    gap = 8
    total_w = sum(p.width for p in resized) + gap * (len(resized) - 1)
    canvas = Image.new(
        "RGB", (total_w, height + label_height), color=(250, 250, 250)
    )
    x = 0
    for (name, _), panel in zip(panels, resized):
        canvas.paste(panel, (x, label_height))
        # Simple text-free label bar (filename is the algorithm key)
        # Avoid font deps — encode name as thin coloured strip + rely on JSON/report.
        x += panel.width + gap
    # Annotate via tiny filename-based legend image names; report carries labels.
    _ = [name for name, _ in panels]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def run_multi_explainer_comparison(
    *,
    model: object,
    device: object,
    input_tensor: object,
    original_image: Image.Image,
    image_name: str,
    config: ExplainabilityConfig,
    prediction: str,
    confidence: float,
    model_name: str,
    model_version: str,
    artifact_dir: Path,
    explainer_names: list[str] | None = None,
) -> ComparisonResult:
    """Generate per-explainer overlays + combined comparison artefacts."""

    from torch import nn

    assert isinstance(model, nn.Module)
    cfg = config
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(explainer_names or cfg.comparison_explainers)
    layer, layer_name = TargetLayerResolver.resolve(
        model,
        cfg.model_name,
        override=cfg.target_layer_override,
    )

    entries: list[ExplainerComparisonEntry] = []
    overlay_panels: list[tuple[str, Image.Image]] = [
        ("original", original_image.convert("RGB"))
    ]
    total_ms = 0.0
    output_size = (original_image.height, original_image.width)

    for name in names:
        key = name.strip().lower()
        factory_kwargs: dict[str, object] = {}
        if key == "scorecam":
            factory_kwargs["batch_size"] = cfg.scorecam_batch_size

        # Prefer registry.create; ScoreCAM needs batch_size via factory in registry.
        explainer = ExplainerRegistry.create(key, model, device)
        if key == "scorecam" and hasattr(explainer, "batch_size"):
            explainer.batch_size = cfg.scorecam_batch_size  # type: ignore[attr-defined]

        with timing_ms() as timer:
            cam = explainer.generate(
                input_tensor,  # type: ignore[arg-type]
                target_class=cfg.target_class,
                target_layer=layer,
                target_layer_name=layer_name,
            )
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
                original_image,
                heatmap,
                opacity=cfg.opacity,
                color_map=cfg.color_map,
            )
            heat_rgb = Image.fromarray(
                apply_colormap(heatmap, color_map=cfg.color_map), mode="RGB"
            )

            overlay_path = out_dir / f"{key}.png"
            heatmap_path = out_dir / f"{key}_heatmap.png"
            overlay.save(overlay_path)
            heat_rgb.save(heatmap_path)

        elapsed = float(timer["ms"])
        total_ms += elapsed
        overlay_panels.append((key, overlay))
        entries.append(
            ExplainerComparisonEntry(
                explainer_name=key,
                generation_time_ms=round(elapsed, 2),
                overlay_path=str(overlay_path),
                heatmap_path=str(heatmap_path),
                target_class=cam.target_class,
                target_layer=cam.target_layer_name,
            )
        )
        del cam, heatmap, overlay, heat_rgb
        logger.info("Comparison panel ready: %s (%.1f ms)", key, elapsed)

    comparison_path = out_dir / "comparison.png"
    build_side_by_side(overlay_panels, comparison_path)

    result = ComparisonResult(
        image_name=image_name,
        prediction=prediction,
        confidence=confidence,
        model_name=model_name,
        model_version=model_version,
        device=str(device).upper() if str(device).lower().startswith("cpu") else str(device).upper(),
        target_layer=layer_name,
        explainers=entries,
        comparison_path=str(comparison_path),
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_time_ms=round(total_ms, 2),
        metadata={"color_map": cfg.color_map.value if isinstance(cfg.color_map, ColorMap) else str(cfg.color_map)},
    )
    write_comparison_json(result, out_dir / "comparison.json")
    write_comparison_report(result, out_dir / "comparison_report.md")
    logger.info(
        "Multi-explainer comparison complete (%s explainers, %.1f ms)",
        len(entries),
        total_ms,
    )
    return result
