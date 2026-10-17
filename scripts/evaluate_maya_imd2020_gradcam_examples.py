"""Isolated full IMD2020 audit using MAYA's production classifier and Grad-CAM.

This script writes only to artifacts/MAYA_GradCAM_Best_Examples. It does not
train models or import/use any localization model.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.explainability.config import ColorMap, ExplainabilityConfig
from ai.explainability.explainers.gradcam import GradCAM
from ai.explainability.heatmap import prepare_heatmap
from ai.explainability.overlay import apply_colormap, overlay_heatmap
from ai.explainability.target_layer import TargetLayerResolver
from ai.explainability.visualization import save_comparison_figure
from ai.inference.confidence import decide_from_probabilities
from ai.inference.inference_config import InferenceConfig
from ai.inference.model_loader import ModelLoader
from ai.inference.preprocessing import preprocess_image


DATASET = ROOT / "dataset 16.09" / "IMD2020"
OUT = ROOT / "artifacts" / "MAYA_GradCAM_Best_Examples"
WORK = OUT / ".work"
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "processed_final" / "best.pt"
THRESHOLDS = (0.25, 0.40, 0.50, 0.60, 0.75)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def discover_pairs() -> tuple[list[dict], list[dict]]:
    audit: list[dict] = []
    pairs: list[dict] = []
    for folder in sorted(p for p in DATASET.rglob("*") if p.is_dir()):
        files = [p for p in folder.iterdir() if p.is_file()]
        originals = [p for p in files if p.suffix.lower() in EXTS and p.stem.lower().endswith("_orig")]
        masks = [p for p in files if p.suffix.lower() == ".png" and "_mask" in p.stem.lower()]
        if not originals and not masks:
            continue
        folder_id = folder.relative_to(DATASET).as_posix()
        orig = originals[0] if len(originals) == 1 else None
        for m in masks:
            fake_stem = m.stem[:m.stem.lower().rfind("_mask")]
            candidates = [p for p in files if p.suffix.lower() in EXTS and p.stem == fake_stem]
            reason = ""
            if len(originals) != 1:
                reason = f"expected exactly one *_orig image in folder; found {len(originals)}"
            elif len(candidates) != 1:
                reason = f"expected exactly one fake image matching mask basename; found {len(candidates)}"
            else:
                try:
                    oi, fi, mi = read_rgb(orig), read_rgb(candidates[0]), None
                    with Image.open(m) as mask:
                        mask.load()
                        mi = mask.copy()
                    if min(oi.size + fi.size + mi.size) < 1:
                        reason = "invalid image or mask dimensions"
                    elif fi.size != mi.size:
                        reason = f"fake/mask dimensions differ: fake={fi.size}, mask={mi.size}"
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    reason = f"unreadable pair: {type(exc).__name__}: {exc}"
            row = {
                "parent_folder": folder_id,
                "sample_id": fake_stem,
                "original_path": str(orig) if orig else "",
                "fake_path": str(candidates[0]) if len(candidates) == 1 else "",
                "mask_path": str(m),
                "original_dimensions": "",
                "fake_dimensions": "",
                "mask_dimensions": "",
                "pairing_status": "valid" if not reason else "excluded",
                "readability_status": "readable" if not reason else "invalid",
                "exclusion_reason": reason,
            }
            if not reason:
                row["original_dimensions"] = f"{oi.width}x{oi.height}"
                row["fake_dimensions"] = f"{fi.width}x{fi.height}"
                row["mask_dimensions"] = f"{mi.width}x{mi.height}"
                pairs.append({"parent": folder_id, "sample": fake_stem, "original": orig, "fake": candidates[0], "mask": m})
            audit.append(row)
        if originals and not masks:
            audit.append({"parent_folder": folder_id, "sample_id": originals[0].stem, "original_path": str(originals[0]), "fake_path": "", "mask_path": "", "original_dimensions": "", "fake_dimensions": "", "mask_dimensions": "", "pairing_status": "excluded", "readability_status": "unknown", "exclusion_reason": "no manipulation masks found in parent folder"})
    return pairs, audit


def binary_metrics(heat: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    pred = heat >= threshold
    gt = mask
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"threshold": threshold, "dice": dice, "iou": iou, "precision": precision, "recall": recall,
            "false_positive_rate_outside_mask": fp / (fp + tn) if fp + tn else 0.0,
            "activation_fraction": float(pred.mean())}


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_contact(rows: list[dict], path: Path, kind: str) -> None:
    thumb_w, thumb_h, label_h, cols = 220, 180, 42, 5
    rows = rows[:25]
    if not rows:
        Image.new("RGB", (500, 80), "white").save(path)
        return
    canvas = Image.new("RGB", (cols * thumb_w, math.ceil(len(rows) / cols) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        p = Path(row["selected_image_path"])
        if kind == "FAKE":
            image = Image.open(row["overlay_path"]).convert("RGB")
        else:
            image = Image.open(row["overlay_path"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + label_h)
        canvas.paste(image, (x + (thumb_w-image.width)//2, y))
        draw.text((x + 4, y + thumb_h + 2), f"{i+1:02d} {row['sample_id']}  conf={float(row['confidence']):.1f}%", fill="black")
    canvas.save(path)


def main() -> None:
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"MAYA production checkpoint missing: {CHECKPOINT}")
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pairs, audit = discover_pairs()
    csv_write(OUT / "reports" / "dataset_audit.csv", audit)
    print(f"Dataset audit: {len(pairs)} valid pairs, {len(audit)-len(pairs)} excluded/other", flush=True)

    infer_cfg = InferenceConfig(project_root=ROOT, checkpoint_path=CHECKPOINT, artifact_dir=WORK,
                                id_state_path=WORK / "unused_ids.json", device_preference="auto")
    loader = ModelLoader(infer_cfg)
    xai_cfg = ExplainabilityConfig(project_root=ROOT, checkpoint_path=CHECKPOINT, artifact_dir=WORK,
                                   id_state_path=WORK / "unused_xai_ids.json", device_preference="auto",
                                   explainer_name="gradcam", target_class=None, output_resolution=224)
    loaded = loader.load()
    layer, layer_name = TargetLayerResolver.resolve(loaded.model, xai_cfg.model_name, override=None)
    explainer = GradCAM(loaded.model, loaded.device)
    results: list[dict] = []
    pair_by_fake: dict[str, dict] = {str(p["fake"]): p for p in pairs}
    image_jobs: list[tuple[str, Path, dict | None]] = []
    for pair in pairs:
        image_jobs.append(("FAKE", pair["fake"], pair))
    # A parent folder may contain many manipulated variants sharing one source
    # original. Infer that REAL image once and retain one explicitly paired fake
    # only as the reference material in a selected REAL bundle.
    real_representatives: dict[str, dict] = {}
    for pair in pairs:
        real_representatives.setdefault(str(pair["original"]), pair)
    for pair in real_representatives.values():
        image_jobs.append(("REAL", pair["original"], pair))
    # Work records make the long complete-dataset run resumable.
    progress = WORK / "inference.jsonl"
    completed: dict[str, dict] = {}
    if progress.is_file():
        for line in progress.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("heatmap_npy") and Path(row["heatmap_npy"]).is_file():
                    completed[row["source_path"]] = row
            except (ValueError, OSError):
                pass
    for n, (label, image_path, pair) in enumerate(image_jobs, 1):
        key = str(image_path)
        if key in completed:
            cached = dict(completed[key])
            cached.update(ground_truth_label=label, source_path=key,
                         parent_folder=pair["parent"],
                         sample_id=image_path.stem if label == "REAL" else pair["sample"],
                         original_path=str(pair["original"]),
                         fake_path=str(pair["fake"]), mask_path=str(pair["mask"]))
            results.append(cached)
            continue
        t0 = time.perf_counter()
        rec = {"ground_truth_label": label, "source_path": key, "parent_folder": pair["parent"],
               "sample_id": image_path.stem if label == "REAL" else pair["sample"], "original_path": str(pair["original"]),
               "fake_path": str(pair["fake"]), "mask_path": str(pair["mask"]),
               "prediction": "", "confidence": "", "real_probability": "", "fake_probability": "",
               "correct": False, "heatmap_path": "", "overlay_path": "", "heatmap_npy": "",
               "inference_status": "failed", "runtime_seconds": 0.0, "error": ""}
        try:
            prepared = preprocess_image(image_path, infer_cfg)
            cam = explainer.generate(prepared.tensor, target_class=None, target_layer=layer, target_layer_name=layer_name)
            decision = decide_from_probabilities(cam.probabilities.tolist(), infer_cfg)
            heat = prepare_heatmap(cam.raw_heatmap, (224, 224), eps=xai_cfg.heatmap_eps, interpolation=xai_cfg.interpolation)
            output_dir = WORK / "heatmaps" / label.lower()
            output_dir.mkdir(parents=True, exist_ok=True)
            slug = f"{pair['parent'].replace('/', '_')}__{pair['sample']}__{label.lower()}"
            heat_npy = output_dir / f"{slug}.npy"
            heat_png = output_dir / f"{slug}.png"
            overlay_path = output_dir / f"{slug}_overlay.png"
            np.save(heat_npy, heat)
            Image.fromarray(apply_colormap(heat, ColorMap.JET)).save(heat_png)
            original = read_rgb(image_path).resize((224, 224), Image.Resampling.BILINEAR)
            overlay_heatmap(original, heat, opacity=xai_cfg.opacity, color_map=xai_cfg.color_map).save(overlay_path)
            rec.update(prediction=decision.predicted_label, confidence=decision.confidence,
                       real_probability=decision.real_probability, fake_probability=decision.fake_probability,
                       correct=decision.predicted_label == label, target_class=cam.target_class,
                       target_layer=layer_name, heatmap_path=str(heat_png), overlay_path=str(overlay_path),
                       heatmap_npy=str(heat_npy), inference_status="success")
            del prepared, cam, heat
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["runtime_seconds"] = round(time.perf_counter() - t0, 4)
        results.append(rec)
        with progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 50 == 0:
            print(f"Analyzed {n}/{len(image_jobs)}", flush=True)

    # Mask metrics use a predeclared threshold sweep on the normalized MAYA map.
    for row in results:
        if row["ground_truth_label"] != "FAKE" or row["inference_status"] != "success":
            continue
        with Image.open(row["mask_path"]) as im:
            gt = np.asarray(im.convert("L").resize((224, 224), Image.Resampling.NEAREST)) > 0
        heat = np.load(row["heatmap_npy"])
        sweep = [binary_metrics(heat, gt, t) for t in THRESHOLDS]
        row["threshold_metrics"] = json.dumps(sweep, separators=(",", ":"))
        row["metrics_at_0_50"] = next(m for m in sweep if m["threshold"] == 0.5)
        row["mean_dice_sweep"] = float(np.mean([m["dice"] for m in sweep]))
        row["mean_iou_sweep"] = float(np.mean([m["iou"] for m in sweep]))
        row["mean_precision_sweep"] = float(np.mean([m["precision"] for m in sweep]))
        row["mean_recall_sweep"] = float(np.mean([m["recall"] for m in sweep]))
        row["outside_activation_at_0_50"] = row["metrics_at_0_50"]["false_positive_rate_outside_mask"]
        row["mask_coverage"] = float(gt.mean())
        row["ranking_score"] = (0.25 * float(row["fake_probability"]) + 0.35 * row["mean_dice_sweep"] +
                                0.20 * row["mean_iou_sweep"] + 0.10 * row["mean_precision_sweep"] +
                                0.10 * row["mean_recall_sweep"])
    real_by_parent = {}
    for row in results:
        if row["ground_truth_label"] == "REAL" and row["inference_status"] == "success":
            activation = float(np.mean(np.load(row["heatmap_npy"]) >= 0.5))
            row["high_activation_fraction_at_0_50"] = activation
            row["ranking_score"] = 0.75 * float(row["real_probability"]) + 0.25 * (1.0 - activation)
            real_by_parent.setdefault(row["parent_folder"], []).append(row)

    fake_candidates = [r for r in results if r["ground_truth_label"] == "FAKE" and r["inference_status"] == "success" and r["correct"]]
    real_candidates = [r for r in results if r["ground_truth_label"] == "REAL" and r["inference_status"] == "success" and r["correct"]]
    for r in fake_candidates:
        r["exclusion_reason"] = "" if 0.001 <= r["mask_coverage"] <= 0.95 else (
            "manipulation mask coverage below 0.1%" if r["mask_coverage"] < 0.001
            else "manipulation mask coverage above 95%"
        )
    fake_candidates = [r for r in fake_candidates if 0.001 <= r["mask_coverage"] <= 0.95]
    fake_candidates.sort(key=lambda r: r["ranking_score"], reverse=True)
    real_candidates.sort(key=lambda r: r["ranking_score"], reverse=True)
    # Greedy diversity: at most two selected examples per parent when alternatives exist.
    def diverse(candidates: list[dict], cap: int = 2) -> list[dict]:
        chosen, deferred, counts = [], [], {}
        for row in candidates:
            if counts.get(row["parent_folder"], 0) < cap and len(chosen) < 25:
                chosen.append(row); counts[row["parent_folder"]] = counts.get(row["parent_folder"], 0) + 1
            else:
                deferred.append(row)
        for row in deferred:
            if len(chosen) == 25: break
            chosen.append(row)
        return chosen
    chosen_fake, chosen_real = diverse(fake_candidates), diverse(real_candidates)

    # Export numbered folders, exact source pairs, MAYA heatmaps, overlays and reports.
    def export(rows: list[dict], class_name: str) -> list[dict]:
        root = OUT / ("FAKE_TOP_25" if class_name == "FAKE" else "REAL_TOP_25")
        root.mkdir(parents=True, exist_ok=True)
        output_rows = []
        for i, row in enumerate(rows, 1):
            folder = root / f"{i:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            pair = pair_by_fake[row["fake_path"]]
            original, fake = read_rgb(pair["original"]), read_rgb(pair["fake"])
            mask = Image.open(pair["mask"]).convert("L")
            if class_name == "FAKE":
                names = {"original_image.png": original, "fake_image.png": fake, "ground_truth_mask.png": mask}
                selected = fake
                cam_orig = next((x for x in results if x["source_path"] == str(pair["original"])), None)
                for name, img in names.items(): img.save(folder / name)
                shutil.copy2(row["heatmap_path"], folder / "gradcam_fake.png")
                shutil.copy2(row["overlay_path"], folder / "gradcam_overlay_fake.png")
                if cam_orig and cam_orig["inference_status"] == "success": shutil.copy2(cam_orig["heatmap_path"], folder / "gradcam_original.png")
                else: Image.new("RGB", (224,224)).save(folder / "gradcam_original.png")
                comp = Image.new("RGB", (896, 480), "white")
                panels = [original, fake, mask.convert("RGB"), Image.open(row["overlay_path"]).convert("RGB")]
                labels = ["Original", "FAKE", "Ground truth mask", "MAYA Grad-CAM overlay"]
                draw = ImageDraw.Draw(comp)
                for j, (im, label) in enumerate(zip(panels, labels)):
                    im.thumbnail((220, 430)); x = j*224
                    comp.paste(im, (x, 30)); draw.text((x+3, 8), label, fill="black")
                comp.save(folder / "comparison.png")
                metrics_obj = {k: row.get(k) for k in ("ground_truth_label", "prediction", "confidence", "real_probability", "fake_probability", "correct", "parent_folder", "sample_id", "ranking_score", "mean_dice_sweep", "mean_iou_sweep", "mean_precision_sweep", "mean_recall_sweep", "outside_activation_at_0_50", "mask_coverage", "threshold_metrics", "source_path", "original_path", "fake_path", "mask_path", "target_layer")}
                Image.open(row["overlay_path"]).verify()
            else:
                original.save(folder / "original_image.png")
                shutil.copy2(row["heatmap_path"], folder / "gradcam_original.png")
                shutil.copy2(row["overlay_path"], folder / "gradcam_overlay_original.png")
                fake.save(folder / "paired_fake_image.png")
                mask.save(folder / "paired_fake_ground_truth_mask.png")
                selected = original
                fake_row = next(x for x in results if x["source_path"] == str(pair["fake"]))
                comp = Image.new("RGB", (672, 480), "white")
                panels = [original, Image.open(row["overlay_path"]).convert("RGB"), fake, mask.convert("RGB")]
                labels = ["REAL original", "MAYA Grad-CAM", "Paired fake reference", "Paired fake mask reference (not GT for original)"]
                draw = ImageDraw.Draw(comp)
                for j, (im, label) in enumerate(zip(panels, labels)):
                    im.thumbnail((165, 430)); x = j*168
                    comp.paste(im, (x, 45)); draw.text((x+2, 5), label[:25], fill="black")
                comp.save(folder / "comparison.png")
                metrics_obj = {k: row.get(k) for k in ("ground_truth_label", "prediction", "confidence", "real_probability", "fake_probability", "correct", "parent_folder", "sample_id", "ranking_score", "high_activation_fraction_at_0_50", "source_path", "original_path", "fake_path", "mask_path", "target_layer")}
                metrics_obj["paired_fake_reference_prediction"] = fake_row.get("prediction")
                metrics_obj["paired_fake_reference_mask_semantics"] = "Mask belongs to paired fake image; it is not ground truth for original_image.png."
            metrics_obj.update({"gradcam_source": "ai.explainability.explainers.gradcam.GradCAM", "checkpoint": str(CHECKPOINT), "target_class_rule": "predicted argmax class (production default)", "export_rank": i})
            (folder / "metrics.json").write_text(json.dumps(metrics_obj, indent=2), encoding="utf-8")
            out = dict(row)
            out.update(export_folder=str(folder), selected_image_path=str(pair["fake"] if class_name == "FAKE" else pair["original"]), overlay_path=str(folder / ("gradcam_overlay_fake.png" if class_name == "FAKE" else "gradcam_overlay_original.png")))
            output_rows.append(out)
        return output_rows

    selected_paths = {r["source_path"] for r in chosen_fake + chosen_real}
    for row in results:
        if row["inference_status"] != "success":
            row["exclusion_reason"] = row.get("error") or "inference failed"
        elif not row["correct"]:
            row["exclusion_reason"] = f"incorrectly classified as {row['prediction']}"
        elif row["source_path"] in selected_paths:
            row["exclusion_reason"] = ""
        elif row["ground_truth_label"] == "FAKE" and not (0.001 <= row.get("mask_coverage", 0) <= 0.95):
            row["exclusion_reason"] = "manipulation mask coverage outside 0.1%–95%"
        else:
            row["exclusion_reason"] = "eligible but below top-25 ranking cutoff"
        row["selection_status"] = "selected" if row["source_path"] in selected_paths else "not selected"
    exp_fake, exp_real = export(chosen_fake, "FAKE"), export(chosen_real, "REAL")
    # Include every attempted image, plus status and exclusions.
    csv_write(OUT / "reports" / "all_samples.csv", results)
    csv_write(OUT / "reports" / "selected_fake.csv", exp_fake)
    csv_write(OUT / "reports" / "selected_real.csv", exp_real)
    make_contact(exp_fake, OUT / "reports" / "fake_contact_sheet.png", "FAKE")
    make_contact(exp_real, OUT / "reports" / "real_contact_sheet.png", "REAL")
    successful = [r for r in results if r["inference_status"] == "success"]
    accuracy = sum(bool(r["correct"]) for r in successful) / len(successful) if successful else 0
    confusion = {label: {pred: sum(r["ground_truth_label"] == label and r["prediction"] == pred for r in successful)
                         for pred in ("REAL", "FAKE")} for label in ("REAL", "FAKE")}
    fakes = [r for r in successful if r["ground_truth_label"] == "FAKE"]
    md = ["# MAYA IMD2020 Grad-CAM example evaluation", "",
          "## Dataset and analysis", "",
          f"- Valid manipulation pairs: {len(pairs)}; audit rows: {len(audit)}.",
          f"- Eligible image analyses: {len(results)}; successful: {len(successful)}; failed: {len(results)-len(successful)}.",
          f"- Classification accuracy across successful REAL and FAKE images: {accuracy:.4f} ({sum(bool(r['correct']) for r in successful)}/{len(successful)}).",
          f"- Confusion counts (ground truth rows, prediction columns): REAL→REAL {confusion['REAL']['REAL']}, REAL→FAKE {confusion['REAL']['FAKE']}; FAKE→REAL {confusion['FAKE']['REAL']}, FAKE→FAKE {confusion['FAKE']['FAKE']}.",
          f"- Selected REAL: {len(exp_real)}; selected FAKE: {len(exp_fake)}.",
          f"- Mean FAKE heatmap/mask Dice at fixed threshold 0.50: {np.mean([r['metrics_at_0_50']['dice'] for r in fakes]) if fakes else 0:.4f}; IoU: {np.mean([r['metrics_at_0_50']['iou'] for r in fakes]) if fakes else 0:.4f}.",
          f"- Mean FAKE Dice across thresholds {list(THRESHOLDS)}: {np.mean([r['mean_dice_sweep'] for r in fakes]) if fakes else 0:.4f}.", "",
          f"- Selected FAKE examples (n={len(chosen_fake)}): mean Dice/IoU at 0.50 = {np.mean([r['metrics_at_0_50']['dice'] for r in chosen_fake]) if chosen_fake else 0:.4f}/{np.mean([r['metrics_at_0_50']['iou'] for r in chosen_fake]) if chosen_fake else 0:.4f}; mean Dice across threshold sweep = {np.mean([r['mean_dice_sweep'] for r in chosen_fake]) if chosen_fake else 0:.4f}.", "",
          "## Ranking methodology", "",
          "Fake score = 0.25 × fake probability + 0.35 × mean Dice + 0.20 × mean IoU + 0.10 × mean precision + 0.10 × mean recall, with overlap averaged across thresholds 0.25, 0.40, 0.50, 0.60, and 0.75. Only correctly classified fake images with mask coverage from 0.1% through 95% are eligible. Real score = 0.75 × real probability + 0.25 × (1 − high-activation area fraction at 0.50); only correctly classified originals are eligible. Greedy selection caps each parent folder at two examples while alternatives exist, then fills any remaining slots by score. all_samples.csv records each exclusion reason and selection status.", "",
          "## Interpretation and limitations", "",
          "The mask is paired only to its fake image. For each fake, the mask is nearest-neighbor resized to the 224×224 MAYA heatmap grid only when dimensions differ. Heatmaps come from MAYA's EfficientNet-B0 classifier and production Grad-CAM implementation, target-layer resolver, ImageNet preprocessing, bilinear upsampling, min-max normalization, JET display and 0.45 opacity overlay. Threshold overlap measures classifier-attention agreement with dataset masks; it does not establish pixel-level manipulation detection. Dataset pairing requires one *_orig image per folder and exact basename matches between fake image and *_mask.png. See dataset_audit.csv and all_samples.csv for exclusions and individual status.", "",
          "## Files", "",
          f"Output root: `{OUT}`", "", "- `FAKE_TOP_25/` and `REAL_TOP_25/`: selected numbered image bundles.",
          "- `reports/all_samples.csv`, `selected_fake.csv`, `selected_real.csv`, `dataset_audit.csv`.",
          "- `reports/fake_contact_sheet.png`, `real_contact_sheet.png`.", "",
          f"Checkpoint used: `{CHECKPOINT}`", "Production code and checkpoint were not modified."]
    (OUT / "reports" / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Complete: REAL={len(exp_real)}, FAKE={len(exp_fake)}, analyses={len(results)}, success={len(successful)}", flush=True)


if __name__ == "__main__":
    main()
