"""Reproducible, isolated ManTraNet evaluation on IMD2020 image/mask pairs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset 16.09" / "IMD2020"
MODEL_DIR = ROOT / "external" / "ManTraNet-pytorch" / "MantraNet"
OUTPUT_DIR = ROOT / "artifacts" / "mantranet_localization" / "batch_evaluation"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)


def relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def discover_pairs(sample_list: Path | None):
    files = sorted((p for p in DATASET_ROOT.rglob("*")
                    if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.as_posix().lower())
    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for path in files:
        by_folder[path.parent].append(path)
    rows: list[dict] = []
    candidates: list[dict] = []
    used_masks: set[Path] = set()
    mask_files = {p for p in files if p.stem.lower().endswith("_mask")}
    selected_paths = read_sample_list(sample_list) if sample_list else None
    selected_found: set[Path] = set()

    for image in files:
        stem = image.stem.lower()
        if image in mask_files:
            continue
        if stem.endswith("_orig") or stem.endswith("_original"):
            rows.append(manifest_row(image, None, "", "excluded_original"))
            continue
        matching_masks = [p for p in by_folder[image.parent]
                          if p.stem.lower() == stem + "_mask"]
        if len(matching_masks) > 1:
            rows.append(manifest_row(image, None, "", "excluded_ambiguous_mask_match"))
            continue
        if not matching_masks:
            rows.append(manifest_row(image, None, "", "excluded_unmatched_image"))
            continue
        mask = matching_masks[0]
        used_masks.add(mask)
        if selected_paths is not None and image.resolve() not in selected_paths:
            rows.append(manifest_row(image, mask, "", "excluded_not_in_explicit_sample_list"))
            continue
        if selected_paths is not None:
            selected_found.add(image.resolve())
        candidates.append({"image": image, "mask": mask, "parent": image.parent})

    for mask in sorted(mask_files, key=lambda p: p.as_posix().lower()):
        if mask not in used_masks:
            rows.append(manifest_row(None, mask, "", "excluded_unmatched_mask"))
    if selected_paths is not None:
        for missing in sorted(selected_paths - selected_found, key=str):
            rows.append({"image": str(missing), "mask": "", "parent_folder": "",
                         "split": "", "status": "excluded_sample_list_entry_not_found"})
    return candidates, rows


def read_sample_list(path: Path) -> set[Path]:
    entries: set[Path] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        if "," in first:
            reader = csv.DictReader(handle)
            field = next((name for name in ("image", "image_path") if name in (reader.fieldnames or [])), None)
            if field is None:
                raise ValueError("Sample-list CSV needs an 'image' or 'image_path' column")
            values = (row.get(field, "") for row in reader)
        else:
            values = (line.strip() for line in handle)
        for value in values:
            value = value.strip()
            if not value or value.startswith("#"):
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            entries.add(candidate.resolve())
    return entries


def manifest_row(image: Path | None, mask: Path | None, split: str, status: str) -> dict:
    parent = image.parent if image is not None else (mask.parent if mask is not None else None)
    return {"image": relative(image), "mask": relative(mask),
            "parent_folder": relative(parent), "split": split, "status": status}


def load_binary_mask(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, bool, bool]:
    with Image.open(path) as source:
        mask = source.copy()
    if mask.mode in ("RGB", "RGBA"):
        channels = np.asarray(mask)[..., :3].astype(np.int16)
        spread = channels.max(axis=2) - channels.min(axis=2)
        arr = np.rint(channels @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    elif mask.mode in ("1", "L", "I", "I;16", "F"):
        arr = np.asarray(mask)
        spread = np.zeros(arr.shape, dtype=np.uint8)
    else:
        raise ValueError(f"unsupported mask mode {mask.mode!r}")
    values = np.unique(arr)
    normalized = False
    exact_binary = len(values) <= 2 and (len(values) == 1 or (values[0] == 0 and values[1] > 0))
    if not exact_binary:
        near_extremes = (arr <= 10) | (arr >= 245)
        if float(np.mean(near_extremes)) < 0.98 or float(np.mean(spread > 8)) > 0.001:
            raise ValueError("mask is not binary or near-grayscale binary")
        arr = (arr >= 128).astype(np.uint8)
        normalized = True
    resized = mask.size != image_size
    if resized:
        # Convert binary mask to 0/255 and use nearest-neighbor to retain class labels.
        binary_image = Image.fromarray((arr > 0).astype(np.uint8) * 255, mode="L")
        binary_image = binary_image.resize(image_size, Image.Resampling.NEAREST)
        arr = np.asarray(binary_image)
    return np.asarray(arr > 0, dtype=bool), resized, normalized


def validate_pair(item: dict) -> None:
    with Image.open(item["image"]) as source:
        source.verify()
    with Image.open(item["image"]) as source:
        image_size = source.size
    load_binary_mask(item["mask"], image_size)


def score_counts(tp: int, fp: int, fn: int) -> tuple[dict[str, float], list[str]]:
    pred_n, truth_n = tp + fp, tp + fn
    undefined = []
    if pred_n + truth_n == 0:
        dice = iou = f1 = 1.0
        undefined.extend(("dice", "iou", "f1"))
    else:
        dice = 2 * tp / (pred_n + truth_n)
        union = tp + fp + fn
        iou = tp / union if union else 1.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    precision = tp / pred_n if pred_n else (1.0 if truth_n == 0 else 0.0)
    recall = tp / truth_n if truth_n else (1.0 if pred_n == 0 else 0.0)
    if pred_n == 0:
        undefined.append("precision")
    if truth_n == 0:
        undefined.append("recall")
    return ({"dice": dice, "iou": iou, "precision": precision,
             "recall": recall, "f1": f1}, undefined)


def split_candidates(candidates: list[dict], seed: int) -> None:
    folders = sorted({item["parent"] for item in candidates}, key=lambda p: p.as_posix().lower())
    rng = random.Random(seed)
    rng.shuffle(folders)
    if len(folders) <= 1:
        validation_count = len(folders)
    else:
        validation_count = min(len(folders) - 1, max(1, round(len(folders) * 0.7)))
    validation_folders = set(folders[:validation_count])
    for item in candidates:
        item["split"] = "validation" if item["parent"] in validation_folders else "test"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_visual(path: Path, rgb: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                threshold: float, title: str) -> None:
    overlay = rgb.astype(np.float32) / 255.0
    selected = prediction >= threshold
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    overlay[selected] = 0.52 * overlay[selected] + 0.48 * red[selected]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(rgb); axes[0].set_title("Manipulated image")
    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground-truth mask")
    heat = axes[2].imshow(prediction, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Raw ManTraNet score map")
    fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(overlay); axes[3].set_title(f"Prediction overlay (score ≥ {threshold:.1f})")
    fig.suptitle(title, fontsize=11)
    for axis in axes:
        axis.axis("off")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def predict_score_map(model, rgb: np.ndarray, device: torch.device,
                      max_side: int) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width)) if max_side else 1.0
    if scale < 1.0:
        infer_width = max(1, round(width * scale))
        infer_height = max(1, round(height * scale))
        infer_rgb = np.asarray(Image.fromarray(rgb).resize(
            (infer_width, infer_height), Image.Resampling.LANCZOS), dtype=np.uint8)
    else:
        infer_rgb = rgb
    tensor = torch.from_numpy(infer_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor.to(device))
        if output.ndim != 4 or tuple(output.shape[:2]) != (1, 1):
            raise ValueError(f"unexpected model output shape {tuple(output.shape)}")
        if output.shape[-2:] != (height, width):
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    elapsed = time.perf_counter() - start
    prediction = output[0, 0].cpu().numpy()
    return prediction, elapsed


def main() -> int:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=30,
                        help="maximum pairs to evaluate; use 0 for all candidates (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="sampling and folder split seed")
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch CPU intra-op threads (default: 1 for predictable memory use)")
    parser.add_argument("--max-side", type=int, default=1280,
                        help="cap inference image's longest side; 0 keeps native resolution (default: 1280)")
    parser.add_argument("--sample-list", type=Path,
                        help="optional CSV (image/image_path column) or newline-separated image paths")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="output directory (default: existing batch_evaluation directory)")
    args = parser.parse_args()
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.max_side < 0:
        parser.error("--max-side must be nonnegative")
    if not DATASET_ROOT.is_dir():
        parser.error(f"dataset root does not exist: {DATASET_ROOT}")

    total_start = time.perf_counter()
    # The default PyTorch thread count can spawn many workers and use excessive
    # memory for the larger, variable-resolution IMD2020 images.
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = OUTPUT_DIR / "progress.log"

    def log(message: str) -> None:
        line = f"+{time.perf_counter() - total_start:8.1f}s {message}"
        print(line, flush=True)
        with progress_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    progress_path.write_text("ManTraNet controlled IMD2020 evaluation progress\n", encoding="utf-8")
    log("Discovering exact image/mask pairs")
    candidates, manifest = discover_pairs(args.sample_list)
    discovered_count = len(candidates)
    log(f"Found {discovered_count} exact pairs; prevalidating sample candidates")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = []
    prevalidation_errors: Counter[str] = Counter()
    limit = args.max_samples or len(candidates)
    stop_index = len(candidates)
    for candidate_index, item in enumerate(candidates):
        if len(selected) >= limit:
            stop_index = candidate_index
            break
        try:
            validate_pair(item)
            selected.append(item)
        except Exception as exc:
            reason = f"excluded_prevalidation_error: {type(exc).__name__}: {exc}"
            prevalidation_errors[type(exc).__name__ + ": " + str(exc)] += 1
            manifest.append(manifest_row(item["image"], item["mask"], "", reason))
    not_selected = candidates[stop_index:]
    for item in not_selected:
        manifest.append(manifest_row(item["image"], item["mask"], "", "excluded_max_samples_limit"))
    split_candidates(selected, args.seed)
    for item in selected:
        manifest.append(manifest_row(item["image"], item["mask"], item["split"], "selected_pending"))
    log(f"Selected {len(selected)} valid pairs from {len({x['parent'] for x in selected})} parent folders")

    previous_cwd = Path.cwd()
    os.chdir(MODEL_DIR)
    try:
        sys.path.insert(0, str(MODEL_DIR))
        from mantranet import MantraNet
        device = torch.device("cpu")
        model = MantraNet(device=device)
        model.load_state_dict(torch.load("MantraNetv4.pt", map_location=device))
    finally:
        os.chdir(previous_cwd)
    model.to(device).eval()
    log("Pretrained ManTraNet loaded once on CPU")

    per_image: list[dict] = []
    inference_seconds = []
    errors: Counter[str] = Counter()
    completed = []
    for index, item in enumerate(selected, start=1):
        image_path, mask_path = item["image"], item["mask"]
        row = next(r for r in reversed(manifest) if r["image"] == relative(image_path) and r["status"] == "selected_pending")
        sample_start = time.perf_counter()
        try:
            log(f"[{index}/{len(selected)}] START {relative(image_path)} split={item['split']}")
            with Image.open(image_path) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            truth, resized, normalized = load_binary_mask(mask_path, (rgb.shape[1], rgb.shape[0]))
            prediction, elapsed = predict_score_map(model, rgb, device, args.max_side)
            if prediction.shape != truth.shape:
                raise ValueError(f"prediction shape {prediction.shape} differs from mask {truth.shape}")
            if not np.isfinite(prediction).all():
                raise ValueError("prediction contains non-finite scores")
            inference_seconds.append(elapsed)
            metrics_by_threshold = {}
            for threshold in THRESHOLDS:
                predicted = prediction >= threshold
                tp = int(np.count_nonzero(predicted & truth))
                fp = int(np.count_nonzero(predicted & ~truth))
                fn = int(np.count_nonzero(~predicted & truth))
                metrics, undefined = score_counts(tp, fp, fn)
                metrics_by_threshold[threshold] = metrics
                per_image.append({"image": relative(image_path), "mask": relative(mask_path),
                                  "parent_folder": relative(item["parent"]), "split": item["split"],
                                  "threshold": threshold, **metrics, "tp": tp, "fp": fp, "fn": fn,
                                  "undefined_metrics": ";".join(undefined),
                                  "inference_seconds": elapsed,
                                  "sample_elapsed_seconds": time.perf_counter() - sample_start,
                                  "truth_positive_pixels": int(np.count_nonzero(truth)),
                                  "prediction_min": float(prediction.min()),
                                  "prediction_max": float(prediction.max()),
                                  "mask_resized_nearest": resized,
                                  "mask_normalized_luma": normalized})
            item.update(metrics=metrics_by_threshold, seconds=elapsed, resized=resized)
            status_parts = ["processed"]
            if resized:
                status_parts.append("mask_resized_nearest")
            if normalized:
                status_parts.append("mask_normalized_luma")
            row["status"] = ";".join(status_parts)
            item["status"] = row["status"]
            row["inference_seconds"] = f"{elapsed:.6f}"
            row["sample_elapsed_seconds"] = f"{time.perf_counter() - sample_start:.6f}"
            completed.append(item)
            map_dir = OUTPUT_DIR / "prediction_maps"
            map_dir.mkdir(exist_ok=True)
            map_name = f"{item['parent'].name}_{image_path.stem}"
            npy_path = map_dir / f"{map_name}_score.npy"
            png_path = map_dir / f"{map_name}_score.png"
            np.save(npy_path, prediction.astype(np.float32))
            Image.fromarray(np.clip(prediction * 255.0, 0, 255).astype(np.uint8), mode="L").save(png_path)
            item["score_npy"] = npy_path
            item["score_png"] = png_path
            log(f"[{index}/{len(selected)}] OK inference={elapsed:.2f}s sample={time.perf_counter()-sample_start:.2f}s range=[{prediction.min():.4f},{prediction.max():.4f}] map={relative(png_path)}")
            if len(inference_seconds) == 10:
                mean_first_ten = float(np.mean(inference_seconds))
                remaining_estimate = mean_first_ten * max(0, len(selected) - index)
                log(f"RUNTIME_ESTIMATE first_10_mean_inference={mean_first_ten:.2f}s remaining_samples={len(selected)-index} estimated_remaining_inference={remaining_estimate/60:.1f}min (excludes startup and visualization)")
        except Exception as exc:
            reason = f"excluded_processing_error: {type(exc).__name__}: {exc}"
            row["status"] = reason
            item["status"] = reason
            row["sample_elapsed_seconds"] = f"{time.perf_counter() - sample_start:.6f}"
            errors[type(exc).__name__ + ": " + str(exc)] += 1
            log(f"[{index}/{len(selected)}] ERROR after {time.perf_counter()-sample_start:.2f}s {relative(image_path)}: {reason}")

    aggregate: list[dict] = []
    for split in ("validation", "test"):
        split_items = [item for item in completed if item["split"] == split]
        for threshold in THRESHOLDS:
            rows = [r for r in per_image if r["split"] == split and r["threshold"] == threshold]
            if not rows:
                for averaging in ("macro_per_image", "micro_global_pixels"):
                    aggregate.append({"split": split, "threshold": threshold, "averaging": averaging,
                                      "sample_count": 0, "dice": "", "iou": "", "precision": "",
                                      "recall": "", "f1": "", "undefined_metric_sample_count": 0})
                continue
            macro = {name: float(np.mean([r[name] for r in rows]))
                     for name in ("dice", "iou", "precision", "recall", "f1")}
            aggregate.append({"split": split, "threshold": threshold, "averaging": "macro_per_image",
                              "sample_count": len(rows), **macro,
                              "undefined_metric_sample_count": sum(bool(r["undefined_metrics"]) for r in rows)})
            tp, fp, fn = (sum(r[key] for r in rows) for key in ("tp", "fp", "fn"))
            micro, micro_undefined = score_counts(tp, fp, fn)
            aggregate.append({"split": split, "threshold": threshold, "averaging": "micro_global_pixels",
                              "sample_count": len(rows), **micro,
                              "undefined_metric_sample_count": int(bool(micro_undefined))})

    validation_rows = [r for r in aggregate if r["split"] == "validation" and
                       r["averaging"] == "macro_per_image" and r["sample_count"]]
    best_threshold = None
    if validation_rows:
        # Stable tie break: choose the lowest threshold among equal validation macro Dice.
        best_threshold = max(validation_rows, key=lambda r: (r["dice"], -r["threshold"]))["threshold"]

    # Choose five lowest-performing completed samples at the validation-selected threshold.
    visual_items = []
    if best_threshold is not None:
        visual_items = sorted(completed,
                              key=lambda x: (x["metrics"][best_threshold]["dice"],
                                             relative(x["image"])))[:5]
    visual_dir = OUTPUT_DIR / "visualizations"
    visual_dir.mkdir(exist_ok=True)
    visual_paths = []
    for item in visual_items:
        score = item["metrics"][best_threshold]["dice"]
        name = item["image"].stem
        out = visual_dir / f"{item['parent'].name}_{name}_{item['split']}_comparison.png"
        with Image.open(item["image"]) as source:
            visual_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        visual_truth, _, _ = load_binary_mask(item["mask"], (visual_rgb.shape[1], visual_rgb.shape[0]))
        visual_prediction = np.load(item["score_npy"], allow_pickle=False)
        make_visual(out, visual_rgb, visual_truth, visual_prediction, best_threshold,
                    f"{relative(item['image'])} | {item['split']} | Dice {score:.4f}")
        visual_paths.append(out)

    write_csv(OUTPUT_DIR / "per_image_metrics.csv",
              ["image", "mask", "parent_folder", "split", "threshold", "dice", "iou",
               "precision", "recall", "f1", "undefined_metrics", "inference_seconds",
               "sample_elapsed_seconds", "tp", "fp", "fn", "truth_positive_pixels",
               "prediction_min", "prediction_max", "mask_resized_nearest",
               "mask_normalized_luma"], per_image)
    write_csv(OUTPUT_DIR / "aggregate_metrics.csv",
              ["split", "threshold", "averaging", "sample_count", "dice", "iou", "precision", "recall", "f1",
               "undefined_metric_sample_count"],
              aggregate)
    for row in manifest:
        if row["status"] == "selected_pending":
            row["status"] = "excluded_not_processed"
    write_csv(OUTPUT_DIR / "sample_manifest.csv",
              ["image", "mask", "parent_folder", "split", "status", "inference_seconds",
               "sample_elapsed_seconds"], manifest)

    elapsed_total = time.perf_counter() - total_start
    avg_inference = float(np.mean(inference_seconds)) if inference_seconds else 0.0
    exclusions = Counter(row["status"].split(":", 1)[0] for row in manifest
                         if row["status"].startswith("excluded"))
    lines = [
        "ManTraNet IMD2020 batch evaluation",
        f"Dataset root: {relative(DATASET_ROOT)}",
        f"Seed: {args.seed}",
        f"PyTorch CPU threads: {args.threads}",
        f"Inference longest-side limit: {args.max_side if args.max_side else 'native image resolution'} pixels",
        "When input is downscaled, RGB uses Lanczos and the score map is bilinearly upsampled to original image dimensions for pixel metrics.",
        f"Maximum samples: {args.max_samples if args.max_samples else 'all'}",
        "Pairing: same-folder exact image stem plus '_mask' (extension may differ); original and mask files excluded.",
        f"Exact pairs discovered: {discovered_count}",
        f"Pairs passing image/mask prevalidation: {len(selected)}",
        f"Selected for pilot/evaluation: {len(selected)}",
        f"Processed successfully: {len(completed)}",
        f"Processing failures: {sum(errors.values())}",
        "Exclusions by reason: " + (", ".join(f"{key}={value}" for key, value in sorted(exclusions.items())) or "none"),
        f"Distinct selected parent folders: {len({item['parent'] for item in selected})}",
        f"Processed validation samples: {sum(item['split'] == 'validation' for item in completed)}",
        f"Processed test samples: {sum(item['split'] == 'test' for item in completed)}",
        "Split: reproducible 70/30 validation/test assignment by parent folder; all images in one folder share a split.",
        "Aggregate metrics: macro_per_image is the unweighted mean of image metrics; micro_global_pixels pools TP/FP/FN.",
        "Undefined-metric rule: zero denominators are listed per image in undefined_metrics; numeric convention is 1 when both masks are empty and 0 for a zero precision/recall denominator when only one side is empty.",
        "Model output is a sigmoid-bounded score map; no probability calibration is claimed.",
        f"Best threshold selected using validation macro Dice only: {best_threshold if best_threshold is not None else 'unavailable'}",
        "Test metrics use that fixed validation-selected threshold; see aggregate_metrics.csv.",
        f"Total runtime: {elapsed_total:.3f} seconds",
        f"Average inference time per successfully processed image: {avg_inference:.3f} seconds",
        "Inference device: CPU",
        f"Prediction score maps saved: {sum('score_npy' in item for item in completed)} (NPY values plus grayscale PNGs)",
        f"Comparison visualizations saved: {len(visual_paths)}",
        f"Output paths: {relative(OUTPUT_DIR / 'per_image_metrics.csv')}; {relative(OUTPUT_DIR / 'aggregate_metrics.csv')}; {relative(OUTPUT_DIR / 'sample_manifest.csv')}; {relative(OUTPUT_DIR / 'results_manifest.json')}; {relative(OUTPUT_DIR / 'progress.log')}",
    ]
    if prevalidation_errors:
        lines.append("Prevalidation errors: " + "; ".join(f"{k} ({v})" for k, v in prevalidation_errors.items()))
    undefined_rows = [row for row in per_image if row["undefined_metrics"]]
    lines.append(f"Per-image threshold rows with at least one undefined metric: {len(undefined_rows)} / {len(per_image)}")
    for split in ("validation", "test"):
        for metric_row in aggregate:
            if (metric_row["split"] == split and metric_row["averaging"] == "macro_per_image"
                    and metric_row["sample_count"]):
                lines.append(f"{split} macro threshold={metric_row['threshold']:.1f} n={metric_row['sample_count']}: "
                             + ", ".join(f"{key}={metric_row[key]:.6f}" for key in
                                          ("dice", "iou", "precision", "recall", "f1")))
    if len({item["parent"] for item in completed}) < 2:
        lines.append("LIMITATION: fewer than two processed parent folders; independent validation/test comparison is impossible.")
    if not completed:
        lines.append("ERROR: no samples processed; inspect manifest processing statuses and error details.")
    if errors:
        lines.append("Processing error details: " + "; ".join(f"{k} ({v})" for k, v in errors.items()))
    lines.extend(f"Visualization: {relative(path)}" for path in visual_paths)
    lines.extend(f"Prediction map: {relative(item['score_npy'])}; {relative(item['score_png'])}"
                 for item in completed if "score_npy" in item)
    summary_path = OUTPUT_DIR / "evaluation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results_manifest = {
        "dataset_root": relative(DATASET_ROOT),
        "settings": {"seed": args.seed, "max_samples": args.max_samples, "threads": args.threads,
                     "max_side": args.max_side, "device": "cpu", "thresholds": list(THRESHOLDS),
                     "split_rule": "70/30 by parent folder; stable seed; validation selects threshold by macro Dice"},
        "counts": {"exact_pairs_discovered": discovered_count,
                   "prevalidated_selected": len(selected), "processed_successfully": len(completed),
                   "processing_failures": sum(errors.values()), "prevalidation_failures": sum(prevalidation_errors.values()),
                   "exclusions_by_reason": dict(exclusions),
                   "validation_processed": sum(x["split"] == "validation" for x in completed),
                   "test_processed": sum(x["split"] == "test" for x in completed)},
        "best_threshold_validation_macro_dice": best_threshold,
        "total_runtime_seconds": elapsed_total,
        "average_inference_seconds_per_success": avg_inference,
        "processing_errors": dict(errors), "prevalidation_errors": dict(prevalidation_errors),
        "samples": [{"image": relative(x["image"]), "mask": relative(x["mask"]),
                     "parent_folder": relative(x["parent"]), "split": x["split"],
                     "status": x.get("status", "not_processed"),
                     "inference_seconds": x.get("seconds"),
                     "metrics_by_threshold": {str(t): x["metrics"][t] for t in THRESHOLDS},
                     "score_npy": relative(x["score_npy"]) if "score_npy" in x else None,
                     "score_png": relative(x["score_png"]) if "score_png" in x else None}
                    for x in selected],
        "artifacts": {"per_image_metrics_csv": relative(OUTPUT_DIR / "per_image_metrics.csv"),
                      "aggregate_metrics_csv": relative(OUTPUT_DIR / "aggregate_metrics.csv"),
                      "sample_manifest_csv": relative(OUTPUT_DIR / "sample_manifest.csv"),
                      "evaluation_summary_txt": relative(summary_path),
                      "progress_log": relative(progress_path),
                      "visualizations": [relative(p) for p in visual_paths]},
    }
    (OUTPUT_DIR / "results_manifest.json").write_text(json.dumps(results_manifest, indent=2), encoding="utf-8")
    log(f"COMPLETE processed={len(completed)}/{len(selected)} runtime={elapsed_total:.1f}s summary={relative(summary_path)}")
    print("\n".join(lines))
    print(f"Output directory: {OUTPUT_DIR}")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
