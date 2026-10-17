
import csv
import random
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts" / "localization_dataset" / "manifest.csv"
TRUFOR_DIR = ROOT / "external" / "TruFor" / "TruFor_train_test"
PYTHON = ROOT / ".venv-trufor" / "Scripts" / "python.exe"
CHECKPOINT = "pretrained_models/trufor.pth.tar"

OUT = ROOT / "artifacts" / "trufor_controlled_eval_40"
STAGED = OUT / "staged"
PREDICTIONS = OUT / "predictions"

THRESHOLDS = [0.10, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
SAMPLES_PER_SPLIT = 40
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_manifest():
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    selected = {}
    rng = random.Random(SEED)

    for split in ("val", "test"):
        candidates = [r for r in rows if r["split"] == split]
        candidates.sort(
            key=lambda r: (r["source_folder"], r["source_fake_path"])
        )

        if len(candidates) < SAMPLES_PER_SPLIT:
            raise RuntimeError(f"Not enough {split} rows: {len(candidates)}")

        selected[split] = sorted(
            rng.sample(candidates, SAMPLES_PER_SPLIT),
            key=lambda r: (r["source_folder"], r["source_fake_path"]),
        )

    # Check source-folder separation between the selected splits.
    folder_splits = defaultdict(set)
    for split, items in selected.items():
        for row in items:
            folder_splits[row["source_folder"]].add(split)

    leakage = [folder for folder, splits in folder_splits.items()
               if len(splits) > 1]
    if leakage:
        raise RuntimeError(f"Source-folder leakage detected: {leakage[:10]}")

    return selected


def prepare_staging(selected):
    # Rebuild staged input folders only. Never delete prediction folders.
    if STAGED.exists():
        shutil.rmtree(STAGED)

    STAGED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    mapping = []

    for split, items in selected.items():
        image_dir = STAGED / split
        image_dir.mkdir(parents=True, exist_ok=True)

        (PREDICTIONS / split).mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(items):
            source_image = ROOT / row["source_fake_path"]
            source_mask = ROOT / row["source_mask_path"]

            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            if not source_mask.is_file():
                raise FileNotFoundError(source_mask)

            staged_name = f"{split}_{i:03d}{source_image.suffix.lower()}"
            staged_image = image_dir / staged_name
            shutil.copy2(source_image, staged_image)

            mapping.append({
                "split": split,
                "staged_name": staged_name,
                "source_folder": row["source_folder"],
                "source_fake_path": row["source_fake_path"],
                "source_mask_path": row["source_mask_path"],
                "staged_image_path": str(staged_image),
            })

    manifest_path = OUT / "sample_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0].keys()))
        writer.writeheader()
        writer.writerows(mapping)

    return mapping


def run_inference(split):
    image_dir = STAGED / split
    output_dir = PREDICTIONS / split
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    # TruFor creates <image filename>.npz, e.g. val_000.jpg.npz.
    remaining = [
        p for p in images
        if not (output_dir / f"{p.name}.npz").is_file()
    ]

    print(
        f"\n{split}: {len(images)} total; "
        f"{len(images) - len(remaining)} already predicted; "
        f"{len(remaining)} remaining.",
        flush=True,
    )

    if not remaining:
        print(f"{split}: all predictions exist; skipping inference.", flush=True)
        return

    # Use a separate directory containing only pending images.
    pending_dir = STAGED / f"{split}_pending"
    if pending_dir.exists():
        shutil.rmtree(pending_dir)
    pending_dir.mkdir(parents=True)

    for image_path in remaining:
        shutil.copy2(image_path, pending_dir / image_path.name)

    command = [
        str(PYTHON),
        "test.py",
        "-g", "-1",
        "-in", str(pending_dir),
        "-out", str(output_dir),
        "-exp", "trufor_ph3",
        "TEST.MODEL_FILE", CHECKPOINT,
    ]

    print(f"Starting {len(remaining)} pending {split} images.", flush=True)
    print(subprocess.list2cmdline(command), flush=True)

    start = time.time()
    subprocess.run(command, cwd=str(TRUFOR_DIR), check=True)

    print(
        f"{split}: inference finished in "
        f"{(time.time() - start) / 60:.1f} minutes.",
        flush=True,
    )

    # Verify that every requested prediction was produced.
    missing = [
        p.name for p in remaining
        if not (output_dir / f"{p.name}.npz").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} predictions are missing for {split}: {missing[:10]}"
        )


def score_predictions(mapping):
    detail_rows = []

    for item in mapping:
        split = item["split"]
        prediction_path = (
            PREDICTIONS / split / f"{item['staged_name']}.npz"
        )

        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing prediction: {prediction_path}. "
                "Run the script again to resume inference."
            )

        with np.load(prediction_path) as data:
            pred = np.asarray(data["map"], dtype=np.float32).squeeze()

        mask_path = ROOT / item["source_mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise RuntimeError(f"Could not read ground-truth mask: {mask_path}")

        # Dataset masks use white for manipulated pixels.
        gt = mask >= 128

        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if not np.isfinite(pred).all():
            raise RuntimeError(f"Invalid prediction values: {prediction_path}")

        for threshold in THRESHOLDS:
            binary = pred >= threshold

            tp = int(np.logical_and(binary, gt).sum())
            fp = int(np.logical_and(binary, ~gt).sum())
            fn = int(np.logical_and(~binary, gt).sum())

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            dice = (
                2 * tp / (2 * tp + fp + fn)
                if 2 * tp + fp + fn else 1.0
            )
            iou = (
                tp / (tp + fp + fn)
                if tp + fp + fn else 1.0
            )

            detail_rows.append({
                "split": split,
                "staged_name": item["staged_name"],
                "source_folder": item["source_folder"],
                "source_fake_path": item["source_fake_path"],
                "source_mask_path": item["source_mask_path"],
                "threshold": threshold,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "gt_foreground_fraction": float(gt.mean()),
                "pred_foreground_fraction": float(binary.mean()),
            })

    detail_path = OUT / "per_image_threshold_metrics.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = []

    for split in ("val", "test"):
        for threshold in THRESHOLDS:
            subset = [
                row for row in detail_rows
                if row["split"] == split and row["threshold"] == threshold
            ]

            summary.append({
                "split": split,
                "threshold": threshold,
                "image_count": len(subset),
                "macro_dice": float(np.mean([r["dice"] for r in subset])),
                "macro_iou": float(np.mean([r["iou"] for r in subset])),
                "macro_precision": float(
                    np.mean([r["precision"] for r in subset])
                ),
                "macro_recall": float(np.mean([r["recall"] for r in subset])),
                "mean_gt_foreground_fraction": float(
                    np.mean([r["gt_foreground_fraction"] for r in subset])
                ),
                "mean_pred_foreground_fraction": float(
                    np.mean([r["pred_foreground_fraction"] for r in subset])
                ),
            })

    summary_path = OUT / "threshold_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # Select threshold only from validation macro Dice.
    validation = [r for r in summary if r["split"] == "val"]
    best = max(validation, key=lambda r: r["macro_dice"])

    test_result = next(
        r for r in summary
        if r["split"] == "test" and r["threshold"] == best["threshold"]
    )

    report = [
        "TruFor controlled localization evaluation",
        f"Seed: {SEED}",
        f"Images per split: {SAMPLES_PER_SPLIT}",
        "Ground-truth convention: white pixels (>=128) are manipulated",
        f"Validation-selected threshold: {best['threshold']:.2f}",
        "",
        "split threshold count macro_dice macro_iou macro_precision "
        "macro_recall mean_gt_fg mean_pred_fg",
    ]

    for row in summary:
        report.append(
            f"{row['split']:>4} {row['threshold']:.2f} "
            f"{row['image_count']:>3} {row['macro_dice']:.4f} "
            f"{row['macro_iou']:.4f} {row['macro_precision']:.4f} "
            f"{row['macro_recall']:.4f} "
            f"{row['mean_gt_foreground_fraction']:.4f} "
            f"{row['mean_pred_foreground_fraction']:.4f}"
        )

    report.extend([
        "",
        "HELD-OUT TEST RESULT",
        f"Fixed validation-selected threshold: {best['threshold']:.2f}",
        f"Test macro Dice: {test_result['macro_dice']:.4f}",
        f"Test macro IoU: {test_result['macro_iou']:.4f}",
        f"Test macro precision: {test_result['macro_precision']:.4f}",
        f"Test macro recall: {test_result['macro_recall']:.4f}",
        "",
        "Pilot evaluation: 40 validation and 40 test images.",
        "Checkpoint provenance remains independently unconfirmed.",
    ])

    report_path = OUT / "evaluation_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "\n".join(report), flush=True)
    print(f"\nReport saved: {report_path}", flush=True)
    print(f"Detailed metrics: {detail_path}", flush=True)
    print(f"Threshold summary: {summary_path}", flush=True)


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    if not (TRUFOR_DIR / "test.py").is_file():
        raise FileNotFoundError(TRUFOR_DIR / "test.py")

    selected = read_manifest()
    print(
        "Selected image counts:",
        {split: len(rows) for split, rows in selected.items()},
        flush=True,
    )

    mapping = prepare_staging(selected)

    # Existing predictions are preserved and reused.
    run_inference("val")
    run_inference("test")
    score_predictions(mapping)


if __name__ == "__main__":
    main()
