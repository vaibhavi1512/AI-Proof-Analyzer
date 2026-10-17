
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "trufor_controlled_eval_40"
PRED_DIR = OUT / "predictions" / "val"
MANIFEST = OUT / "sample_manifest.csv"

THRESHOLDS = [0.10, 0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]

with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
    manifest = list(csv.DictReader(f))

per_image = []
errors = []
missing = []

for row in manifest:
    if row["split"] != "val":
        continue

    pred_path = PRED_DIR / (row["staged_name"] + ".npz")
    mask_path = ROOT / row["source_mask_path"]

    if not pred_path.is_file():
        missing.append(row["staged_name"])
        continue

    try:
        with np.load(pred_path) as data:
            pred = np.asarray(data["map"], dtype=np.float32).squeeze()

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        gt = mask >= 128

        if pred.shape != gt.shape:
            pred = cv2.resize(
                pred,
                (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if not np.isfinite(pred).all():
            raise ValueError("Prediction contains non-finite values")

        for threshold in THRESHOLDS:
            binary = pred >= threshold
            tp = int(np.logical_and(binary, gt).sum())
            fp = int(np.logical_and(binary, ~gt).sum())
            fn = int(np.logical_and(~binary, gt).sum())

            dice = (
                2 * tp / (2 * tp + fp + fn)
                if 2 * tp + fp + fn else 1.0
            )
            iou = (
                tp / (tp + fp + fn)
                if tp + fp + fn else 1.0
            )
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            per_image.append({
                "image": row["staged_name"],
                "source_folder": row["source_folder"],
                "threshold": threshold,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
            })

    except Exception as exc:
        errors.append((row["staged_name"], str(exc)))

detail_path = OUT / "preliminary_validation_per_image.csv"
with detail_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "image", "source_folder", "threshold",
        "dice", "iou", "precision", "recall"
    ])
    writer.writeheader()
    writer.writerows(per_image)

summary = []
for threshold in THRESHOLDS:
    rows = [r for r in per_image if r["threshold"] == threshold]
    if not rows:
        continue

    summary.append({
        "threshold": threshold,
        "images": len(rows),
        "macro_dice": np.mean([r["dice"] for r in rows]),
        "macro_iou": np.mean([r["iou"] for r in rows]),
        "macro_precision": np.mean([r["precision"] for r in rows]),
        "macro_recall": np.mean([r["recall"] for r in rows]),
    })

summary_path = OUT / "preliminary_validation_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "threshold", "images", "macro_dice", "macro_iou",
        "macro_precision", "macro_recall"
    ])
    writer.writeheader()
    writer.writerows(summary)

best = max(summary, key=lambda r: r["macro_dice"]) if summary else None

report = [
    "TruFor — Preliminary Validation Evaluation",
    "",
    "Completed validation predictions: 28 expected",
    f"Successfully evaluated: {len(set(r['image'] for r in per_image))}",
    f"Missing predictions: {len(missing)}",
    f"Evaluation errors: {len(errors)}",
    "",
    "Results are preliminary and must not be reported as final test performance.",
]

if best:
    report.extend([
        f"Best observed threshold: {best['threshold']:.2f}",
        f"Macro Dice: {best['macro_dice']:.4f}",
        f"Macro IoU: {best['macro_iou']:.4f}",
        f"Macro precision: {best['macro_precision']:.4f}",
        f"Macro recall: {best['macro_recall']:.4f}",
    ])

report.extend([
    "",
    "Missing predictions: " + ", ".join(missing),
    "Errors: " + repr(errors),
    "",
    "Held-out test evaluation: pending",
    "Full validation evaluation: pending",
    "Checkpoint provenance: not independently confirmed",
])

report_path = OUT / "PRELIMINARY_RESULTS.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("\nSaved files:")
print(detail_path)
print(summary_path)
print(report_path)
