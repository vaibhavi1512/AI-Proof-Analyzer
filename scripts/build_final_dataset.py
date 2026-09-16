"""Build the final IMD2020-only MAYA dataset with case-level isolation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
IMD = ROOT / "IMD2020"
LEGACY_HOLDOUT = ROOT / "dataset" / "independent_imd2020_v1"
RAW = ROOT / "dataset" / "raw"
OUTPUT = ROOT / "dataset" / "final"
EXTERNAL_OUTPUT = ROOT / "dataset" / "final_external"
SPLITS = ("train", "validation", "test")
SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FIELDS = [
    "source_path",
    "source_dataset",
    "label",
    "case_id",
    "width",
    "height",
    "aspect_ratio",
    "file_size",
    "sha256",
    "split",
    "original_filename",
    "output_path",
]
LEGACY_FAKE_CASES = {
    "z2", "z3", "z5", "z6", "z7", "z8", "z9",
    "z10", "z11", "z12", "z21", "z23", "z24", "z25",
}


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    case_id: str
    width: int
    height: int
    file_size: int
    sha256: str


def image_files(folder: Path) -> tuple[list[Path], list[dict[str, str]]]:
    files: list[Path] = []
    exclusions: list[dict[str, str]] = []
    for path in sorted(folder.rglob("*")) if folder.is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            exclusions.append(
                {
                    "source_path": str(path.relative_to(ROOT)),
                    "reason": f"unreadable:{type(exc).__name__}",
                }
            )
            continue
        files.append(path)
    return files, exclusions


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def make_record(path: Path, label: str, case_id: str) -> Record:
    with Image.open(path) as image:
        width, height = image.width, image.height
    return Record(
        path=path,
        label=label,
        case_id=case_id,
        width=width,
        height=height,
        file_size=path.stat().st_size,
        sha256=digest(path),
    )


def legacy_hashes() -> set[str]:
    return {digest(path) for path in image_files(LEGACY_HOLDOUT)[0]}


def inventory(
    exclusions: list[dict[str, str]],
    legacy: set[str],
) -> tuple[list[Record], dict[str, list[Record]], dict[str, list[Record]]]:
    if not IMD.is_dir():
        raise FileNotFoundError(f"IMD2020 folder not found: {IMD}")
    records: list[Record] = []
    by_case: dict[str, list[Record]] = defaultdict(list)
    by_hash: dict[str, list[Record]] = defaultdict(list)
    for case in sorted(path for path in IMD.iterdir() if path.is_dir()):
        paths, skipped = image_files(case)
        exclusions.extend(skipped)
        masks = [path for path in paths if "_mask." in path.name.lower()]
        exclusions.extend(
            {
                "source_path": str(path.relative_to(ROOT)),
                "reason": "mask_excluded",
            }
            for path in masks
        )
        images = [path for path in paths if path not in masks]
        originals = [path for path in images if path.stem.lower().endswith("_orig")]
        if len(originals) != 1:
            exclusions.extend(
                {
                    "source_path": str(path.relative_to(ROOT)),
                    "reason": f"incomplete_case:{case.name}",
                }
                for path in images
            )
            continue
        case_id = f"imd2020:{case.name}"
        case_records = [make_record(originals[0], "REAL", case_id)]
        case_records.extend(
            make_record(path, "FAKE", case_id)
            for path in images
            if path not in originals
        )
        for record in case_records:
            records.append(record)
            by_case[case.name].append(record)
            by_hash[record.sha256].append(record)
            if record.sha256 in legacy:
                exclusions.append(
                    {
                        "source_path": str(record.path.relative_to(ROOT)),
                        "reason": "legacy_holdout_reserved_overlap_documented_not_excluded",
                    }
                )
    return records, by_case, by_hash


def connected_components(
    cases: dict[str, list[Record]],
    by_hash: dict[str, list[Record]],
) -> list[list[str]]:
    parent = {case: case for case in cases}

    def find(case: str) -> str:
        while parent[case] != case:
            parent[case] = parent[parent[case]]
            case = parent[case]
        return case

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for records in by_hash.values():
        names = sorted({record.case_id.removeprefix("imd2020:") for record in records})
        for name in names[1:]:
            union(names[0], name)
    groups: dict[str, list[str]] = defaultdict(list)
    for case in sorted(cases):
        groups[find(case)].append(case)
    return sorted(groups.values(), key=lambda group: tuple(group))


def canonical_records(
    by_hash: dict[str, list[Record]],
    exclusions: list[dict[str, str]],
) -> list[Record]:
    chosen: list[Record] = []
    for sha, records in sorted(by_hash.items()):
        selected = min(records, key=lambda record: str(record.path))
        chosen.append(selected)
        for record in records:
            if record.path != selected.path:
                exclusions.append(
                    {
                        "source_path": str(record.path.relative_to(ROOT)),
                        "reason": "exact_duplicate_excluded",
                    }
                )
    return chosen


def select_external(
    cases: dict[str, list[Record]],
    components: list[list[str]],
    exclusions: list[dict[str, str]],
) -> set[str]:
    component_by_case = {
        case: component for component in components for case in component
    }
    eligible = [
        case for case in sorted(cases)
        if sum(record.label == "FAKE" for record in cases[case]) == 1
        and case not in LEGACY_FAKE_CASES
        and len(component_by_case[case]) == 1
    ]
    external = set(eligible[:62])
    if len(external) != 62:
        raise RuntimeError(f"Expected 62 external cases, found {len(external)}")
    for case in sorted(external):
        for record in cases[case]:
            exclusions.append(
                {
                    "source_path": str(record.path.relative_to(ROOT)),
                    "reason": "external_case",
                }
            )
    return external


def assign_primary(
    components: list[list[str]],
    cases: dict[str, list[Record]],
    external: set[str],
) -> dict[str, str]:
    remaining = [
        component for component in components
        if not set(component) & external
    ]
    targets = {"train": 246, "validation": 52, "test": 54}
    if sum(len(component) for component in remaining) != sum(targets.values()):
        raise RuntimeError("Primary case count does not equal 352")
    counts = {split: 0 for split in SPLITS}
    fake_counts = {split: 0 for split in SPLITS}
    rng = random.Random(SEED)
    rng.shuffle(remaining)
    remaining.sort(
        key=lambda component: (
            -sum(
                record.label == "FAKE"
                for case in component
                for record in cases[case]
            ),
            tuple(component),
        )
    )
    assignments: dict[str, str] = {}
    for component in remaining:
        fake_count = sum(
            record.label == "FAKE"
            for case in component
            for record in cases[case]
        )
        available = [
            split for split in SPLITS
            if counts[split] + len(component) <= targets[split]
        ]
        if not available:
            raise RuntimeError(f"Unable to place component {component}")
        chosen = min(
            available,
            key=lambda split: (
                fake_counts[split] / max(counts[split], 1),
                counts[split] / targets[split],
                split,
            ),
        )
        for case in component:
            assignments[case] = chosen
        counts[chosen] += len(component)
        fake_counts[chosen] += fake_count
    if counts != targets:
        raise RuntimeError(f"Unexpected case split counts: {counts}")
    return assignments


def save_processed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        scale = 224 / min(rgb.width, rgb.height)
        resized = rgb.resize(
            (round(rgb.width * scale), round(rgb.height * scale)),
            Image.Resampling.BILINEAR,
        )
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        resized.crop((left, top, left + 224, top + 224)).save(
            destination, "JPEG", quality=95, optimize=True
        )


def manifest_row(
    record: Record,
    split: str,
    output: Path,
) -> dict[str, str]:
    return {
        "source_path": str(record.path.relative_to(ROOT)),
        "source_dataset": "IMD2020",
        "label": record.label,
        "case_id": record.case_id,
        "width": str(record.width),
        "height": str(record.height),
        "aspect_ratio": f"{record.width / record.height:.8f}",
        "file_size": str(record.file_size),
        "sha256": record.sha256,
        "split": split,
        "original_filename": record.path.name,
        "output_path": str(output.relative_to(ROOT)),
    }


def materialize(
    records: list[Record],
    assignments: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in sorted(records, key=lambda item: (assignments[item.case_id.removeprefix("imd2020:")], item.sha256)):
        split = assignments[record.case_id.removeprefix("imd2020:")]
        output = OUTPUT / split / record.label / f"{record.sha256[:20]}.jpg"
        save_processed(record.path, output)
        rows.append(manifest_row(record, split, output))
    return rows


def materialize_external(
    records: list[Record],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in sorted(records, key=lambda item: (item.label, item.sha256)):
        output = EXTERNAL_OUTPUT / record.label / f"{record.sha256[:20]}.jpg"
        save_processed(record.path, output)
        rows.append(manifest_row(record, "external", output))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geometry(rows: list[dict[str, str]]) -> dict[str, float | int]:
    if not rows:
        return {"count": 0}
    widths = [int(row["width"]) for row in rows]
    heights = [int(row["height"]) for row in rows]
    aspects = [float(row["aspect_ratio"]) for row in rows]
    sizes = [int(row["file_size"]) for row in rows]
    return {
        "count": len(rows),
        "width_min": min(widths),
        "width_max": max(widths),
        "width_median": statistics.median(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "height_median": statistics.median(heights),
        "aspect_ratio_min": min(aspects),
        "aspect_ratio_max": max(aspects),
        "aspect_ratio_median": statistics.median(aspects),
        "file_size_min": min(sizes),
        "file_size_max": max(sizes),
        "file_size_median": statistics.median(sizes),
    }


def audit(
    primary: list[dict[str, str]],
    external: list[dict[str, str]],
    components: list[list[str]],
    legacy: set[str],
) -> dict[str, int]:
    partitions = primary + external
    hashes: dict[str, set[str]] = defaultdict(set)
    cases: dict[str, set[str]] = defaultdict(set)
    component_lookup = {
        f"imd2020:{case}": index
        for index, component in enumerate(components)
        for case in component
    }
    component_splits: dict[int, set[str]] = defaultdict(set)
    for row in partitions:
        hashes[row["sha256"]].add(row["split"])
        cases[row["case_id"]].add(row["split"])
        component_splits[component_lookup[row["case_id"]]].add(row["split"])
    primary_hashes = {row["sha256"] for row in primary}
    external_hashes = {row["sha256"] for row in external}
    missing = sum(
        not (ROOT / row["output_path"]).is_file() for row in partitions
    )
    return {
        "sha_cross_partition": sum(len(value) > 1 for value in hashes.values()),
        "case_cross_partition": sum(len(value) > 1 for value in cases.values()),
        "duplicate_component_cross_partition": sum(
            len(value) > 1 for value in component_splits.values()
        ),
        "external_primary_sha_overlap": len(primary_hashes & external_hashes),
        "primary_legacy_holdout_sha_overlap": len(primary_hashes & legacy),
        "external_legacy_holdout_sha_overlap": len(external_hashes & legacy),
        "masks_materialized_as_classification": sum(
            "_mask." in row["original_filename"].lower() for row in partitions
        ),
        "missing_materialized_outputs": missing,
        "manifest_file_count_mismatch": int(
            len(primary) != sum(1 for _ in (OUTPUT / "manifest.csv").open())
            - 1
            or len(external) != sum(1 for _ in (EXTERNAL_OUTPUT / "manifest.csv").open())
            - 1
        ),
    }


def build_summary(
    rows: list[dict[str, str]],
    exclusions: list[dict[str, str]],
    audit_result: dict[str, int],
    kind: str,
) -> dict:
    return {
        "dataset": kind,
        "seed": SEED,
        "source_proportions": {"IMD2020": 1.0},
        "counts_by_split_label": {
            split: dict(Counter(
                row["label"] for row in rows if row["split"] == split
            ))
            for split in (SPLITS if kind == "primary" else ("external",))
        },
        "cases_by_split": {
            split: len({
                row["case_id"] for row in rows if row["split"] == split
            })
            for split in (SPLITS if kind == "primary" else ("external",))
        },
        "images": len(rows),
        "geometry": {
            split: geometry([
                row for row in rows if row["split"] == split
            ])
            for split in (SPLITS if kind == "primary" else ("external",))
        },
        "exclusions_by_reason": dict(
            Counter(item["reason"] for item in exclusions)
        ),
        "verification": audit_result,
        "limitations": [
            "The legacy independent_imd2020_v1 holdout remains untouched and is not used as the new external evaluation set.",
            "The same IMD2020 REAL originals may overlap the legacy holdout by SHA because the new experiment is case-level isolated from its own partitions, not independent of that legacy artifact.",
            "The external set has exactly one manipulated image per case, so it evaluates unseen-case generalization but not the full manipulation-count distribution.",
            "RAW is intentionally excluded from this primary dataset and remains a separate future domain-shift evaluation resource.",
        ],
    }


def main() -> int:
    if OUTPUT.exists() or EXTERNAL_OUTPUT.exists():
        raise FileExistsError(
            "Refusing to overwrite dataset/final or dataset/final_external."
        )
    exclusions: list[dict[str, str]] = []
    legacy = legacy_hashes()
    records, cases, by_hash = inventory(exclusions, legacy)
    components = connected_components(cases, by_hash)
    external_cases = select_external(cases, components, exclusions)
    canonical = canonical_records(by_hash, exclusions)
    canonical_by_case = defaultdict(list)
    for record in canonical:
        canonical_by_case[record.case_id.removeprefix("imd2020:")].append(record)
    assignments = assign_primary(components, canonical_by_case, external_cases)
    primary_records = [
        record for record in canonical
        if record.case_id.removeprefix("imd2020:") not in external_cases
    ]
    external_records = [
        record for record in canonical
        if record.case_id.removeprefix("imd2020:") in external_cases
    ]
    for record in external_records:
        exclusions.append(
            {
                "source_path": str(record.path.relative_to(ROOT)),
                "reason": "external_case",
            }
        )
    raw_hashes = {digest(path) for path in image_files(RAW)[0]}
    raw_overlap = sum(record.sha256 in raw_hashes for record in records)
    exclusions.append(
        {
            "source_path": "dataset/raw",
            "reason": f"imd2020_raw_sha_overlap_audit:{raw_overlap}",
        }
    )
    OUTPUT.mkdir(parents=True)
    EXTERNAL_OUTPUT.mkdir(parents=True)
    primary_rows = materialize(primary_records, assignments)
    external_rows = materialize_external(external_records)
    write_csv(OUTPUT / "manifest.csv", primary_rows, FIELDS)
    write_csv(EXTERNAL_OUTPUT / "manifest.csv", external_rows, FIELDS)
    write_csv(OUTPUT / "exclusions.csv", exclusions, ["source_path", "reason"])
    audit_result = audit(primary_rows, external_rows, components, legacy)
    blocking_audit_keys = {
        "sha_cross_partition",
        "case_cross_partition",
        "duplicate_component_cross_partition",
        "external_primary_sha_overlap",
        "masks_materialized_as_classification",
        "missing_materialized_outputs",
        "manifest_file_count_mismatch",
    }
    if any(audit_result[key] for key in blocking_audit_keys):
        raise RuntimeError(f"Leakage audit failed: {audit_result}")
    primary_summary = build_summary(primary_rows, exclusions, audit_result, "primary")
    external_summary = build_summary(external_rows, exclusions, audit_result, "external")
    (OUTPUT / "dataset_summary.json").write_text(
        json.dumps(primary_summary, indent=2), encoding="utf-8"
    )
    (EXTERNAL_OUTPUT / "dataset_summary.json").write_text(
        json.dumps(external_summary, indent=2), encoding="utf-8"
    )
    report = {
        "primary": primary_summary["counts_by_split_label"],
        "primary_cases": primary_summary["cases_by_split"],
        "primary_images": len(primary_rows),
        "external": external_summary["counts_by_split_label"],
        "external_cases": external_summary["cases_by_split"],
        "external_images": len(external_rows),
        "leakage": audit_result,
        "raw_overlap_sha_count": raw_overlap,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
