"""Build the derived localization dataset (no training, source trees untouched).

Example:
    python scripts/prepare_localization_dataset.py
    python scripts/prepare_localization_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.localization.prepare_dataset import (
    LocalizationPrepConfig,
    LocalizationPrepError,
    default_config,
    format_audit_report,
    prepare_localization_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = default_config(ROOT)
    parser = argparse.ArgumentParser(
        description="Prepare 256x256 fake/mask pairs for localization (derived dataset only)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=defaults.source_dir,
        help="IMD2020 sample-folder root (read-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_dir,
        help="Derived output directory (never inside dataset 16.09 or dataset/).",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing derived output directory. Never deletes source data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    config = LocalizationPrepConfig(
        project_root=ROOT,
        source_dir=args.source,
        output_dir=args.output,
        image_size=args.size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        mask_threshold=args.mask_threshold,
        overwrite=bool(args.overwrite),
    )
    try:
        report = prepare_localization_dataset(config)
    except LocalizationPrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(format_audit_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
