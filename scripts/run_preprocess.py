#!/usr/bin/env python
"""
CLI entry-point for the CIC-IDS2017 preprocessing pipeline.

Usage
-----
  python scripts/run_preprocess.py                   # uses config.yaml
  python scripts/run_preprocess.py --config my.yaml  # custom config
  python scripts/run_preprocess.py --task binary      # override task mode
  python scripts/run_preprocess.py --split random     # override split strategy
  python scripts/run_preprocess.py --scaler standard  # override scaler
  python scripts/run_preprocess.py --resample smote   # enable + set resampling

Run from the project root directory.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Ensure project root is on sys.path so ``pipeline`` can be imported.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.build_dataset import load_config, main  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIC-IDS2017 preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    p.add_argument(
        "--task",
        type=str,
        choices=["binary", "multiclass", "both"],
        default=None,
        help="Override labels.task_mode in config.",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["random", "day_based", "time_based"],
        default=None,
        help="Override split.strategy in config.",
    )
    p.add_argument(
        "--scaler",
        type=str,
        choices=["robust", "standard", "minmax", "none"],
        default=None,
        help="Override transform.scaler in config.",
    )
    p.add_argument(
        "--resample",
        type=str,
        choices=["smote", "adasyn", "smote_tomek", "smote_enn",
                 "random_over", "random_under"],
        default=None,
        help="Enable imbalance resampling with the given method.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random_seed.",
    )
    return p.parse_args()


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Patch *cfg* with CLI overrides (if any)."""
    if args.task is not None:
        cfg.setdefault("labels", {})["task_mode"] = args.task
    if args.split is not None:
        cfg.setdefault("split", {})["strategy"] = args.split
    if args.scaler is not None:
        cfg.setdefault("transform", {})["scaler"] = args.scaler
    if args.resample is not None:
        cfg.setdefault("imbalance", {})["enabled"] = True
        cfg["imbalance"]["method"] = args.resample
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    return cfg


def cli_main() -> None:
    """Parse CLI args, patch config, and run the pipeline."""
    args = _parse_args()

    import yaml

    config_path = pathlib.Path(args.config)
    cfg = load_config(config_path)
    cfg = _apply_overrides(cfg, args)

    patched_path = pathlib.Path("config_patched.yaml")
    with open(patched_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False)

    main(config_path=patched_path)

    if patched_path.exists():
        patched_path.unlink()


if __name__ == "__main__":
    cli_main()
