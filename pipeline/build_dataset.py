"""
Pipeline orchestrator — ties preprocess → split → impute → scale → save
into a single reproducible run.

Stage order:
  1. Ingest raw CSVs
  2. Clean (dedup BEFORE drop; inf→NaN only)
  3. Normalise labels, create targets
  4. Split (day-based by default)
  5. Impute NaN (fitted on TRAIN only)
  6. Scale features (fitted on TRAIN only)
  7. Resample (optional, train only)
  8. Class weights
  9. Anomaly-detection set (optional)
  10. Save artefacts
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Union

import joblib
import numpy as np
import yaml

from pipeline.preprocess import (
    clean_dataframe,
    create_binary_labels,
    create_multiclass_labels,
    load_all_csvs,
    normalise_labels,
    print_label_distribution,
)
from pipeline.split_transform import (
    build_anomaly_detection_set,
    compute_class_weights,
    fit_and_transform,
    fit_imputer_and_transform,
    resample_training_data,
    split_dataset,
)
from pipeline.utils import get_logger, save_dataframe, setup_logging

logger = get_logger(__name__)


def load_config(config_path: Union[str, pathlib.Path] = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = pathlib.Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded configuration from %s", config_path)
    return cfg


def _write_summary(lines: list[str], output_dir: pathlib.Path) -> None:
    """Persist summary report."""
    path = output_dir / "summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Summary written to %s", path)


def main(config_path: Union[str, pathlib.Path] = "config.yaml") -> None:
    """Execute the full preprocessing pipeline."""
    t0 = time.time()
    cfg = load_config(config_path)

    raw_dir = pathlib.Path(cfg["paths"]["raw_data_dir"])
    out_dir = pathlib.Path(cfg["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = pathlib.Path(cfg["paths"].get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_file = cfg["paths"].get("log_file")
    setup_logging(log_file=log_file)

    seed = cfg.get("random_seed", 42)
    np.random.seed(seed)

    summary: list[str] = ["CIC-IDS2017 Preprocessing Summary", "=" * 50]

    # ── 1. INGEST ──
    logger.info("▶ Stage 1: Ingest")
    ingest_cfg = cfg.get("ingest", {})
    df = load_all_csvs(
        raw_dir,
        encoding=ingest_cfg.get("encoding", "utf-8"),
        fallback_encoding=ingest_cfg.get("fallback_encoding", "latin-1"),
        chunk_size=ingest_cfg.get("chunk_size"),
    )
    summary.append(f"Rows after ingest: {len(df)}")
    summary.append(f"Columns after ingest: {df.shape[1]}")

    # ── 2. CLEAN ──
    logger.info("▶ Stage 2: Clean")
    clean_cfg = cfg.get("cleaning", {})
    df = clean_dataframe(
        df,
        columns_to_drop=clean_cfg.get("columns_to_drop", []),
        inf_strategy=clean_cfg.get("replace_inf_with", "nan"),
        deduplicate=clean_cfg.get("remove_duplicates", True),
        do_clip_outliers=clean_cfg.get("clip_outliers", False),
        clip_factor=clean_cfg.get("clip_factor", 3.0),
    )
    summary.append(f"Rows after cleaning: {len(df)}")
    summary.append(f"Columns after cleaning: {df.shape[1]}")

    # ── 3. LABELS ──
    logger.info("▶ Stage 3: Labels")
    label_cfg = cfg.get("labels", {})
    label_col = label_cfg.get("label_column", "label")
    task_mode = label_cfg.get("task_mode", "binary")

    df = normalise_labels(df, label_column=label_col)
    print_label_distribution(df, column=label_col, title="After normalisation")

    target_col: str
    label_map = None

    if task_mode in ("binary", "both"):
        df = create_binary_labels(df, label_column=label_col)
    if task_mode in ("multiclass", "both"):
        df, label_map = create_multiclass_labels(df, label_column=label_col)

    if task_mode == "binary":
        target_col = "label_binary"
    elif task_mode == "multiclass":
        target_col = "label_multi"
    else:
        target_col = "label_multi"

    if label_col in df.columns and label_col != target_col:
        cols_to_keep = {"source_file", target_col, "label_binary", "label_multi"}
        if label_col not in cols_to_keep:
            df.drop(columns=[label_col], inplace=True)

    summary.append(f"Task mode: {task_mode}")
    summary.append(f"Target column: {target_col}")
    if label_map:
        summary.append(f"Label map: {dict(label_map)}")

    # ── 4. SPLIT ──
    logger.info("▶ Stage 4: Split")
    split_cfg = cfg.get("split", {})

    extra_target = None
    if task_mode == "both" and "label_binary" in df.columns:
        extra_target = df["label_binary"].copy()
        df.drop(columns=["label_binary"], inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df,
        target_col=target_col,
        strategy=split_cfg.get("strategy", "day_based"),
        test_size=split_cfg.get("test_size", 0.20),
        val_size=split_cfg.get("val_size", 0.10),
        seed=seed,
        day_assignment=split_cfg.get("day_assignment"),
    )
    summary.append(
        f"Split ({split_cfg.get('strategy', 'day_based')}): "
        f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}"
    )

    # ── 5. IMPUTE (fitted on train only) ──
    logger.info("▶ Stage 5: Impute NaN (fitted on train only)")
    nan_strategy = clean_cfg.get("nan_strategy", "median")
    X_train, X_val, X_test = fit_imputer_and_transform(
        X_train, X_val, X_test,
        strategy=nan_strategy,
        output_dir=out_dir,
    )
    summary.append(f"Imputation: {nan_strategy} (fitted on train only)")

    # ── 6. SCALE (fitted on train only) ──
    logger.info("▶ Stage 6: Scale")
    transform_cfg = cfg.get("transform", {})
    save_fmt = transform_cfg.get("save_format", "parquet")

    X_train, X_val, X_test = fit_and_transform(
        X_train, X_val, X_test,
        scaler_name=transform_cfg.get("scaler", "robust"),
        output_dir=out_dir,
    )

    # ── 7. RESAMPLE (optional, train only) ──
    imb_cfg = cfg.get("imbalance", {})
    if imb_cfg.get("enabled", False):
        logger.info("▶ Stage 7: Resample")
        X_train, y_train = resample_training_data(
            X_train, y_train,
            method=imb_cfg.get("method", "smote"),
            min_class_samples=imb_cfg.get("min_class_samples", 6),
            k_neighbors=imb_cfg.get("smote_k_neighbors", 5),
            sampling_strategy=imb_cfg.get("sampling_strategy", "auto"),
            seed=seed,
        )
        summary.append(f"Resampling ({imb_cfg.get('method')}): train={len(X_train)}")
    else:
        logger.info("▶ Stage 7: Resample — SKIPPED (disabled)")

    # ── 8. CLASS WEIGHTS ──
    cw_cfg = cfg.get("class_weights", {})
    if cw_cfg.get("enabled", True):
        logger.info("▶ Stage 8: Class weights")
        weights = compute_class_weights(y_train, mode=cw_cfg.get("mode", "balanced"))
        joblib.dump(weights, out_dir / "class_weights.joblib")
        summary.append(f"Class weights: {weights}")

    # ── 9. ANOMALY-DETECTION SET ──
    ad_cfg = cfg.get("anomaly_detection", {})
    if ad_cfg.get("enabled", False):
        logger.info("▶ Stage 9: Anomaly-detection set")
        X_benign = build_anomaly_detection_set(
            X_train, y_train, benign_label=ad_cfg.get("benign_label", 0)
        )
        save_dataframe(X_benign, out_dir / "X_train_benign", fmt=save_fmt)
        summary.append(f"Anomaly-detection set: {len(X_benign)} benign rows")

    # ── 10. SAVE ──
    logger.info("▶ Stage 10: Saving artefacts")
    save_dataframe(X_train, out_dir / "X_train", fmt=save_fmt)
    save_dataframe(X_val,   out_dir / "X_val",   fmt=save_fmt)
    save_dataframe(X_test,  out_dir / "X_test",  fmt=save_fmt)

    y_train.to_frame().reset_index(drop=True).to_parquet(out_dir / "y_train.parquet", index=False)
    y_val.to_frame().reset_index(drop=True).to_parquet(out_dir / "y_val.parquet",     index=False)
    y_test.to_frame().reset_index(drop=True).to_parquet(out_dir / "y_test.parquet",   index=False)

    if label_map is not None:
        joblib.dump(dict(label_map), out_dir / "label_map.joblib")

    if cfg.get("output", {}).get("save_csv_copy", False):
        save_dataframe(X_train, out_dir / "X_train", fmt="csv")
        save_dataframe(X_val,   out_dir / "X_val",   fmt="csv")
        save_dataframe(X_test,  out_dir / "X_test",  fmt="csv")

    # ── SUMMARY ──
    elapsed = time.time() - t0
    summary.append(f"\nTotal time: {elapsed:.1f}s")
    for line in summary:
        logger.info(line)

    if cfg.get("output", {}).get("summary_to_file", True):
        _write_summary(summary, reports_dir)

    logger.info("✅ Pipeline complete. Artefacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
