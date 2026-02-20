#!/usr/bin/env python
"""
Unified CLI for training and evaluating a single model.

Usage
-----
    python scripts/train_model.py --model lr  --task binary
    python scripts/train_model.py --model rf  --task binary
    python scripts/train_model.py --model xgb --task multiclass
    python scripts/train_model.py --model if  --task binary       # Isolation Forest

Extra options
-------------
    # Fixed threshold (no tuning) - DEFAULT
    python scripts/train_model.py --model lr --task binary --threshold 0.5

    # Enable internal threshold tuning (binary only, not for IF)
    python scripts/train_model.py --model lr --task binary --tune-threshold --tune-size 0.10

Outputs
-------
    models/<task>_<model>.joblib
    reports/metrics/<task>_<model>_test.json
    reports/figures/<task>_<model>_test_confusion.png
    reports/figures/<task>_<model>_test_roc.png        (binary only)
    reports/figures/<task>_<model>_test_pr.png         (binary only)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is importable
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.baselines import MODELS  # noqa: E402
from pipeline.evaluate import evaluate_model  # noqa: E402
from pipeline.utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single model",
    )
    parser.add_argument(
        "--model", required=True, choices=list(MODELS.keys()),
        help="Model key: lr, rf, xgb, if",
    )
    parser.add_argument(
        "--task", required=True, choices=["binary", "multiclass"],
        help="Classification task: binary or multiclass",
    )
    parser.add_argument(
        "--processed-dir", type=str, default="data/processed",
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--reports-dir", type=str, default="reports",
        help="Directory to save reports/figures/metrics",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )

    # Threshold control
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed threshold for binary classification when tuning is disabled (default: 0.5).",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Enable threshold tuning on an internal split from the training set (binary only, not for IF).",
    )
    parser.add_argument(
        "--tune-size",
        type=float,
        default=0.10,
        help="Fraction of the training set reserved for threshold tuning (default: 0.10).",
    )

    args = parser.parse_args()

    setup_logging(log_file=str(pathlib.Path(args.reports_dir) / "training.log"))

    proc = pathlib.Path(args.processed_dir)
    models_dir = pathlib.Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_key = args.model
    task = args.task
    display_name, trainer = MODELS[model_key]
    run_name = f"{task}_{model_key}"

    logger.info("=" * 60)
    logger.info("MODEL: %s  |  TASK: %s  |  RUN: %s", display_name, task, run_name)
    logger.info("=" * 60)

    # ── Load data (Train & Test only) ────────────────────────────────
    logger.info("Loading data from %s ...", proc)
    X_train = pd.read_parquet(proc / "X_train.parquet")
    X_test = pd.read_parquet(proc / "X_test.parquet")
    y_train = pd.read_parquet(proc / "y_train.parquet").iloc[:, 0]
    y_test = pd.read_parquet(proc / "y_test.parquet").iloc[:, 0]
    
    # We do NOT load X_val/y_val as per simplified protocol.

    class_weights = joblib.load(proc / "class_weights.joblib")

    label_map = None
    if task == "multiclass":
        # Only load label_map for multiclass mode
        label_map_path = proc / "label_map.joblib"
        if not label_map_path.exists():
            raise FileNotFoundError(
                f"Missing label_map.joblib for multiclass task. Expected at: {label_map_path}"
            )
        label_map = joblib.load(label_map_path)

    logger.info("  Train (Mon-Thu): %d  |  Test (Fri): %d", len(y_train), len(y_test))

    # ── Derive labels based on task ──────────────────────────────────
    if task == "binary":
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
        class_names = ["BENIGN", "Attack"]
        labels = [0, 1]

        # Recompute binary class weights from y_train for safety
        from sklearn.utils.class_weight import compute_class_weight
        cw_array = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        class_weights = {0: float(cw_array[0]), 1: float(cw_array[1])}

        logger.info("  Binary: BENIGN=%d, Attack=%d",
                    int((y_train == 0).sum()), int((y_train == 1).sum()))
    else:
        # Multiclass: use label_map for class names
        inv_map = {v: k for k, v in (label_map or {}).items()}
        all_labels = sorted(set(y_train.unique()) | set(y_test.unique()))
        class_names = [inv_map.get(int(i), f"class_{int(i)}") for i in all_labels]
        labels = all_labels
        logger.info("  Multiclass: %d unique classes across splits", len(all_labels))

    # ── Train ────────────────────────────────────────────────────────
    logger.info("Training %s ...", display_name)
    t0 = time.time()

    # Threshold tuning is optional and only relevant for supervised binary classifiers
    do_threshold_tuning = bool(args.tune_threshold and task == "binary" and model_key != "if")
    chosen_threshold = float(args.threshold)
    threshold_report = None

    if do_threshold_tuning:
        logger.info("  Threshold Tuning ENABLED (Internal Split: %.0f%% set aside).", args.tune_size * 100)
        # Internal tuning split to avoid tuning on Test data
        # We train on 'fit' set, tune on 'tune' set, and use the 'fit'-trained model for final inference.
        X_fit, X_tune, y_fit, y_tune = train_test_split(
            X_train, y_train,
            test_size=float(args.tune_size),
            random_state=args.seed,
            stratify=y_train,
        )
        logger.info("  Internal Split: Fit=%d, Tune=%d", len(y_fit), len(y_tune))

        # Recompute weights for fit subset
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_fit)
        class_weights_fit = {0: float(cw[0]), 1: float(cw[1])}
        
        # Train on FIT subset
        if model_key == "xgb":
            num_classes = len(class_names)
            model = trainer(
                X_fit, y_fit,
                class_weight=class_weights_fit, seed=args.seed,
                num_classes=num_classes,
            )
        else:
            model = trainer(X_fit, y_fit, class_weight=class_weights_fit, seed=args.seed)
            
        # Tune on TUNE subset
        y_proba_tune = _predict_proba_safe(model, X_tune, task)
        if y_proba_tune is not None:
            logger.info("  Sweeping thresholds on internal TUNE split...")
            thresholds = np.linspace(0.01, 0.99, 99)
            best = _sweep_thresholds(np.asarray(y_tune), np.asarray(y_proba_tune), thresholds)

            chosen_threshold = float(best["threshold"])
            threshold_report = best
            logger.info(
                "  Chosen threshold: %.2f (Tune F1: %.4f)",
                chosen_threshold, float(best["f1"])
            )
        else:
            logger.warning("  Could not predict probabilities for tuning. Reverting to default threshold.")

    else:
        # Standard training on FULL training set
        logger.info("  Threshold Tuning DISABLED. Training on FULL train set.")
        logger.info("  Using fixed threshold: %.2f", chosen_threshold)
        
        if model_key == "if":
            if task != "binary":
                logger.error("Isolation Forest only supports binary task.")
                sys.exit(1)
            X_benign = pd.read_parquet(proc / "X_train_benign.parquet")
            model = trainer(X_benign, seed=args.seed)
        elif model_key == "xgb":
            num_classes = len(class_names)
            model = trainer(
                X_train, y_train,
                class_weight=class_weights, seed=args.seed,
                num_classes=num_classes,
            )
        else:
            model = trainer(X_train, y_train, class_weight=class_weights, seed=args.seed)

    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    # ── Predict & Evaluate (Test Set Only) ───────────────────────────
    logger.info("Generating predictions on TEST set (Friday) ...")

    if model_key == "if":
        # Isolation Forest: 1=inlier(benign), -1=outlier(attack) → 0/1
        y_pred_test = np.where(model.predict(X_test) == -1, 1, 0)
        y_proba_test = None
    else:
        y_proba_test = _predict_proba_safe(model, X_test, task)

        # Apply threshold (Multiclass ignores threshold)
        if task == "binary" and y_proba_test is not None:
            y_pred_test = _apply_threshold_binary(y_proba_test, chosen_threshold)
        else:
            y_pred_test = model.predict(X_test)

        # XGBoost label remap check (multiclass)
        remap = getattr(model, "_label_remap", None)
        if remap is not None:
            y_pred_test = np.array([remap.get(int(p), p) for p in y_pred_test])

    # ── Evaluate Test ────────────────────────────────────────────────
    logger.info("Evaluating on TEST set ...")
    metrics_test = evaluate_model(
        y_test, y_pred_test, y_proba_test,
        model_name=f"{run_name}_test",
        task=task,
        class_names=class_names,
        labels=labels,
        reports_dir=args.reports_dir,
    )
    if threshold_report:
        metrics_test["threshold_tuning"] = threshold_report
        
    import json
    with open(pathlib.Path(args.reports_dir) / "metrics" / f"{run_name}_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)

    # ── Save model ───────────────────────────────────────────────────
    model_path = models_dir / f"{run_name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model: %s (%.1f MB)",
                model_path.name, model_path.stat().st_size / 1e6)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✅ %s complete", run_name)
    logger.info("  Model:   %s", model_path)
    logger.info("  Metrics: %s", pathlib.Path(args.reports_dir) / "metrics" / f"{run_name}_test.json")
    logger.info("=" * 60)


def _predict_proba_safe(model: Any, X: pd.DataFrame, task: str) -> Optional[np.ndarray]:
    """Return probabilities if available. Binary -> (n,) pos_class."""
    try:
        proba = model.predict_proba(X)
        if task == "binary":
            return proba[:, 1]
        return proba
    except (AttributeError, RuntimeError):
        return None


def _apply_threshold_binary(proba_pos: np.ndarray, threshold: float) -> np.ndarray:
    return (proba_pos >= threshold).astype(int)


def _sweep_thresholds(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, Union[float, int]]:
    """Compute metrics for sweeps and return best by F1 (attack class)."""
    from sklearn.metrics import confusion_matrix

    best: Dict[str, Union[float, int]] = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "fpr": 1.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
    }

    y_true = np.asarray(y_true)

    for t in thresholds:
        y_hat = (proba_pos >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if f1 > float(best["f1"]):
            best.update({
                "threshold": float(t),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "fpr": float(fpr),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })

    return best


if __name__ == "__main__":
    main()
