"""
Standardised evaluation module for all classifiers.

Outputs saved automatically:
    reports/metrics/<model_name>.json
    reports/figures/<model_name>_confusion.png
    reports/figures/<model_name>_roc.png      (binary only)
    reports/figures/<model_name>_pr.png       (binary only)
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from pipeline.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true,
    y_pred,
    y_proba=None,
    *,
    model_name: str,
    task: str = "binary",
    class_names: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
    reports_dir: Union[str, pathlib.Path] = "reports",
) -> Dict[str, Any]:
    """Run full evaluation for a single model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    reports_dir = pathlib.Path(reports_dir)
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = _compute_core_metrics(
        y_true, y_pred, task=task, labels=labels,
    )

    if y_proba is not None:
        prob_metrics = _compute_proba_metrics(
            y_true, y_proba, task=task, labels=labels,
        )
        metrics.update(prob_metrics)

    report_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report_dict

    _log_metrics(model_name, metrics)

    json_path = metrics_dir / f"{model_name}.json"
    _save_metrics_json(metrics, json_path)

    _save_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        labels=labels,
        title=f"{model_name} - Confusion Matrix",
        out_path=figures_dir / f"{model_name}_confusion.png",
    )

    if task == "binary" and y_proba is not None:
        _save_roc_curve(
            y_true, y_proba,
            title=f"{model_name} - ROC Curve",
            out_path=figures_dir / f"{model_name}_roc.png",
        )
        _save_pr_curve(
            y_true, y_proba,
            title=f"{model_name} - Precision-Recall Curve",
            out_path=figures_dir / f"{model_name}_pr.png",
        )

    logger.info("Evaluation complete for %s - outputs in %s", model_name, reports_dir)
    return metrics


def _compute_core_metrics(
    y_true, y_pred, *, task: str, labels=None,
) -> Dict[str, Any]:
    """Accuracy, balanced accuracy, precision, recall, F1."""
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if task == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        })
        metrics["precision"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    else:
        metrics["precision_weighted"] = float(precision_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0,
        ))
        metrics["recall_weighted"] = float(recall_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0,
        ))
        metrics["f1_weighted"] = float(f1_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0,
        ))

    return metrics


def _compute_proba_metrics(
    y_true, y_proba, *, task: str, labels=None,
) -> Dict[str, Any]:
    """ROC-AUC, PR-AUC, and Average Precision."""
    metrics: Dict[str, Any] = {}
    try:
        if task == "binary":
            proba = np.asarray(y_proba)
            if proba.ndim == 2:
                proba = proba[:, 1]

            y_true_bin = (np.asarray(y_true).astype(int)).ravel()

            metrics["roc_auc"] = float(roc_auc_score(y_true_bin, proba))
            metrics["average_precision"] = float(average_precision_score(y_true_bin, proba))

            prec_vals, rec_vals, _ = precision_recall_curve(y_true_bin, proba)
            metrics["pr_auc"] = float(auc(rec_vals, prec_vals))
        else:
            present = np.unique(y_true).tolist()
            metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(
                y_true, y_proba,
                multi_class="ovr", average="weighted",
                labels=present,
            ))
    except ValueError as exc:
        logger.warning("Could not compute probability metrics: %s", exc)

    return metrics


def _save_confusion_matrix(
    y_true, y_pred,
    class_names=None, labels=None,
    title: str = "Confusion Matrix",
    out_path: Union[str, pathlib.Path] = "confusion.png",
) -> None:
    """Generate and save a confusion matrix (matplotlib only)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    n_classes = cm.shape[0]
    fig_size = max(6, n_classes * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Use class names if provided, else numeric labels
    tick_labels = class_names
    if tick_labels is None:
        tick_labels = labels if labels is not None else list(range(n_classes))

    # Show matrix as image
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    # Annotate cells
    fmt = "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(int(cm[i, j]), fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", pathlib.Path(out_path).name)


def _save_roc_curve(
    y_true, y_proba,
    title: str = "ROC Curve",
    out_path: Union[str, pathlib.Path] = "roc.png",
) -> None:
    """Plot and save the ROC curve (binary only)."""
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]

    y_true_bin = (np.asarray(y_true).astype(int)).ravel()
    fpr, tpr, _ = roc_curve(y_true_bin, proba)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve: %s", pathlib.Path(out_path).name)


def _save_pr_curve(
    y_true, y_proba,
    title: str = "Precision-Recall Curve",
    out_path: Union[str, pathlib.Path] = "pr.png",
) -> None:
    """Plot and save the Precision-Recall curve (binary only)."""
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]

    y_true_bin = (np.asarray(y_true).astype(int)).ravel()
    prec_vals, rec_vals, _ = precision_recall_curve(y_true_bin, proba)
    pr_auc_val = auc(rec_vals, prec_vals)

    baseline = float(np.mean(y_true_bin == 1))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_vals, prec_vals, lw=2, label=f"PR (AUC = {pr_auc_val:.4f})")
    ax.axhline(y=baseline, lw=1, linestyle="--", label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curve: %s", pathlib.Path(out_path).name)


def _save_metrics_json(
    metrics: Dict[str, Any],
    out_path: Union[str, pathlib.Path],
) -> None:
    """Save metrics dict to JSON (skip non-serialisable values)."""
    def _make_serialisable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serialisable(v) for v in obj]
        return obj

    clean = _make_serialisable(metrics)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)
    logger.info("Saved metrics JSON: %s", pathlib.Path(out_path).name)


def _log_metrics(model_name: str, metrics: Dict[str, Any]) -> None:
    """Print key scalar metrics to the logger."""
    logger.info("── %s ──", model_name)
    scalar_keys = [
        "accuracy", "balanced_accuracy",
        "precision", "recall", "f1",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "tp", "fp", "fn", "tn", "fpr",
        "roc_auc", "pr_auc", "average_precision",
        "roc_auc_ovr_weighted",
    ]
    for key in scalar_keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                logger.info("  %-24s %.4f", key, val)
            else:
                logger.info("  %-24s %s", key, val)