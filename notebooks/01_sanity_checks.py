"""
Sanity-check script for the preprocessed CIC-IDS2017 dataset.

Produces:
  • Class distribution bar charts (before / after splitting)
  • Missing-value summary
  • Feature histograms (sampled for speed)
  • Correlation heat-map (top features)
  • Basic leakage checks

Saves figures to ``reports/`` and prints stats to the console.

Usage
-----
    python notebooks/01_sanity_checks.py
    python notebooks/01_sanity_checks.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root importable
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def _save_fig(fig: plt.Figure, path: pathlib.Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path.name)


# ------------------------------------------------------------------ #
#  1. Class distribution                                              #
# ------------------------------------------------------------------ #

def plot_class_distribution(
    y: pd.Series,
    title: str,
    out_path: pathlib.Path,
) -> None:
    """Bar chart of class counts."""
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.bar(ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    for i, (idx, v) in enumerate(counts.items()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    _save_fig(fig, out_path)


# ------------------------------------------------------------------ #
#  2. Missing values                                                  #
# ------------------------------------------------------------------ #

def check_missing_values(X: pd.DataFrame, name: str) -> None:
    """Log missing-value summary."""
    missing = X.isnull().sum()
    total_missing = missing.sum()
    logger.info("─── Missing values (%s) ───", name)
    if total_missing == 0:
        logger.info("  No missing values.")
    else:
        for col in missing[missing > 0].index:
            logger.info("  %-40s %10d", col, missing[col])
        logger.info("  TOTAL missing cells: %d", total_missing)


# ------------------------------------------------------------------ #
#  3. Feature histograms (sampled)                                    #
# ------------------------------------------------------------------ #

def plot_feature_histograms(
    X: pd.DataFrame,
    out_path: pathlib.Path,
    max_features: int = 20,
    sample_n: int = 50_000,
) -> None:
    """Plot histograms for the first *max_features* numeric columns."""
    cols = X.select_dtypes(include=[np.number]).columns[:max_features]
    sample = X[cols].sample(n=min(sample_n, len(X)), random_state=42)

    n = len(cols)
    ncols_grid = 4
    nrows_grid = int(np.ceil(n / ncols_grid))
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 3 * nrows_grid))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(sample[col].dropna(), bins=50, color="teal", edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Histograms (sampled)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(fig, out_path)


# ------------------------------------------------------------------ #
#  4. Correlation heatmap                                             #
# ------------------------------------------------------------------ #

def plot_correlation_heatmap(
    X: pd.DataFrame,
    out_path: pathlib.Path,
    max_features: int = 30,
) -> None:
    """Heatmap of Pearson correlations for top *max_features* columns."""
    cols = X.select_dtypes(include=[np.number]).columns[:max_features]
    corr = X[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr, ax=ax, cmap="coolwarm", center=0,
        square=True, linewidths=0.3, fmt=".1f",
        annot=len(cols) <= 15,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    _save_fig(fig, out_path)


# ------------------------------------------------------------------ #
#  5. Leakage check                                                   #
# ------------------------------------------------------------------ #

def check_leakage(X: pd.DataFrame) -> None:
    """Warn if suspicious columns still exist."""
    suspicious = {"flow_id", "source_ip", "destination_ip", "timestamp",
                  "source_port", "destination_port", "src_ip", "dst_ip"}
    found = [c for c in X.columns if c.lower() in suspicious]
    if found:
        logger.warning("⚠  Potential leakage columns still present: %s", found)
    else:
        logger.info("✅ No obvious leakage columns detected.")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main(processed_dir: str = "data/processed") -> None:
    setup_logging()
    proc = pathlib.Path(processed_dir)
    reports = pathlib.Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    if not proc.exists():
        logger.error("Processed directory '%s' not found. Run preprocessing first.", proc)
        sys.exit(1)

    # Load splits
    X_train = pd.read_parquet(proc / "X_train.parquet")
    y_train = pd.read_parquet(proc / "y_train.parquet").iloc[:, 0]
    X_test  = pd.read_parquet(proc / "X_test.parquet")
    y_test  = pd.read_parquet(proc / "y_test.parquet").iloc[:, 0]

    logger.info("Train: %d rows × %d cols", *X_train.shape)
    logger.info("Test : %d rows × %d cols", *X_test.shape)

    # 1. Class distributions
    plot_class_distribution(y_train, "Training Set Class Distribution", reports / "class_dist_train.png")
    plot_class_distribution(y_test,  "Test Set Class Distribution",     reports / "class_dist_test.png")

    # 2. Missing values
    check_missing_values(X_train, "X_train")
    check_missing_values(X_test,  "X_test")

    # 3. Feature histograms
    plot_feature_histograms(X_train, reports / "feature_histograms.png")

    # 4. Correlation heatmap
    plot_correlation_heatmap(X_train, reports / "correlation_heatmap.png")

    # 5. Leakage check
    check_leakage(X_train)

    logger.info("✅ Sanity checks complete. Figures saved to %s", reports)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks on processed data")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()
    main(processed_dir=args.processed_dir)
