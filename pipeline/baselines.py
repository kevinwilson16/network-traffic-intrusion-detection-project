"""
Baseline model training and evaluation to validate the processed dataset.

Models:
  • Logistic Regression (linear baseline)
  • Random Forest (ensemble)
  • XGBoost (gradient boosting)
  • Isolation Forest (unsupervised anomaly detection)

Moved from ``src/models/baselines.py`` — no logic changes.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from pipeline.utils import get_logger

logger = get_logger(__name__)


def _evaluate(name: str, y_true, y_pred) -> Dict[str, float]:
    """Compute and log standard metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    logger.info("── %s ──", name)
    for k, v in metrics.items():
        logger.info("  %-12s %.4f", k, v)
    logger.info("\n%s", classification_report(y_true, y_pred, zero_division=0))
    return metrics


def train_logistic_regression(
    X_train, y_train, X_test, y_test,
    class_weight: Optional[Dict] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate Logistic Regression."""
    model = LogisticRegression(
        max_iter=1000, class_weight=class_weight or "balanced",
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _evaluate("Logistic Regression", y_test, y_pred)
    return {"model": model, "metrics": metrics}


def train_random_forest(
    X_train, y_train, X_test, y_test,
    class_weight: Optional[Dict] = None,
    n_estimators: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate Random Forest."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight or "balanced",
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = _evaluate("Random Forest", y_test, y_pred)
    return {"model": model, "metrics": metrics}


def train_xgboost(
    X_train, y_train, X_test, y_test,
    class_weight: Optional[Dict] = None,
    n_estimators: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate XGBoost."""
    import xgboost as xgb

    # Convert class weights to sample weights
    sample_weights = None
    if class_weight:
        sample_weights = np.array([class_weight.get(int(y), 1.0) for y in y_train])

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    metrics = _evaluate("XGBoost", y_test, y_pred)
    return {"model": model, "metrics": metrics}


def train_isolation_forest(
    X_train_benign, X_test, y_test,
    contamination: float = 0.15,
    n_estimators: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train Isolation Forest on benign data, predict on test set."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train_benign)
    raw_pred = model.predict(X_test)
    # IF: 1=inlier(benign), -1=outlier(attack) → convert to 0/1
    y_pred = np.where(raw_pred == -1, 1, 0)
    metrics = _evaluate("Isolation Forest", y_test, y_pred)
    return {"model": model, "metrics": metrics}


def run_all_baselines(
    processed_dir: Union[str, pathlib.Path] = "data/processed",
    models_dir: Union[str, pathlib.Path] = "models",
    seed: int = 42,
) -> None:
    """Load processed data and run all baseline models."""
    processed_dir = pathlib.Path(processed_dir)
    models_dir = pathlib.Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_parquet(processed_dir / "X_train.parquet")
    X_test  = pd.read_parquet(processed_dir / "X_test.parquet")
    y_train = pd.read_parquet(processed_dir / "y_train.parquet").iloc[:, 0]
    y_test  = pd.read_parquet(processed_dir / "y_test.parquet").iloc[:, 0]

    # Load class weights if available
    cw_path = processed_dir / "class_weights.joblib"
    class_weight = joblib.load(cw_path) if cw_path.exists() else None

    results = {}

    # Supervised
    results["lr"] = train_logistic_regression(X_train, y_train, X_test, y_test, class_weight, seed)
    results["rf"] = train_random_forest(X_train, y_train, X_test, y_test, class_weight, seed=seed)
    results["xgb"] = train_xgboost(X_train, y_train, X_test, y_test, class_weight, seed=seed)

    # Unsupervised
    benign_path = processed_dir / "X_train_benign.parquet"
    if benign_path.exists():
        X_benign = pd.read_parquet(benign_path)
        results["if"] = train_isolation_forest(X_benign, X_test, y_test, seed=seed)

    # Save models
    for name, res in results.items():
        joblib.dump(res["model"], models_dir / f"{name}_model.joblib")
        logger.info("Saved %s model to %s", name, models_dir / f"{name}_model.joblib")

    logger.info("✅ All baselines complete.")


if __name__ == "__main__":
    from pipeline.utils import setup_logging
    setup_logging()
    run_all_baselines()
