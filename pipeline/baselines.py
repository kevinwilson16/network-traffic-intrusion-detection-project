"""
Baseline model training functions.

Each function takes training data, fits a model, and returns the fitted model.
Evaluation is handled separately by ``pipeline.evaluate``.

Models:
  - Logistic Regression (linear baseline)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pipeline.utils import get_logger

logger = get_logger(__name__)


# ================================================================== #
#  SUPERVISED CLASSIFIERS                                             #
# ================================================================== #

def train_logistic_regression(
    X_train, y_train,
    class_weight: Optional[Dict] = None,
    seed: int = 42,
) -> Any:
    """Train Logistic Regression and return the fitted model."""
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight or "balanced",
        random_state=seed,
        solver="saga",
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    logger.info("Logistic Regression trained in %.1fs", time.time() - t0)
    return model

# ================================================================== #
#  REGISTRY (used by train_model.py)                                  #
# ================================================================== #

MODELS = {
    "lr":  ("Logistic Regression", train_logistic_regression),
}