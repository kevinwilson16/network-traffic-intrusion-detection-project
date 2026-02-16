"""
Feature selection utilities: variance threshold, correlation filter,
and mutual-information ranking.

Moved from ``src/features/selection.py`` — no logic changes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
)

from pipeline.utils import get_logger

logger = get_logger(__name__)


def variance_filter(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Drop features with variance ≤ *threshold* (fitted on train)."""
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X_train)
    mask = sel.get_support()
    kept = X_train.columns[mask].tolist()
    dropped = X_train.columns[~mask].tolist()

    if dropped:
        logger.info("Variance filter dropped %d features: %s", len(dropped), dropped)
    else:
        logger.info("Variance filter: all features kept.")

    return X_train[kept], X_val[kept], X_test[kept], kept


def correlation_filter(
    X_train: pd.DataFrame,
    threshold: float = 0.95,
) -> List[str]:
    """Return list of features to drop (high pairwise |r| > *threshold*)."""
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info(
        "Correlation filter (threshold=%.2f): %d features to drop.",
        threshold, len(to_drop),
    )
    return to_drop


def mutual_info_ranking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Rank features by mutual information with the target.

    Returns a DataFrame with columns ``feature`` and ``mi_score``,
    sorted descending.  If *top_k* is given, only the top-k are returned.
    """
    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    ranking = (
        pd.DataFrame({"feature": X_train.columns, "mi_score": mi})
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    if top_k is not None:
        ranking = ranking.head(top_k)
        logger.info("MI ranking: top %d features selected.", top_k)
    else:
        logger.info("MI ranking computed for all %d features.", len(ranking))

    return ranking
