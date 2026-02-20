"""
Train/test splitting, NaN imputation, feature scaling, and
imbalance resampling for CIC-IDS2017.

Protocol:
1. Train = Monday, Tuesday, Wednesday, Thursday
2. Test = Friday

"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from pipeline.utils import get_logger

logger = get_logger(__name__)

# Type alias for the six-element split result
SplitResult = Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
]


# ================================================================== #
#  SPLITTING STRATEGIES                                               #
# ================================================================== #

def _random_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> SplitResult:
    """Stratified random split into train/test (val empty)."""
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Only one split: train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed,
    )
    
    # Empty validation
    X_val = pd.DataFrame(columns=X.columns)
    y_val = pd.Series(dtype=y.dtype)

    logger.info(
        "Random split -> train=%d  val=0 (disabled)  test=%d",
        len(X_train), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


_DEFAULT_DAY_ASSIGNMENT: Dict[str, List[str]] = {
    "train": ["monday", "tuesday", "wednesday", "thursday"],
    "val":   [],  # Disabled
    "test":  ["friday"],
}


def _day_based_split(
    df: pd.DataFrame,
    target_col: str,
    day_assignment: Optional[Dict[str, List[str]]] = None,
) -> SplitResult:
    """Assign entire days to train/test (prevents temporal leakage)."""
    assignment = day_assignment or _DEFAULT_DAY_ASSIGNMENT

    if "source_file" not in df.columns:
        raise KeyError("Column 'source_file' is required for day-based split.")

    source = df["source_file"].str.lower()

    def _match(keywords: List[str]) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        if not keywords:
            return mask
        for kw in keywords:
            mask |= source.str.contains(kw.lower(), na=False)
        return mask

    train_mask = _match(assignment["train"])
    test_mask = _match(assignment["test"])
    # Val is empty by default/design

    # Rows matching no group -> train (catch-all)
    unmatched = ~(train_mask | test_mask)
    if unmatched.any():
        logger.warning(
            "%d rows matched no day keyword - assigned to train.", int(unmatched.sum())
        )
        train_mask |= unmatched

    # Drop source_file (not a feature)
    def _split_xy(mask):
        if not mask.any():
            return df.iloc[:0].drop(columns=["source_file", target_col], errors="ignore"), \
                   df.iloc[:0][target_col]
        
        subset = df[mask].drop(columns=["source_file"], errors="ignore")
        return subset.drop(columns=[target_col]), subset[target_col]

    X_train, y_train = _split_xy(train_mask)
    X_test, y_test = _split_xy(test_mask)
    
    # Empty val
    X_val = pd.DataFrame(columns=X_train.columns)
    y_val = pd.Series(dtype=y_train.dtype, name=target_col)

    for split in (X_train, X_val, X_test):
        split.reset_index(drop=True, inplace=True)
    for split in (y_train, y_val, y_test):
        split.reset_index(drop=True, inplace=True)

    logger.info(
        "Day-based split -> train=%d (Mon-Thu)  val=0 (disabled)  test=%d (Fri)",
        len(X_train), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _time_based_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
) -> SplitResult:
    """Chronological split based on row order (Train/Test only)."""
    n = len(df)
    test_n = int(n * test_size)
    train_n = n - test_n

    if "source_file" in df.columns:
        df = df.drop(columns=["source_file"])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, y_train = X.iloc[:train_n], y.iloc[:train_n]
    X_test, y_test = X.iloc[train_n:], y.iloc[train_n:]
    
    X_val = pd.DataFrame(columns=X.columns)
    y_val = pd.Series(dtype=y.dtype)

    for split in (X_train, X_val, X_test, y_train, y_val, y_test):
        split.reset_index(drop=True, inplace=True)

    logger.info(
        "Time-based split -> train=%d  val=0  test=%d",
        len(X_train), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "day_based",
    test_size: float = 0.20,
    val_size: float = 0.10,
    seed: int = 42,
    day_assignment: Optional[Dict[str, List[str]]] = None,
) -> SplitResult:
    """Dispatch to the chosen splitting strategy."""
    strategy = strategy.lower().strip()
    if strategy == "random":
        return _random_split(df, target_col, test_size, val_size, seed)
    elif strategy == "day_based":
        return _day_based_split(df, target_col, day_assignment)
    elif strategy == "time_based":
        return _time_based_split(df, target_col, test_size, val_size)
    else:
        raise ValueError(
            f"Unknown split strategy: {strategy!r}. "
            "Choose from: random, day_based, time_based."
        )


# ================================================================== #
#  POST-SPLIT NaN IMPUTATION (fitted on train only)                   #
# ================================================================== #

def fit_imputer_and_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "median",
    output_dir: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit imputer on train only, transform all splits safely."""
    feature_names = list(X_train.columns)

    # Impute only if data exists
    if strategy == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        imputer = SimpleImputer(strategy=strategy)

    imputer.fit(X_train)
    logger.info("Fitted SimpleImputer(strategy='%s') on training data.", strategy)
    
    def _transform(df, name):
        if df.empty:
            return df
        return pd.DataFrame(imputer.transform(df), columns=feature_names)

    X_train = _transform(X_train, "train")
    X_val   = _transform(X_val, "val")
    X_test  = _transform(X_test, "test")

    remaining = int(X_train.isna().sum().sum()) + int(X_test.isna().sum().sum())
    logger.info("NaN remaining after imputation (Train+Test): %d", remaining)

    if output_dir is not None:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(imputer, out / "imputer.joblib")
        logger.info("Saved imputer to %s", out / "imputer.joblib")

    return X_train, X_val, X_test


# ================================================================== #
#  FEATURE SCALING (fitted on train only)                             #
# ================================================================== #

_SCALER_MAP = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def fit_and_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_name: str = "robust",
    output_dir: Optional[Union[str, pathlib.Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit scaler on train, transform all splits."""
    scaler_name = scaler_name.lower().strip()
    feature_names = list(X_train.columns)

    if scaler_name == "none":
        logger.info("Scaler is 'none' - data unchanged.")
        return X_train, X_val, X_test

    cls = _SCALER_MAP.get(scaler_name)
    if cls is None:
        raise ValueError(f"Unknown scaler: {scaler_name!r}. Choose from {list(_SCALER_MAP)}.")

    scaler = cls()
    scaler.fit(X_train)
    logger.info("Fitted %s on training data (%d features).", scaler_name, len(feature_names))

    def _transform(df):
        if df.empty:
            return df
        return pd.DataFrame(scaler.transform(df), columns=feature_names)

    X_train = _transform(X_train)
    X_val   = _transform(X_val)
    X_test  = _transform(X_test)

    if output_dir is not None:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, out / "scaler.joblib")
        joblib.dump(feature_names, out / "feature_names.joblib")
        logger.info("Saved scaler and feature list to %s", out)

    return X_train, X_val, X_test


# ================================================================== #
#  IMBALANCE RESAMPLING (train only)                                  #
# ================================================================== #

def resample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
    min_class_samples: int = 6,
    k_neighbors: int = 5,
    sampling_strategy: Union[str, dict] = "auto",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply resampling to training set with safeguards for tiny classes."""
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    logger.info("Resampling with method='%s'.", method)
    logger.info("  Before: %d samples", len(y_train))

    class_counts = y_train.value_counts()
    tiny = class_counts[class_counts < min_class_samples].index.tolist()
    if tiny:
        logger.warning("Classes with < %d samples (excluded): %s", min_class_samples, tiny)

    safe_strategy = sampling_strategy
    if isinstance(sampling_strategy, str) and sampling_strategy == "auto" and tiny:
        majority_count = int(class_counts.max())
        safe_strategy = {
            cls: majority_count
            for cls in class_counts.index
            if cls not in tiny and class_counts[cls] < majority_count
        }
        if not safe_strategy:
            logger.warning("No valid classes to resample. Returning as-is.")
            return X_train, y_train

    method = method.lower().strip()
    eff_k = max(1, min(k_neighbors, int(class_counts.min()) - 1))

    samplers = {
        "smote": lambda: SMOTE(sampling_strategy=safe_strategy, k_neighbors=eff_k, random_state=seed),
        "adasyn": lambda: ADASYN(sampling_strategy=safe_strategy, n_neighbors=eff_k, random_state=seed),
        "smote_tomek": lambda: SMOTETomek(smote=SMOTE(sampling_strategy=safe_strategy, k_neighbors=eff_k, random_state=seed), random_state=seed),
        "smote_enn": lambda: SMOTEENN(smote=SMOTE(sampling_strategy=safe_strategy, k_neighbors=eff_k, random_state=seed), random_state=seed),
        "random_over": lambda: RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed),
        "random_under": lambda: RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=seed),
    }

    if method not in samplers:
        raise ValueError(f"Unknown method: {method!r}. Choose from {list(samplers)}.")

    X_res, y_res = samplers[method]().fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)

    logger.info("  After:  %d samples", len(y_res))
    return X_res, y_res


# ================================================================== #
#  ANOMALY-DETECTION SET                                              #
# ================================================================== #

def build_anomaly_detection_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    benign_label: int = 0,
) -> pd.DataFrame:
    """Return benign-only training set for Isolation Forest etc."""
    mask = y_train == benign_label
    X_benign = X_train[mask].reset_index(drop=True)
    logger.info("Anomaly-detection set: %d benign (of %d total).", len(X_benign), len(X_train))
    return X_benign


# ================================================================== #
#  CLASS WEIGHTS                                                      #
# ================================================================== #

def compute_class_weights(
    y_train: pd.Series,
    mode: str = "balanced",
) -> Dict[int, float]:
    """Compute balanced class weights for sklearn models."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.sort(y_train.unique())
    weights = compute_class_weight(mode, classes=classes, y=y_train)
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.info("Class weights (%s): %s", mode, weight_dict)
    return weight_dict
