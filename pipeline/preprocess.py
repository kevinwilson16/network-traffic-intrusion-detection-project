"""
Data ingestion, cleaning, and label normalisation for CIC-IDS2017.

Merged from the original ``ingest.py``, ``clean.py``, and ``labels.py``.

Pipeline order (within this module):
  1. ``load_all_csvs()``    — discover + load raw CSVs, tag with source_file
  2. ``clean_dataframe()``  — normalise cols, inf→NaN, dedup BEFORE drop,
                              then drop leakage columns
  3. ``normalise_labels()`` — unify label variants (incl. \\x96 dashes)
  4. ``create_binary_labels()`` / ``create_multiclass_labels()``

NOTE: NaN imputation is NOT done here.  It happens after the train/val/test
split in ``split_transform.py`` so the imputer is fitted on training data only.
"""

from __future__ import annotations

import pathlib
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pipeline.utils import get_logger, read_csv_safe

logger = get_logger(__name__)


# ================================================================== #
#  INGESTION                                                          #
# ================================================================== #

def discover_csv_files(raw_dir: Union[str, pathlib.Path]) -> List[pathlib.Path]:
    """Find all CSV files in *raw_dir* (sorted alphabetically)."""
    raw_dir = pathlib.Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    logger.info("Discovered %d CSV file(s) in %s", len(csv_files), raw_dir)
    for f in csv_files:
        logger.info("  → %s", f.name)
    return csv_files


def load_single_csv(
    filepath: pathlib.Path,
    encoding: str = "utf-8",
    fallback_encoding: str = "latin-1",
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load one CSV and tag rows with ``source_file`` column."""
    df = read_csv_safe(
        filepath,
        encoding=encoding,
        fallback_encoding=fallback_encoding,
        chunk_size=chunk_size,
    )
    df["source_file"] = filepath.stem
    logger.info("  %s → %d rows × %d cols", filepath.name, *df.shape)
    return df


def load_all_csvs(
    raw_dir: Union[str, pathlib.Path],
    encoding: str = "utf-8",
    fallback_encoding: str = "latin-1",
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Discover, load, and concatenate all CSVs in *raw_dir*."""
    csv_files = discover_csv_files(raw_dir)
    frames: List[pd.DataFrame] = []
    file_stats: List[tuple] = []

    for fpath in csv_files:
        df = load_single_csv(fpath, encoding, fallback_encoding, chunk_size)
        file_stats.append((fpath.name, len(df)))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Summary
    logger.info("─── Ingest summary ───")
    for name, count in file_stats:
        logger.info("  %-60s %10d rows", name, count)
    logger.info("  %-60s %10d rows", "TOTAL", len(combined))
    logger.info("  Columns: %d", combined.shape[1])
    return combined


# ================================================================== #
#  COLUMN NAME NORMALISATION                                          #
# ================================================================== #

_COLUMN_ALIASES: Dict[str, str] = {
    "src_ip": "source_ip",
    "dst_ip": "destination_ip",
    "src_port": "source_port",
    "dst_port": "destination_port",
    "fwd_header_length.1": "fwd_header_length_1",
}


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip whitespace, replace special chars with ``_``."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[ /]+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    df.rename(columns=_COLUMN_ALIASES, inplace=True)

    # Deduplicate exact column names
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    logger.info("Normalised %d column names.", len(df.columns))
    return df


# ================================================================== #
#  INFINITY HANDLING                                                   #
# ================================================================== #

def handle_infinities(df: pd.DataFrame, strategy: str = "nan") -> pd.DataFrame:
    """Replace ±inf values (``"nan"`` or ``"clip"``)."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[num_cols].values)
    inf_count = int(inf_mask.sum())

    if inf_count == 0:
        logger.info("No infinite values found.")
        return df

    if strategy == "nan":
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced %d inf values with NaN.", inf_count)
    elif strategy == "clip":
        for col in num_cols:
            finite = df[col][np.isfinite(df[col])]
            if finite.empty:
                df[col].replace([np.inf, -np.inf], 0, inplace=True)
            else:
                df[col].replace(np.inf, finite.max(), inplace=True)
                df[col].replace(-np.inf, finite.min(), inplace=True)
        logger.info("Clipped %d inf values to column min/max.", inf_count)
    else:
        raise ValueError(f"Unknown inf strategy: {strategy!r}")
    return df


# ================================================================== #
#  DUPLICATE REMOVAL                                                  #
# ================================================================== #

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (ignoring ``source_file`` tag)."""
    check_cols = [c for c in df.columns if c != "source_file"]
    before = len(df)
    df.drop_duplicates(subset=check_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    removed = before - len(df)
    logger.info("Removed %d duplicate rows (%d → %d).", removed, before, len(df))
    return df


# ================================================================== #
#  COLUMN DROPPING                                                    #
# ================================================================== #

def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Drop specified columns if they exist."""
    existing = [c for c in columns_to_drop if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)
        logger.info("Dropped columns: %s", existing)
    return df


# ================================================================== #
#  OUTLIER CLIPPING                                                   #
# ================================================================== #

def clip_outliers(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Clip numeric columns to ``[Q1 - factor*IQR, Q3 + factor*IQR]``."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    clipped_total = 0
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        clipped_total += int(mask.sum())
        df[col] = df[col].clip(lower=lower, upper=upper)
    logger.info("Clipped %d outlier values (factor=%.1f).", clipped_total, factor)
    return df


# ================================================================== #
#  CLEANING ORCHESTRATOR                                              #
# ================================================================== #

def clean_dataframe(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
    inf_strategy: str = "nan",
    deduplicate: bool = True,
    do_clip_outliers: bool = False,
    clip_factor: float = 3.0,
) -> pd.DataFrame:
    """Full cleaning pipeline.

    Order of operations:
      1. Normalise column names
      2. Coerce non-numeric strings to NaN
      3. Replace ±inf with NaN
      4. Deduplicate BEFORE dropping identifying columns (preserves PortScan)
      5. Drop leakage / non-feature columns
      6. Optionally clip outliers

    NaN imputation is deferred to after the split (see ``split_transform.py``).
    """
    logger.info("═══ Cleaning pipeline started ═══")
    logger.info("  Input shape: %d rows × %d cols", *df.shape)

    df = normalise_column_names(df)

    # Coerce non-numeric strings in numeric columns
    label_col = "label" if "label" in df.columns else None
    exclude = {"source_file"}
    if label_col:
        exclude.add(label_col)
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = handle_infinities(df, strategy=inf_strategy)

    # Dedup BEFORE dropping IPs/ports (prevents PortScan collapse)
    if deduplicate:
        df = remove_duplicates(df)

    # Drop leakage columns AFTER dedup
    if columns_to_drop:
        df = drop_columns(df, columns_to_drop)

    if do_clip_outliers:
        df = clip_outliers(df, factor=clip_factor)

    num_cols = df.select_dtypes(include=[np.number]).columns
    nan_count = int(df[num_cols].isna().sum().sum())
    logger.info("  Remaining NaN values: %d (will be imputed after split).", nan_count)
    logger.info("  Output shape: %d rows × %d cols", *df.shape)
    logger.info("═══ Cleaning pipeline finished ═══")
    return df


# ================================================================== #
#  LABEL NORMALISATION                                                #
# ================================================================== #

CANONICAL_LABELS: Dict[str, str] = {
    "benign": "BENIGN",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS Slowloris",
    "dos slowhttptest": "DoS SlowHTTPTest",
    "heartbleed": "Heartbleed",
    "ddos": "DDoS",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack - xss": "Web Attack - XSS",
    "web attack - sql injection": "Web Attack - SQL Injection",
    "infiltration": "Infiltration",
    "portscan": "PortScan",
    "bot": "Bot",
}

# Regex to normalise all dash-like characters
_DASH_PATTERN = re.compile(r"[\x96\u2013\u2014\u2015\u2212]+")


def _normalise_raw_label(raw) -> str:
    """Strip, lowercase, collapse spaces, normalise dashes."""
    if not isinstance(raw, str):
        return str(raw).strip().lower()
    text = raw.strip().lower()
    text = _DASH_PATTERN.sub("-", text)
    return " ".join(text.split())


def normalise_labels(
    df: pd.DataFrame,
    label_column: str = "label",
) -> pd.DataFrame:
    """Apply the canonical label mapping.  Unknown labels are kept (title-cased)."""
    if label_column not in df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    # Drop rows with missing labels
    nan_mask = df[label_column].isna()
    if nan_mask.any():
        n_dropped = int(nan_mask.sum())
        logger.warning("Dropping %d rows with missing labels.", n_dropped)
        df = df[~nan_mask].reset_index(drop=True)

    raw_labels = df[label_column].astype(str)
    normed = raw_labels.map(_normalise_raw_label)
    mapped = normed.map(CANONICAL_LABELS)

    unmapped_mask = mapped.isna()
    if unmapped_mask.any():
        unknown = normed[unmapped_mask].unique()
        logger.warning("Unknown label(s) not in canonical map: %s", list(unknown))
        mapped[unmapped_mask] = raw_labels[unmapped_mask].str.strip().str.title()

    df[label_column] = mapped
    logger.info("Labels normalised. Unique: %s", sorted(df[label_column].unique()))
    return df


def create_binary_labels(
    df: pd.DataFrame,
    label_column: str = "label",
    target_column: str = "label_binary",
) -> pd.DataFrame:
    """BENIGN → 0, all attacks → 1."""
    df[target_column] = (df[label_column] != "BENIGN").astype(int)
    counts = df[target_column].value_counts().to_dict()
    logger.info("Binary labels created → %s", counts)
    return df


def create_multiclass_labels(
    df: pd.DataFrame,
    label_column: str = "label",
    target_column: str = "label_multi",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Integer-encode multiclass labels (BENIGN = 0)."""
    unique_labels = sorted(df[label_column].unique())
    if "BENIGN" in unique_labels:
        unique_labels.remove("BENIGN")
        unique_labels = ["BENIGN"] + unique_labels

    label_map: Dict[str, int] = OrderedDict(
        (lbl, idx) for idx, lbl in enumerate(unique_labels)
    )
    df[target_column] = df[label_column].map(label_map)

    logger.info("Multiclass label map:")
    for lbl, code in label_map.items():
        count = int((df[target_column] == code).sum())
        logger.info("  %3d  %-30s  %10d rows", code, lbl, count)
    return df, label_map


def print_label_distribution(
    df: pd.DataFrame,
    column: str = "label",
    title: str = "Label distribution",
) -> None:
    """Log label value-counts as a formatted table."""
    counts = df[column].value_counts()
    total = len(df)
    logger.info("─── %s (%s) ───", title, column)
    for lbl, cnt in counts.items():
        pct = 100.0 * cnt / total
        logger.info("  %-30s %10d  (%5.2f%%)", lbl, cnt, pct)
    logger.info("  %-30s %10d", "TOTAL", total)
