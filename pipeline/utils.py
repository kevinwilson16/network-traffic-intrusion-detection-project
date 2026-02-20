"""
Utility helpers: structured logging and robust I/O for CSV / Parquet / Feather.

Merged from the original ``src/utils/io.py`` and ``src/utils/logging.py``.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Generator, List, Optional, Union

import pandas as pd


# ================================================================== #
#  LOGGING                                                            #
# ================================================================== #

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger (console + optional file).  Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    root.addHandler(ch)

    # File (optional)
    if log_file:
        log_path = pathlib.Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        root.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger."""
    return logging.getLogger(name)


# ================================================================== #
#  CSV READING                                                        #
# ================================================================== #

logger = get_logger(__name__)


def read_csv_safe(
    filepath: Union[str, pathlib.Path],
    encoding: str = "utf-8",
    fallback_encoding: str = "latin-1",
    chunk_size: Optional[int] = None,
    dtype: Optional[dict] = None,
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """Read a CSV with encoding fallback and optional chunking.

    Tries *encoding* first; if a ``UnicodeDecodeError`` is raised,
    retries with *fallback_encoding* (common for CIC-IDS2017 files
    saved with Windows-1252 encoding).
    """
    filepath = pathlib.Path(filepath)
    read_kwargs: dict = dict(low_memory=False, on_bad_lines="warn")
    if dtype is not None:
        read_kwargs["dtype"] = dtype

    if chunk_size is not None and chunk_size > 0:
        return _read_chunked(filepath, encoding, fallback_encoding,
                             chunk_size, read_kwargs)

    # Single-shot read with encoding fallback
    for enc in (encoding, fallback_encoding):
        try:
            result = pd.read_csv(filepath, encoding=enc, **read_kwargs)
            logger.info(
                "Loaded %s (1 chunks, encoding=%s)", filepath.name, enc
            )
            return result
        except UnicodeDecodeError:
            logger.warning(
                "Encoding %s failed for %s — trying fallback.",
                enc, filepath.name,
            )
    raise RuntimeError(
        f"Could not read {filepath.name} with any encoding "
        f"({encoding}, {fallback_encoding})."
    )


def _read_chunked(
    filepath: pathlib.Path,
    encoding: str,
    fallback_encoding: str,
    chunk_size: int,
    read_kwargs: dict,
) -> pd.DataFrame:
    """Read a large CSV in chunks, concatenate, return a single DataFrame."""
    enc_to_use = encoding
    try:
        # Test first chunk
        pd.read_csv(filepath, encoding=encoding, nrows=5, **read_kwargs)
    except UnicodeDecodeError:
        logger.warning(
            "Encoding %s failed for %s — trying fallback.",
            encoding, filepath.name,
        )
        enc_to_use = fallback_encoding

    chunks: List[pd.DataFrame] = []
    reader = pd.read_csv(
        filepath, encoding=enc_to_use, chunksize=chunk_size, **read_kwargs
    )
    n_chunks = 0
    try:
        for chunk in reader:
            chunks.append(chunk)
            n_chunks += 1
    except UnicodeDecodeError:
        # The bad byte appeared mid-file — restart with fallback encoding
        logger.warning(
            "Encoding %s failed mid-read for %s — retrying with %s.",
            enc_to_use, filepath.name, fallback_encoding,
        )
        enc_to_use = fallback_encoding
        chunks = []
        reader = pd.read_csv(
            filepath, encoding=enc_to_use, chunksize=chunk_size, **read_kwargs
        )
        n_chunks = 0
        for chunk in reader:
            chunks.append(chunk)
            n_chunks += 1

    logger.info(
        "Loaded %s (%d chunks, encoding=%s)",
        filepath.name, n_chunks, enc_to_use,
    )
    return pd.concat(chunks, ignore_index=True)


# ================================================================== #
#  DATAFRAME SAVING                                                   #
# ================================================================== #

def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Save DataFrame as Parquet (Snappy compression)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved Parquet: %s  (%d rows × %d cols)", path.name, *df.shape)


def save_feather(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Save DataFrame as Feather."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(path)
    logger.info("Saved Feather: %s  (%d rows × %d cols)", path.name, *df.shape)


def save_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Save DataFrame as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved CSV: %s  (%d rows × %d cols)", path.name, *df.shape)


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, pathlib.Path],
    fmt: str = "parquet",
) -> None:
    """Save a DataFrame in the requested format (parquet|feather|csv)."""
    fmt = fmt.lower()
    path = pathlib.Path(path)
    if fmt == "parquet":
        save_parquet(df, path.with_suffix(".parquet"))
    elif fmt == "feather":
        save_feather(df, path.with_suffix(".feather"))
    elif fmt == "csv":
        save_csv(df, path.with_suffix(".csv"))
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use parquet|feather|csv.")
