# CIC-IDS2017 Preprocessing Pipeline

End-to-end preprocessing pipeline that transforms raw CIC-IDS2017 CSVs into clean, ML-ready train/val/test splits in Parquet format — ready for Logistic Regression, Random Forest, XGBoost, and Isolation Forest.

---

## Quick Start

```powershell
# 1. Create virtual environment
cd "c:\Users\kevin\OneDrive - Heriot-Watt University\Desktop\Dissertation AG"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place CIC-IDS2017 CSVs in data/raw/
#    (see "Dataset Acquisition" below)

# 4. Run the preprocessing pipeline
python scripts/run_preprocess.py

# 5. (Optional) Run sanity checks
python notebooks/01_sanity_checks.py

# 6. (Optional) Run baseline models
python -m src.models.baselines
```

---

## Dataset Acquisition

The CIC-IDS2017 dataset is published by the Canadian Institute for Cybersecurity.

### Official File Layout

The dataset comprises **8 daily CSV files** generated from pcap captures:

| Day | Filename | Attack types |
|-----|----------|-------------|
| Monday | `Monday-WorkingHours.pcap_ISCX.csv` | Benign only |
| Tuesday | `Tuesday-WorkingHours.pcap_ISCX.csv` | FTP-Patator, SSH-Patator |
| Wednesday | `Wednesday-workingHours.pcap_ISCX.csv` | DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest), Heartbleed |
| Thursday AM | `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Web Attacks (Brute Force, XSS, SQL Injection) |
| Thursday PM | `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Infiltration |
| Friday AM | `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Bot |
| Friday PM | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | PortScan |
| Friday PM | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | DDoS |

### How to Get the Files

1. **Download** from the official UNB mirror:
   - Visit: https://www.unb.ca/cic/datasets/ids-2017.html
   - Or search for a Kaggle mirror (e.g. "CIC-IDS2017 dataset kaggle")
2. **Extract** the CSVs and place them in `data/raw/`.
3. The pipeline auto-discovers any `.csv` file in that folder — filenames don't need to match exactly, but they must contain day keywords (`monday`, `tuesday`, etc.) for day-based splitting.

### Large File Notes

- Total dataset is ~1.6 GB across all CSVs (~2.8M rows, 79+ features).
- The pipeline uses chunked reading (configurable in `config.yaml` → `ingest.chunk_size`).
- Output Parquet files are ~60–70% smaller than CSV equivalents.
- For machines with <8 GB RAM, reduce `chunk_size` to 100,000.

---

## Project Structure

```
├── config.yaml              ← Central configuration
├── requirements.txt         ← Pinned dependencies
├── README.md
├── data/
│   ├── raw/                 ← Place CIC-IDS2017 CSVs here
│   └── processed/           ← Pipeline outputs (Parquet, scalers, etc.)
├── src/
│   ├── utils/
│   │   ├── io.py            ← Fast CSV reading, Parquet/Feather/CSV writers
│   │   └── logging.py       ← Structured logging
│   ├── data/
│   │   ├── ingest.py        ← Load & tag all daily CSVs
│   │   ├── clean.py         ← Column normalization, inf/NaN, dedup, outliers
│   │   ├── labels.py        ← Canonical label mapping, binary/multiclass
│   │   ├── split.py         ← Random / day-based / time-based splitting
│   │   ├── transform.py     ← Scaling, SMOTE/resampling, anomaly-det set
│   │   └── build_dataset.py ← Orchestrates the full pipeline
│   ├── features/
│   │   └── selection.py     ← Variance, correlation, mutual info filters
│   └── models/
│       └── baselines.py     ← LR, RF, XGBoost, Isolation Forest
├── scripts/
│   └── run_preprocess.py    ← CLI entry-point with argparse
├── notebooks/
│   └── 01_sanity_checks.py  ← Visualizations & leakage checks
├── models/                  ← Saved model artifacts
└── reports/                 ← Figures & summary stats
```

---

## Configuration (`config.yaml`)

All pipeline behaviour is controlled through `config.yaml`. Key sections:

| Section | Key Options |
|---------|-------------|
| `paths` | `raw_data_dir`, `processed_data_dir`, `log_file` |
| `cleaning` | `columns_to_drop`, `nan_strategy` (median/mean/zero/drop), `replace_inf_with`, `remove_duplicates`, `clip_outliers` |
| `labels` | `task_mode` (binary/multiclass/both) |
| `split` | `strategy` (random/day_based/time_based), `day_assignment` |
| `transform` | `scaler` (robust/standard/minmax/none), `save_format` |
| `imbalance` | `enabled`, `method` (smote/adasyn/smote_tomek/smote_enn/random_over/random_under) |
| `class_weights` | `enabled`, `mode` (balanced) |
| `anomaly_detection` | `enabled` — produce benign-only training set |
| `feature_selection` | `variance_threshold`, `correlation_threshold`, `mutual_info_top_k` |

### CLI Overrides

```powershell
python scripts/run_preprocess.py --task multiclass --split random --scaler standard --resample smote --seed 123
```

---

## Pipeline Outputs (`data/processed/`)

After a successful run, you'll find:

| File | Description |
|------|-------------|
| `X_train.parquet` | Training features (scaled) |
| `X_val.parquet` | Validation features (scaled) |
| `X_test.parquet` | Test features (scaled) |
| `y_train.parquet` | Training labels |
| `y_val.parquet` | Validation labels |
| `y_test.parquet` | Test labels |
| `scaler.joblib` | Fitted scaler object |
| `feature_names.joblib` | Ordered list of feature names |
| `class_weights.joblib` | Computed class weights dict |
| `label_map.joblib` | Label → integer mapping (multiclass) |
| `X_train_benign.parquet` | Benign-only train set (for Isolation Forest) |

---

## Design Decisions

### Columns Dropped (Default)

`Flow ID`, `Source IP`, `Destination IP`, `Timestamp`, `Source Port`, `Destination Port` — these leak identity/temporal information that wouldn't be available in a realistic deployment scenario (e.g. a production IDS that must generalize to unseen IPs/times).

### Split Strategy

**Day-based** is the default and recommended split:
- **Train**: Monday, Tuesday, Wednesday
- **Validation**: Thursday
- **Test**: Friday

This prevents temporal leakage — the model never sees future traffic patterns during training. Random splits inflate metrics because network traffic is temporally correlated.

### Imbalance Handling

Two levels are provided:

1. **Algorithm-level** (default ON): `class_weight="balanced"` is passed to sklearn models, giving higher weight to minority attack classes at zero computational cost.

2. **Data-level** (default OFF, toggle in config): SMOTE / ADASYN / hybrid methods resample the training set. Safeguards prevent SMOTE from crashing on classes with fewer than `k_neighbors + 1` samples.

### CIC-IDS2017 Known Quirks Handled

- **Messy column names**: whitespace, mixed case, special characters → normalized
- **Infinite values**: division-by-zero artifacts in flow features → replaced with NaN, then imputed
- **Label inconsistencies**: "Web Attack – Brute Force" vs "Web Attack  Brute Force" → canonical mapping
- **Duplicate rows**: exact duplicates removed (common in CIC-IDS2017)
- **Non-numeric artifacts**: string values in numeric columns coerced, then imputed

---

## VS Code Setup

```powershell
# Select Python interpreter in VS Code:
# 1. Press Ctrl+Shift+P
# 2. Type "Python: Select Interpreter"
# 3. Choose .\.venv\Scripts\python.exe
```

For IntelliSense to work on `src.*` imports, ensure your workspace root is this project folder.

---

## Running Baseline Models

After preprocessing:

```powershell
python -m src.models.baselines
```

This trains Logistic Regression, Random Forest, XGBoost, and Isolation Forest — printing classification reports and ROC-AUC for each. Use it to validate the dataset produces sensible metrics.
