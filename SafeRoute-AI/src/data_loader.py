"""
data_loader.py — SafeRoute AI
Load the single CSV (or multiple CSVs) from data/raw/, clean and validate,
return a tidy DataFrame ready for analysis.

Assumptions:
- Data lives in data/raw/ relative to the project root (SafeRoute-AI/).
- The dataset is per-accident records (not aggregated state/year), so EDA and
  modeling will aggregate as needed.
- `festival` column has ~99 % nulls — filled with "none".
- `risk_score` is already computed in the source data; we use it but also
  derive a severity-based tier for the classification model.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve the data/raw directory relative to this file's location ──────────
_SRC_DIR = pathlib.Path(__file__).parent          # SafeRoute-AI/src/
_PROJECT_ROOT = _SRC_DIR.parent                   # SafeRoute-AI/
DATA_RAW = _PROJECT_ROOT / "data" / "raw"

# ── State-name normalisation map ─────────────────────────────────────────────
# (Handles common spelling variants found in Indian road-safety datasets)
STATE_ALIASES: dict[str, str] = {
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "pondichery": "Puducherry",
    "uttaranchal": "Uttarakhand",
    "jammu & kashmir": "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir",
    "nct of delhi": "Delhi",
    "delhi (nct)": "Delhi",
    "andaman & nicobar": "Andaman & Nicobar Islands",
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "dadra & nagar haveli": "Dadra & Nagar Haveli",
    "daman & diu": "Daman & Diu",
    "d&n haveli": "Dadra & Nagar Haveli",
}


def _normalise_state(name: str) -> str:
    """Lower-strip then apply the alias map; fall back to title-case."""
    key = name.strip().lower()
    return STATE_ALIASES.get(key, name.strip().title())


def load_raw() -> pd.DataFrame:
    """Read every CSV in data/raw/ and concatenate into one DataFrame."""
    csv_files = sorted(DATA_RAW.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_RAW}")

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp, low_memory=False)
        df["_source_file"] = fp.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise, cast dtypes, fill known sparse columns."""
    df = df.copy()

    # ── State normalisation ──────────────────────────────────────────────────
    if "state" in df.columns:
        original_states = df["state"].unique().tolist()
        df["state"] = df["state"].astype(str).apply(_normalise_state)
        normalised_states = df["state"].unique().tolist()
        changed = set(original_states) - set(normalised_states)
        if changed:
            print(f"[data_loader] State names normalised: {sorted(changed)}")

    # ── Date parsing ─────────────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

    # ── Fill sparse columns ───────────────────────────────────────────────────
    if "festival" in df.columns:
        df["festival"] = df["festival"].fillna("none")

    # ── Cast numeric columns ──────────────────────────────────────────────────
    num_cols = ["temperature", "vehicles_involved", "casualties", "lanes",
                "hour", "traffic_signal", "is_weekend", "is_peak_hour"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop full duplicates ─────────────────────────────────────────────────
    n_before = len(df)
    if "accident_id" in df.columns:
        df = df.drop_duplicates(subset=["accident_id"])
    else:
        df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[data_loader] Dropped {n_dropped} duplicate rows.")

    # ── Severity tier (used as classification target) ─────────────────────────
    severity_map = {"minor": "Low", "major": "Medium", "fatal": "High"}
    if "accident_severity" in df.columns:
        df["severity_tier"] = df["accident_severity"].map(severity_map).fillna("Low")

    return df


def validate(df: pd.DataFrame) -> dict:
    """Return a data-quality summary dict (for display in Streamlit)."""
    total_rows = len(df)
    quality: dict = {
        "total_records": total_rows,
        "total_columns": len(df.columns),
        "duplicate_accident_ids": (
            df["accident_id"].duplicated().sum() if "accident_id" in df.columns else 0
        ),
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": df.select_dtypes(include="number").describe().round(3).to_dict(),
        "states_found": sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else [],
        "date_range": (
            str(df["date"].min().date()) + " to " + str(df["date"].max().date())
            if "date" in df.columns and not df["date"].isna().all()
            else "N/A"
        ),
        "source_files": df["_source_file"].unique().tolist() if "_source_file" in df.columns else [],
    }
    return quality


def load_and_prepare() -> Tuple[pd.DataFrame, dict]:
    """
    Public entry-point.  Returns (clean_df, quality_report).
    """
    raw = load_raw()
    clean_df = clean(raw)
    report = validate(clean_df)
    return clean_df, report


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, qr = load_and_prepare()
    print(f"\n[OK] Loaded {qr['total_records']:,} records across {qr['total_columns']} columns.")
    print(f"[OK] Date range : {qr['date_range']}")
    print(f"[OK] States     : {qr['states_found']}")
    print(f"[OK] Source CSVs: {qr['source_files']}")
    print(f"\nNull counts (non-zero only):")
    for col, cnt in qr["null_counts"].items():
        if cnt > 0:
            print(f"  {col}: {cnt} ({qr['null_pct'][col]:.1f}%)")
