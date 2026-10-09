"""
modeling.py — SafeRoute AI

Model: Random Forest classifier.
Target: `severity_tier` (Low / Medium / High) derived from `accident_severity`.

Dataset size (20,000 per-crash records with 7 states) is adequate for a
Random Forest classifier.  We deliberately avoid building a "state-level
aggregate" model because aggregating 20 k rows into ~28 state-year combos
would leave far too few samples for a meaningful train/test split.

Features used (all categorical or numeric — RF handles both):
  - hour, day_of_week (encoded), is_weekend, is_peak_hour
  - road_type (encoded), lanes, traffic_signal
  - weather (encoded), visibility (encoded)
  - temperature, traffic_density (encoded)
  - cause (encoded)
  - vehicles_involved, casualties
  - risk_score   ← pre-computed in the source dataset; included as a feature
                   so the model can learn its relationship to severity tier.
"""

from __future__ import annotations

import os
import pathlib
import warnings
from typing import Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

_SRC_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent
MODELS_DIR = _PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"

FEATURE_COLS = [
    "hour", "is_weekend", "is_peak_hour",
    "road_type", "lanes", "traffic_signal",
    "weather", "visibility",
    "temperature", "traffic_density",
    "cause", "vehicles_involved", "casualties",
    "risk_score", "day_of_week",
]
TARGET_COL = "severity_tier"
TIER_ORDER = ["Low", "Medium", "High"]


# ── Encoding ──────────────────────────────────────────────────────────────────
_encoders: dict[str, LabelEncoder] = {}

def _encode(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            _encoders[col] = le
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract and encode feature matrix X and target vector y."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        print(f"[modeling] Skipping missing feature columns: {missing}")

    X = df[available].copy()

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    X = _encode(X, cat_cols)

    # Fill any remaining NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))

    y = df[TARGET_COL].copy()
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, n_estimators: int = 200, random_state: int = 42):
    """
    Train Random Forest classifier.  Returns (model, metrics_dict, feature_importance_df).
    Saves the trained model to models/model.pkl.
    """
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=TIER_ORDER)

    fi = (
        pd.DataFrame({"feature": X.columns, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "accuracy": report["accuracy"],
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": fi,
        "classes": TIER_ORDER,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features_used": X.columns.tolist(),
    }

    # Save model
    joblib.dump({"model": clf, "encoders": _encoders, "features": X.columns.tolist()}, MODEL_PATH)
    print(f"[modeling] Model saved to: {MODEL_PATH}")
    print(f"[modeling] Accuracy: {metrics['accuracy']:.3f}")

    return clf, metrics


# ── Load saved model ──────────────────────────────────────────────────────────
def load_model():
    """Load model.pkl if it exists, else return None."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_feature_importance(fi: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    top = fi.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color="#3b82d4", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, classes: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (test set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── State Risk Ranking (scoring fallback AND primary output) ──────────────────
def state_risk_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a transparent, composite risk ranking per state.

    Score components (all normalised 0-1 before weighting):
      1. avg_risk_score          (from source data)     weight 0.35
      2. fatal_rate              (fatal / total)         weight 0.30
      3. avg_casualties_per_acc                          weight 0.20
      4. avg_vehicles_involved                           weight 0.15

    No traffic-volume denominator is available, so this is frequency-based.
    """
    if "state" not in df.columns:
        return pd.DataFrame()

    g = df.groupby("state")

    agg = pd.DataFrame({
        "total_accidents": g["accident_id"].count(),
        "avg_risk_score": g["risk_score"].mean(),
        "fatal_rate": g.apply(
            lambda x: (x["accident_severity"] == "fatal").mean()
        ),
        "avg_casualties": g["casualties"].mean(),
        "avg_vehicles": g["vehicles_involved"].mean(),
    }).reset_index()

    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx > mn else pd.Series([0.5] * len(s), index=s.index)

    agg["composite_score"] = (
        _norm(agg["avg_risk_score"]) * 0.35
        + _norm(agg["fatal_rate"]) * 0.30
        + _norm(agg["avg_casualties"]) * 0.20
        + _norm(agg["avg_vehicles"]) * 0.15
    )

    agg = agg.sort_values("composite_score", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1

    # Assign tier labels based on composite score
    thresholds = agg["composite_score"].quantile([1 / 3, 2 / 3])
    agg["risk_tier"] = pd.cut(
        agg["composite_score"],
        bins=[-0.001, thresholds.iloc[0], thresholds.iloc[1], 1.001],
        labels=["Low", "Medium", "High"],
    )

    return agg[[
        "rank", "state", "total_accidents",
        "avg_risk_score", "fatal_rate", "avg_casualties",
        "avg_vehicles", "composite_score", "risk_tier",
    ]].round(4)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_SRC_DIR))
    from data_loader import load_and_prepare

    df, _ = load_and_prepare()
    clf, metrics = train(df)
    print(f"[OK] Accuracy : {metrics['accuracy']:.3f}")
    print(f"[OK] Features : {metrics['features_used']}")
    ranking = state_risk_ranking(df)
    print(f"\n[OK] State Risk Ranking:\n{ranking.to_string(index=False)}")
    print("\n[OK] modeling.py OK")
