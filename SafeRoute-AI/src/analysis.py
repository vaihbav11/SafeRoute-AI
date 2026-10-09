"""
analysis.py — SafeRoute AI
EDA functions that return Matplotlib figures and summary DataFrames.

NOTE ON TERMINOLOGY:
  All metrics here are "accident frequency" counts (or averages).
  We do NOT have traffic-volume / vehicle-km exposure data, so we
  cannot compute true "accident rates".  The word "risk" is avoided
  in chart labels; "frequency" is used instead.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ── Shared palette ────────────────────────────────────────────────────────────
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
SEVERITY_COLORS = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}


def _savefig(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout()
    return fig


# ── 1. Accidents over time ────────────────────────────────────────────────────
def accidents_over_time(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Monthly accident count trend."""
    if "date" not in df.columns:
        return None, pd.DataFrame()

    monthly = (
        df.set_index("date")
        .resample("ME")["accident_id"]
        .count()
        .reset_index()
        .rename(columns={"accident_id": "count", "date": "month"})
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["month"], monthly["count"], color=PALETTE[0], linewidth=1.8)
    ax.fill_between(monthly["month"], monthly["count"], alpha=0.15, color=PALETTE[0])
    ax.set_title("Monthly Accident Frequency (all states)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Accident Count")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _savefig(fig), monthly


# ── 2. Yearly trend by state ──────────────────────────────────────────────────
def yearly_by_state(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Line chart — one line per state, yearly accident count."""
    if "year" not in df.columns or "state" not in df.columns:
        return None, pd.DataFrame()

    agg = (
        df.groupby(["year", "state"])["accident_id"]
        .count()
        .reset_index()
        .rename(columns={"accident_id": "count"})
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (state, grp) in enumerate(agg.groupby("state")):
        ax.plot(grp["year"], grp["count"], marker="o",
                label=state, color=PALETTE[i % len(PALETTE)], linewidth=1.8)

    ax.set_title("Yearly Accident Frequency by State", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Accident Count")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _savefig(fig), agg


# ── 3. State comparison (total accidents) ────────────────────────────────────
def state_comparison(df: pd.DataFrame, top_n: int = 10) -> tuple[plt.Figure, pd.DataFrame]:
    """Horizontal bar chart — top-N states by total accident count."""
    if "state" not in df.columns:
        return None, pd.DataFrame()

    counts = (
        df.groupby("state")["accident_id"]
        .count()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"accident_id": "total_accidents"})
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.45)))
    bars = ax.barh(counts["state"], counts["total_accidents"],
                   color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} States by Accident Frequency (all years)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Accidents")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in bars:
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    return _savefig(fig), counts


# ── 4. Breakdown by cause ─────────────────────────────────────────────────────
def cause_breakdown(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Stacked bar — accident causes by state."""
    if "cause" not in df.columns or "state" not in df.columns:
        return None, pd.DataFrame()

    pivot = (
        df.groupby(["state", "cause"])["accident_id"]
        .count()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", linewidth=0.4)
    ax.set_title("Accident Frequency by Cause and State", fontsize=13, fontweight="bold")
    ax.set_xlabel("State")
    ax.set_ylabel("Accident Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Cause", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _savefig(fig), pivot.reset_index()


# ── 5. Breakdown by weather ───────────────────────────────────────────────────
def weather_breakdown(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Pie chart — accident share by weather condition."""
    if "weather" not in df.columns:
        return None, pd.DataFrame()

    counts = (
        df.groupby("weather")["accident_id"]
        .count()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(counts)],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Accidents by Weather Condition", fontsize=13, fontweight="bold")
    return _savefig(fig), counts.reset_index().rename(columns={"accident_id": "count"})


# ── 6. Severity distribution by road type ─────────────────────────────────────
def severity_by_road(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Grouped bar — severity tier breakdown per road type."""
    if "road_type" not in df.columns or "severity_tier" not in df.columns:
        return None, pd.DataFrame()

    pivot = (
        df.groupby(["road_type", "severity_tier"])["accident_id"]
        .count()
        .unstack(fill_value=0)
    )
    # Ensure column order
    col_order = [c for c in ["Low", "Medium", "High"] if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(pivot))
    width = 0.25
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[col],
               width=width, label=col,
               color=SEVERITY_COLORS.get(col, PALETTE[i]),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_title("Severity Distribution by Road Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accident Count")
    ax.legend(title="Severity Tier")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _savefig(fig), pivot.reset_index()


# ── 7. Risk score distribution ────────────────────────────────────────────────
def risk_score_dist(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Histogram of the pre-computed risk_score field."""
    if "risk_score" not in df.columns:
        return None, pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["risk_score"].dropna(), bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.4)
    ax.axvline(df["risk_score"].mean(), color=PALETTE[3], linestyle="--",
               linewidth=1.5, label=f"Mean = {df['risk_score'].mean():.2f}")
    ax.set_title("Distribution of Risk Scores (pre-computed in source data)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    stats = df["risk_score"].describe().reset_index()
    stats.columns = ["statistic", "value"]
    return _savefig(fig), stats


# ── 8. Hour-of-day heatmap ────────────────────────────────────────────────────
def hourly_heatmap(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Heatmap — accident count by hour × day-of-week."""
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        return None, pd.DataFrame()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        df.groupby(["day_of_week", "hour"])["accident_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(day_order)
    )

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Accident Count")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7)
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order, fontsize=9)
    ax.set_title("Accident Frequency by Hour × Day-of-Week", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    return _savefig(fig), pivot.reset_index()


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(os.path.dirname(__file__)))
    from data_loader import load_and_prepare

    df, _ = load_and_prepare()
    fig, tbl = accidents_over_time(df)
    print(f"[OK] accidents_over_time: {len(tbl)} monthly buckets")
    fig2, tbl2 = state_comparison(df)
    print(f"[OK] state_comparison:\n{tbl2}")
    print("[OK] analysis.py OK")
