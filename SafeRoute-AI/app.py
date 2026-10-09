"""
app.py — SafeRoute AI  |  Streamlit entry point
Run:  streamlit run app.py

Tabs
  1. Overview        — project context, SDG alignment, dataset at-a-glance
  2. Data Quality    — null audit, duplicate check, dtype summary
  3. EDA             — trend, state comparison, cause/weather/road breakdowns
  4. Model & Risk    — RF classifier metrics + state composite-risk ranking
"""

from __future__ import annotations

import sys
import pathlib

# ── Make src/ importable regardless of CWD ───────────────────────────────────
_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeRoute AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Caching data / model ──────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data…")
def get_data():
    from data_loader import load_and_prepare
    return load_and_prepare()


@st.cache_resource(show_spinner="Training Random Forest (first run only)…")
def get_model_metrics(df: pd.DataFrame):
    from modeling import train, state_risk_ranking
    clf, metrics = train(df)
    ranking = state_risk_ranking(df)
    return clf, metrics, ranking


# ── Load data ─────────────────────────────────────────────────────────────────
df, quality_report = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/UN_emblem_blue.svg/200px-UN_emblem_blue.svg.png",
        width=60,
    )
    st.title("SafeRoute AI")
    st.caption("Road Safety Risk Analysis · UN SDG 3.6 & 11.2")
    st.markdown("---")

    st.markdown("**Filter data**")
    all_states = sorted(df["state"].dropna().unique().tolist())
    selected_states = st.multiselect(
        "States / UTs", all_states, default=all_states, key="state_filter"
    )

    year_min = int(df["year"].min()) if "year" in df.columns else 2022
    year_max = int(df["year"].max()) if "year" in df.columns else 2025
    year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.markdown("**Top-N for charts**")
    top_n = st.slider("Top N states (bar chart)", 3, len(all_states), min(7, len(all_states)))

    st.markdown("---")
    st.caption("Data: `data/raw/indian_roads_dataset.csv`")
    st.caption(f"Records: {quality_report['total_records']:,}")

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = df["state"].isin(selected_states)
if "year" in df.columns:
    mask &= df["year"].between(year_range[0], year_range[1])
fdf = df[mask].copy()

# ── Helper ────────────────────────────────────────────────────────────────────
def section(title: str, icon: str = ""):
    st.markdown(f"### {icon} {title}" if icon else f"### {title}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Overview", "🔍 Data Quality", "📊 EDA", "🤖 Model & Risk Ranking"]
)

with tab1:
    st.markdown(
        """
        <h1 style='margin-bottom:0'>🛣️ SafeRoute AI</h1>
        <p style='color:#57606a;font-size:1.05rem;margin-top:4px'>
        Lightweight road-safety accident-frequency analysis for India &nbsp;·&nbsp;
        <b>UN SDG 3.6</b> (halve global road traffic deaths)  &nbsp;&amp;&nbsp;
        <b>SDG 11.2</b> (safe, affordable transport for all)
        </p>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(fdf):,}")
    col2.metric("States / UTs", fdf["state"].nunique())
    col3.metric(
        "Date Range",
        quality_report["date_range"],
    )
    col4.metric("Avg Risk Score", f"{fdf['risk_score'].mean():.3f}" if "risk_score" in fdf.columns else "N/A")

    st.markdown("---")

    cola, colb = st.columns([1.6, 1])
    with cola:
        section("Project Description", "📋")
        st.markdown(
            """
SafeRoute AI analyses **per-accident records** (20,000 crashes across 7 Indian states,
2022–2025) to surface patterns in accident frequency, severity, and contributing factors.

**Scope of this MVP**
| Module | Purpose |
|---|---|
| `data_loader.py` | Load, normalise, clean, validate |
| `analysis.py` | EDA — temporal trends, state comparison, cause / weather breakdowns |
| `modeling.py` | Random Forest severity classifier + composite state risk ranking |
| `app.py` | This Streamlit UI |

**⚠️ Important caveat — no exposure/volume data**
We do not have vehicle-km or traffic-volume figures per state, so all metrics are
*accident frequency counts*, not true *accident rates*.  States with more vehicles
naturally appear worse.  Treat rankings as descriptive, not causal.
            """
        )

    with colb:
        section("SDG Alignment", "🌐")
        st.markdown(
            """
**SDG 3.6** — *By 2030, halve the number of global deaths and injuries from road
traffic accidents.*

Findings from this tool can inform:
- Time-of-day / weather-specific enforcement priorities
- High-frequency state-level resource allocation

**SDG 11.2** — *Provide access to safe, affordable, accessible and sustainable
transport systems.*

Road-type severity breakdowns highlight where infrastructure investment
(signals, lane markings, junction design) is most needed.
            """
        )

    st.markdown("---")
    section("Dataset at a Glance", "📂")
    st.dataframe(fdf.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    section("Data Quality Report", "🔍")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{quality_report['total_records']:,}")
    c2.metric("Columns", quality_report["total_columns"])
    c3.metric("Duplicate IDs", quality_report["duplicate_accident_ids"])

    st.markdown("#### Null Counts & Percentages")
    null_df = (
        pd.DataFrame(
            {
                "Column": list(quality_report["null_counts"].keys()),
                "Null Count": list(quality_report["null_counts"].values()),
                "Null %": list(quality_report["null_pct"].values()),
            }
        )
        .sort_values("Null %", ascending=False)
    )
    # Colour high-null rows
    def highlight_nulls(row):
        if row["Null %"] > 50:
            return ["background-color:#fde8e8"] * len(row)
        if row["Null %"] > 10:
            return ["background-color:#fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(null_df.style.apply(highlight_nulls, axis=1), use_container_width=True)

    st.markdown("#### Column Data Types")
    dtype_df = pd.DataFrame(
        {"Column": list(quality_report["dtypes"].keys()),
         "Dtype": list(quality_report["dtypes"].values())}
    )
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown("#### Numeric Summary")
    num_summary = pd.DataFrame(quality_report["numeric_summary"]).T
    num_summary.index.name = "Column"
    st.dataframe(num_summary.reset_index(), use_container_width=True)

    st.markdown("#### Source Files Detected")
    for f in quality_report["source_files"]:
        st.code(f)

    st.markdown("#### States Found (after normalisation)")
    st.write(", ".join(quality_report["states_found"]))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    from analysis import (
        accidents_over_time,
        yearly_by_state,
        state_comparison,
        cause_breakdown,
        weather_breakdown,
        severity_by_road,
        risk_score_dist,
        hourly_heatmap,
    )

    section("Exploratory Data Analysis", "📊")
    st.caption(
        "All counts are **accident frequency** (raw counts). "
        "No traffic-volume denominator is available."
    )

    # ── Row 1: time trend + yearly by state ──────────────────────────────────
    r1a, r1b = st.columns(2)
    with r1a:
        fig, tbl = accidents_over_time(fdf)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Show data"):
                st.dataframe(tbl, use_container_width=True)
        else:
            st.info("Date column not available.")

    with r1b:
        fig, tbl = yearly_by_state(fdf)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Show data"):
                st.dataframe(tbl, use_container_width=True)
        else:
            st.info("Year/State columns not available.")

    st.markdown("---")

    # ── Row 2: state comparison bar ───────────────────────────────────────────
    fig, tbl = state_comparison(fdf, top_n=top_n)
    if fig:
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Show data"):
            st.dataframe(tbl, use_container_width=True)

    st.markdown("---")

    # ── Row 3: cause & weather ────────────────────────────────────────────────
    r3a, r3b = st.columns([1.6, 1])
    with r3a:
        fig, tbl = cause_breakdown(fdf)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Show data"):
                st.dataframe(tbl, use_container_width=True)

    with r3b:
        fig, tbl = weather_breakdown(fdf)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Show data"):
                st.dataframe(tbl, use_container_width=True)

    st.markdown("---")

    # ── Row 4: severity by road + risk score dist ─────────────────────────────
    r4a, r4b = st.columns(2)
    with r4a:
        fig, tbl = severity_by_road(fdf)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Show data"):
                st.dataframe(tbl, use_container_width=True)

    with r4b:
        fig, tbl = risk_score_dist(fdf)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
            with st.expander("Show stats"):
                st.dataframe(tbl, use_container_width=True)

    st.markdown("---")

    # ── Row 5: hourly heatmap (full width) ────────────────────────────────────
    fig, tbl = hourly_heatmap(fdf)
    if fig:
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Show data"):
            st.dataframe(tbl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL & RISK RANKING
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    from modeling import (
        plot_feature_importance,
        plot_confusion_matrix,
        state_risk_ranking,
    )

    section("Random Forest Classifier", "🤖")
    st.caption(
        "Target: **severity_tier** (Low / Medium / High) derived from `accident_severity`.\n"
        "Train/test split: 80/20, stratified."
    )

    # Train on FULL dataset (not filtered) for stable metrics
    clf, metrics, ranking = get_model_metrics(df)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc_col, train_col, test_col = st.columns(3)
    acc_col.metric("Overall Accuracy", f"{metrics['accuracy']:.1%}")
    train_col.metric("Train samples", f"{metrics['n_train']:,}")
    test_col.metric("Test samples", f"{metrics['n_test']:,}")

    st.markdown("#### Per-Class Metrics (test set)")
    report_dict = metrics["classification_report"]
    report_rows = []
    for label in ["Low", "Medium", "High"]:
        if label in report_dict:
            r = report_dict[label]
            report_rows.append({
                "Class": label,
                "Precision": round(r["precision"], 3),
                "Recall": round(r["recall"], 3),
                "F1-Score": round(r["f1-score"], 3),
                "Support": int(r["support"]),
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

    st.markdown("---")

    # ── Plots ─────────────────────────────────────────────────────────────────
    col_fi, col_cm = st.columns(2)
    with col_fi:
        fig = plot_feature_importance(metrics["feature_importance"])
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Feature importance table"):
            st.dataframe(metrics["feature_importance"], use_container_width=True)

    with col_cm:
        fig = plot_confusion_matrix(metrics["confusion_matrix"], metrics["classes"])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── State Risk Ranking ────────────────────────────────────────────────────
    section("State Composite Risk Ranking", "🏆")
    st.caption(
        "Composite score = weighted sum of normalised: avg risk_score (0.35) · "
        "fatal rate (0.30) · avg casualties (0.20) · avg vehicles involved (0.15).\n"
        "**No traffic-volume denominator** — treat as frequency-based ranking only."
    )

    # Re-rank on filtered data if user changed state/year filter
    filtered_ranking = state_risk_ranking(fdf)

    if not filtered_ranking.empty:
        # Colour tier column
        def color_tier(val):
            colors = {"High": "#fde8e8", "Medium": "#fff3cd", "Low": "#e8f5e9"}
            return f"background-color:{colors.get(val, '')}"

        styled = filtered_ranking.style.applymap(color_tier, subset=["risk_tier"])
        st.dataframe(styled, use_container_width=True)

        # Bar chart
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(8, max(3, len(filtered_ranking) * 0.45)))
        tier_colors = {
            "High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"
        }
        bar_colors = [
            tier_colors.get(str(t), "#1f77b4")
            for t in filtered_ranking["risk_tier"]
        ]
        ax.barh(
            filtered_ranking["state"],
            filtered_ranking["composite_score"],
            color=bar_colors,
            edgecolor="white",
        )
        ax.invert_yaxis()
        ax.set_xlabel("Composite Risk Score (0–1)")
        ax.set_title("State Risk Ranking (composite)", fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No data available for the selected filters.")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.82rem;color:#57606a;border-top:1px solid #e5e7eb;padding-top:8px'>
        <b>Honesty notice:</b> Accuracy figures are computed on real held-out test data,
        not fabricated.  The composite risk score is a transparent formula — inspect the
        weights in <code>src/modeling.py → state_risk_ranking()</code>.
        No causal claims are made; all results are observational frequency summaries.
        </div>
        """,
        unsafe_allow_html=True,
    )
