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

# ── Global UI styling ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700&display=swap');

    /* ── Base typography ── */
    html, body, [class*="css"], .stMarkdown, .stText, p, li, span, label {
        font-family: 'Inter', sans-serif !important;
        font-size: 14.5px;
        line-height: 1.7;
        color: #1f2328;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
        color: #0f172a !important;
    }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.15rem !important; }

    /* ── Main background ── */
    .stApp { background-color: #f8fafc; }
    .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem !important;
        max-width: 1280px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: none !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-family: 'Sora', sans-serif !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #3b82f6 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 6px 8px;
        gap: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        border-radius: 8px !important;
        padding: 6px 16px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.18s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #0f172a !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"]    { display: none !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 22px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748b !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Sora', sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        border-radius: 8px !important;
        border: 1.5px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #1e293b !important;
        padding: 6px 18px !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background: #0f172a !important;
        color: #ffffff !important;
        border-color: #0f172a !important;
        box-shadow: 0 4px 10px rgba(15,23,42,0.18) !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        border-radius: 8px !important;
        background: #0f172a !important;
        color: #ffffff !important;
        border: none !important;
        padding: 8px 22px !important;
    }
    .stDownloadButton > button:hover {
        background: #1e40af !important;
    }

    /* ── Selectbox / Multiselect ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 8px !important;
        border-color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 6px !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border-color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
        border: 1px solid #e2e8f0 !important;
    }
    iframe[title="st_aggrid"] { border-radius: 10px !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        border-radius: 8px !important;
    }

    /* ── Info / Warning / Error boxes ── */
    .stAlert {
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }

    /* ── Divider ── */
    hr { border-color: #e2e8f0 !important; margin: 1.5rem 0 !important; }

    /* ── Code blocks ── */
    code {
        font-size: 13px !important;
        background: #f1f5f9 !important;
        border-radius: 5px !important;
        padding: 2px 6px !important;
        color: #0f172a !important;
    }

    /* ── Caption text ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 12.5px !important;
        color: #64748b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "🏠 Overview",
        "🔍 Data Quality",
        "📊 EDA",
        "🤖 Model & Risk Ranking",
        "🗺️ Risk Map",
        "💡 Recommendations",
        "💬 Ask SafeRoute",
    ]
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    from risk_map import build_map

    section("India Accident Frequency Map", "🗺️")
    st.caption(
        "⚠️ **Accident frequency by state** — NOT an exposure-adjusted risk rate. "
        "States with more registered vehicles or better reporting will show higher counts. "
        "No traffic-volume denominator is available. Colour tiers are the same composite "
        "frequency tiers from the Model & Risk Ranking tab."
    )

    @st.cache_resource(show_spinner="Building risk map…")
    def get_map(_ranking):
        # _ranking prefix prevents st.cache_resource hashing the DataFrame
        return build_map(_ranking)

    ranking_for_map, _, _ = get_model_metrics(df)  # reuse cached clf, metrics, ranking
    _, _, ranking_for_map = get_model_metrics(df)

    map_fig, unmatched_states, map_err = get_map(ranking_for_map)

    if map_err:
        st.error(map_err)
    else:
        if unmatched_states:
            st.warning(
                f"⚠️ The following states in the accident data had **no matching GeoJSON feature** "
                f"and will appear grey on the map: **{', '.join(unmatched_states)}**. "
                "This is usually a name-spelling difference between the GeoJSON and the dataset. "
                "See `src/risk_map.py → _GEOJSON_TO_DATA` to add a mapping."
            )
        import plotly.graph_objects as go
        st.plotly_chart(map_fig, use_container_width=True)

        with st.expander("State ranking table"):
            st.dataframe(ranking_for_map, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    from recommendations import (
        get_recommendations,
        generate_safety_brief,
        state_summary_stats,
        top_causes,
    )

    section("Policy Recommendation Panel", "💡")
    st.caption(
        "Rule-based recommendations triggered by transparent if/then thresholds — "
        "no new ML model. Each recommendation shows the actual triggering number. "
        "_This flags where further investigation is warranted, not a guarantee an "
        "intervention prevents crashes._"
    )

    all_states_rec = sorted(df["state"].dropna().unique().tolist())
    sel_state = st.selectbox("Select a State / UT", all_states_rec, key="rec_state")

    recs = get_recommendations(df, sel_state)

    # ── State summary metrics ─────────────────────────────────────────────────
    stats_dict = state_summary_stats(df, sel_state)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Accidents", f"{stats_dict['total_records']:,}")
    if "total_fatal" in stats_dict:
        m2.metric("Fatal", f"{stats_dict['total_fatal']:,}")
        m3.metric("Major", f"{stats_dict['total_major']:,}")
        m4.metric("Minor", f"{stats_dict['total_minor']:,}")

    st.markdown("---")

    # ── Recommendations ───────────────────────────────────────────────────────
    tier_icons = {
        "Enforcement / Awareness": "🚔",
        "Emergency Response / Infrastructure": "🚑",
        "Infrastructure / Enforcement": "🛣️",
        "Infrastructure": "🏗️",
        "Multi-Sector Review": "📋",
        "Monitoring": "📊",
    }
    for rec in recs:
        icon = tier_icons.get(rec.category, "💡")
        with st.container():
            st.markdown(
                f"#### {icon} {rec.title}",
            )
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(rec.detail)
            with col_b:
                st.info(f"**Trigger:** {rec.trigger_value}\n\n**Category:** {rec.category}")
            st.markdown("---")

    # ── Safety Brief download ─────────────────────────────────────────────────
    section(f"Download Safety Brief for {sel_state}", "📄")
    brief_md = generate_safety_brief(df, sel_state, recs)
    st.download_button(
        label=f"⬇️ Download Safety Brief for {sel_state}",
        data=brief_md,
        file_name=f"SafeRoute_Brief_{sel_state.replace(' ', '_')}.md",
        mime="text/markdown",
    )
    with st.expander("Preview brief"):
        st.markdown(brief_md)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ASK SAFEROUTE (Gemini Chat)
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    from chat_assistant import get_gemini_api_key, chat as gemini_chat

    section("Ask SafeRoute", "💬")

    api_key = get_gemini_api_key()

    if not api_key:
        st.warning(
            "🔑 **Gemini API key not found.**\n\n"
            "To enable the chat assistant:\n"
            "- **Local:** set the environment variable `GEMINI_API_KEY=<your_key>` "
            "before running `streamlit run app.py`.\n"
            "- **Streamlit Cloud:** add `GEMINI_API_KEY` to your app's Secrets "
            "(Settings → Secrets).\n\n"
            "Get a free key at [aistudio.google.com](https://aistudio.google.com)."
        )
    else:
        st.caption("Ask me anything — road safety, general knowledge, or anything else.")

        # ── Starter prompt buttons ────────────────────────────────────────────
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        starter_prompts = [
            "Which states have the highest accident risk?",
            "What are common causes of road accidents in India?",
            "Give me road safety tips for highway driving.",
        ]

        if not st.session_state.chat_history:
            st.markdown("**Quick questions:**")
            cols = st.columns(3)
            for i, prompt in enumerate(starter_prompts):
                if cols[i].button(prompt, key=f"starter_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.spinner("Thinking…"):
                        reply = gemini_chat(st.session_state.chat_history, api_key)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

        # ── Chat history display ──────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Chat input ────────────────────────────────────────────────────────
        user_input = st.chat_input("Ask anything…")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = gemini_chat(st.session_state.chat_history, api_key)
                st.markdown(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # ── Clear chat ────────────────────────────────────────────────────────
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()
