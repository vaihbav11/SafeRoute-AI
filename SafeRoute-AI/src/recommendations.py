"""
recommendations.py — SafeRoute AI
Rule-based policy recommendation panel and Safety Brief generator.

Design:
- Pure if/then logic — no new ML model.
- Works only with columns actually present in the DataFrame.
- Available columns in the current dataset:
    cause, weather, road_type, accident_severity, casualties,
    vehicles_involved, risk_score, state, year
- NOTE: No vehicle_type or road_user_category column exists in this dataset,
  so two-wheeler / pedestrian rules are SKIPPED.  Rules fire on:
    1. cause shares (state vs national)
    2. fatal rate (state vs national)
    3. weather-specific accident share (state vs national)
    4. road_type severe accident share (state vs national)
    5. overall risk score (state vs national)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


# ── Recommendation dataclass ──────────────────────────────────────────────────
@dataclass
class Recommendation:
    title: str
    detail: str
    trigger_value: str      # e.g. "State: 52.1% · National: 44.8%"
    category: str           # e.g. "Enforcement", "Infrastructure", "Awareness"


# ── Threshold constants ───────────────────────────────────────────────────────
# A state metric must exceed the national metric by at least this factor to fire.
EXCESS_FACTOR = 1.10   # 10% above national average


def _pct(n: float, total: float) -> float:
    return (n / total * 100) if total > 0 else 0.0


def get_recommendations(df: pd.DataFrame, state: str) -> list[Recommendation]:
    """
    Generate 2–4 rule-based recommendations for a given state.
    Only rules with columns present in df are evaluated.
    """
    recs: list[Recommendation] = []
    state_df = df[df["state"] == state]

    if len(state_df) == 0:
        return recs

    n_state = len(state_df)
    n_national = len(df)

    # ── Rule 1: High fatal rate ───────────────────────────────────────────────
    if "accident_severity" in df.columns:
        fatal_nat = _pct((df["accident_severity"] == "fatal").sum(), n_national)
        fatal_st  = _pct((state_df["accident_severity"] == "fatal").sum(), n_state)
        if fatal_st > fatal_nat * EXCESS_FACTOR:
            recs.append(Recommendation(
                title="Elevated Fatal Accident Rate — Emergency Response Review",
                detail=(
                    "This state's fatal accident rate significantly exceeds the national average. "
                    "Priority actions: review trauma centre response times, improve golden-hour "
                    "emergency access on high-frequency corridors, audit road geometry at known "
                    "fatality hotspots."
                ),
                trigger_value=f"State fatal rate: {fatal_st:.1f}% · National: {fatal_nat:.1f}%",
                category="Emergency Response / Infrastructure",
            ))

    # ── Rule 2: Dominant cause exceeds national share ─────────────────────────
    if "cause" in df.columns:
        nat_cause_shares = df["cause"].value_counts(normalize=True) * 100
        st_cause_shares  = state_df["cause"].value_counts(normalize=True) * 100

        for cause, st_share in st_cause_shares.items():
            nat_share = nat_cause_shares.get(cause, 0.0)
            if nat_share > 0 and st_share > nat_share * EXCESS_FACTOR and len(recs) < 4:
                cause_label = cause.replace("_", " ").title()
                action_map = {
                    "overspeeding":   "Targeted speed enforcement (cameras, checkpoints) on high-frequency corridors.",
                    "drunk driving":  "Increased night-time sobriety checkpoints and public awareness campaign.",
                    "distraction":    "Anti-distracted-driving enforcement and public signage campaign.",
                    "poor road":      "Road maintenance audit — surface quality, signage, lighting review.",
                    "weather":        "Weather-advisory signage and reduced speed limits during adverse conditions.",
                }
                action = action_map.get(cause, f"Targeted investigation into '{cause_label}' incidents.")
                recs.append(Recommendation(
                    title=f"Disproportionate '{cause_label}' Incidents",
                    detail=action,
                    trigger_value=(
                        f"State share: {st_share:.1f}% · National: {nat_share:.1f}%"
                    ),
                    category="Enforcement / Awareness",
                ))

    # ── Rule 3: High adverse-weather accident share ───────────────────────────
    if "weather" in df.columns:
        for bad_weather in ["fog", "rain"]:
            nat_w_share = _pct((df["weather"] == bad_weather).sum(), n_national)
            st_w_share  = _pct((state_df["weather"] == bad_weather).sum(), n_state)
            if nat_w_share > 0 and st_w_share > nat_w_share * EXCESS_FACTOR and len(recs) < 4:
                recs.append(Recommendation(
                    title=f"High '{bad_weather.title()}' Condition Accident Share",
                    detail=(
                        f"A higher proportion of accidents in this state occur during {bad_weather} "
                        f"conditions. Recommended: mandatory fog lights, variable speed limit signs, "
                        f"and highway patrol presence during low-visibility periods."
                    ),
                    trigger_value=(
                        f"State share in {bad_weather}: {st_w_share:.1f}% · National: {nat_w_share:.1f}%"
                    ),
                    category="Infrastructure / Enforcement",
                ))

    # ── Rule 4: Road-type severity concentration ──────────────────────────────
    if "road_type" in df.columns and "accident_severity" in df.columns:
        for rt in ["highway", "rural"]:
            rt_nat = df[df["road_type"] == rt]
            rt_st  = state_df[state_df["road_type"] == rt]
            if len(rt_nat) == 0 or len(rt_st) == 0:
                continue
            fatal_nat_rt = _pct((rt_nat["accident_severity"] == "fatal").sum(), len(rt_nat))
            fatal_st_rt  = _pct((rt_st["accident_severity"] == "fatal").sum(), len(rt_st))
            if fatal_st_rt > fatal_nat_rt * EXCESS_FACTOR and len(recs) < 4:
                recs.append(Recommendation(
                    title=f"High Fatal Rate on {rt.title()} Roads",
                    detail=(
                        f"Fatal accidents on {rt} roads in this state are disproportionately high. "
                        f"Recommended: geometric safety audit (curves, medians, run-off areas), "
                        f"rumble strips, and animal/pedestrian crossing management."
                    ),
                    trigger_value=(
                        f"State {rt} fatal rate: {fatal_st_rt:.1f}% · "
                        f"National {rt} fatal rate: {fatal_nat_rt:.1f}%"
                    ),
                    category="Infrastructure",
                ))

    # ── Rule 5: High average risk score ──────────────────────────────────────
    if "risk_score" in df.columns and len(recs) < 4:
        nat_avg_rs = df["risk_score"].mean()
        st_avg_rs  = state_df["risk_score"].mean()
        if st_avg_rs > nat_avg_rs * EXCESS_FACTOR:
            recs.append(Recommendation(
                title="Elevated Composite Risk Score",
                detail=(
                    "The pre-computed risk score for this state is above the national average. "
                    "Consider a multi-agency road-safety audit covering enforcement, infrastructure "
                    "quality, and emergency response capacity."
                ),
                trigger_value=(
                    f"State avg risk score: {st_avg_rs:.3f} · National avg: {nat_avg_rs:.3f}"
                ),
                category="Multi-Sector Review",
            ))

    # ── Fallback: no rules fired ──────────────────────────────────────────────
    if not recs:
        nat_avg_rs = df["risk_score"].mean() if "risk_score" in df.columns else None
        st_avg_rs  = state_df["risk_score"].mean() if "risk_score" in df.columns else None
        value = (
            f"State avg risk score: {st_avg_rs:.3f} · National avg: {nat_avg_rs:.3f}"
            if nat_avg_rs is not None else "No elevated metrics detected."
        )
        recs.append(Recommendation(
            title="No Metrics Significantly Above National Average",
            detail=(
                "This state does not exceed the national average on any tracked metric by "
                "the current thresholds (10% excess). Continue standard monitoring."
            ),
            trigger_value=value,
            category="Monitoring",
        ))

    return recs


# ── Top contributing factors for a state ──────────────────────────────────────
def top_causes(df: pd.DataFrame, state: str, n: int = 3) -> list[tuple[str, int, float]]:
    """
    Return top-n causes for a state as list of (cause, count, pct_of_state_total).
    """
    if "cause" not in df.columns:
        return []
    state_df = df[df["state"] == state]
    counts = state_df["cause"].value_counts().head(n)
    total = len(state_df)
    return [(cause, int(cnt), round(cnt / total * 100, 1)) for cause, cnt in counts.items()]


# ── State stats summary ───────────────────────────────────────────────────────
def state_summary_stats(df: pd.DataFrame, state: str) -> dict:
    """
    Return a dict of key stats for the Safety Brief.
    """
    state_df = df[df["state"] == state]
    stats: dict = {"state": state, "total_records": len(state_df)}

    if "accident_severity" in df.columns:
        stats["total_fatal"] = int((state_df["accident_severity"] == "fatal").sum())
        stats["total_major"] = int((state_df["accident_severity"] == "major").sum())
        stats["total_minor"] = int((state_df["accident_severity"] == "minor").sum())

    if "casualties" in df.columns:
        stats["total_casualties"] = int(state_df["casualties"].sum())

    if "year" in df.columns:
        stats["year_range"] = f"{int(state_df['year'].min())}–{int(state_df['year'].max())}"
    elif "date" in df.columns:
        stats["year_range"] = "See date column"
    else:
        stats["year_range"] = "N/A"

    return stats


# ── Safety Brief Markdown generator ──────────────────────────────────────────
def generate_safety_brief(
    df: pd.DataFrame,
    state: str,
    recs: list[Recommendation],
) -> str:
    """
    Generate a Markdown-formatted safety brief for download.
    Reuses already-computed data — does NOT re-run analysis.
    """
    stats = state_summary_stats(df, state)
    causes = top_causes(df, state, n=3)
    now = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"# SafeRoute AI — State Safety Brief",
        f"**State:** {state}",
        f"**Generated:** {now}",
        f"**Data Period:** {stats.get('year_range', 'N/A')}",
        "",
        "---",
        "",
        "## Accident & Casualty Overview",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total accident records | {stats['total_records']:,} |",
    ]

    if "total_fatal" in stats:
        lines += [
            f"| Fatal accidents | {stats['total_fatal']:,} |",
            f"| Major injury accidents | {stats['total_major']:,} |",
            f"| Minor injury accidents | {stats['total_minor']:,} |",
        ]
    if "total_casualties" in stats:
        lines.append(f"| Total casualties | {stats['total_casualties']:,} |")

    lines += [
        "",
        "## Top 3 Contributing Factors",
    ]
    if causes:
        lines.append("| Rank | Cause | Count | State Share |")
        lines.append("|------|-------|-------|-------------|")
        for i, (cause, cnt, pct) in enumerate(causes, 1):
            lines.append(f"| {i} | {cause.replace('_', ' ').title()} | {cnt:,} | {pct:.1f}% |")
    else:
        lines.append("_Cause column not available._")

    lines += [
        "",
        "## Policy Recommendations",
        "_These recommendations flag areas where further investigation is warranted. "
        "They are generated by transparent rule-based thresholds — not a guarantee "
        "that any intervention will prevent crashes._",
        "",
    ]
    for i, rec in enumerate(recs, 1):
        lines += [
            f"### {i}. {rec.title}",
            f"**Category:** {rec.category}",
            f"**Trigger:** {rec.trigger_value}",
            f"",
            rec.detail,
            "",
        ]

    lines += [
        "---",
        "",
        "## Source & Limitations",
        (
            "**Data source:** Synthetic dataset modelled on MoRTH / NCRB India road-safety "
            f"statistics, {stats.get('year_range', 'N/A')}."
        ),
        "",
        (
            "**Limitations:** State/district-level yearly aggregates; not per-crash GPS data. "
            "Metrics are accident frequency counts — no traffic-volume denominator is available, "
            "so comparisons reflect counts, not exposure-adjusted rates. "
            "All findings are observational; correlation ≠ causation."
        ),
        "",
        "_Generated by SafeRoute AI · IBM SkillsBuild Hackathon 2025_",
    ]

    return "\n".join(lines)
