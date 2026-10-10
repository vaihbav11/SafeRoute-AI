"""
risk_map.py — SafeRoute AI
Plotly choropleth map of India coloured by accident-frequency tier.

GeoJSON source (fetched at runtime, cached):
  https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson

If the fetch fails (offline/firewall), the function returns (None, error_message) and
the caller shows a user-facing instruction instead of crashing.

State-name normalisation is applied before merging.
Any state in the accident data that fails to match a GeoJSON feature is logged to
standard output AND returned in the unmatched list.

Design assumptions:
- Color metric: 'risk_tier' from modeling.state_risk_ranking() (Low / Medium / High)
  — the same tier the Model tab already shows.  Labelled "accident frequency tier"
  in the UI because no traffic-volume denominator is available.
- Hover shows: state, total_accidents, fatal_rate (as %), risk_tier.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional

import pandas as pd

# Primary GeoJSON URL — India states (includes updated Telangana boundary)
_GEOJSON_URL = (
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson"
)

# Fallback URL
_GEOJSON_URL_ALT = (
    "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States"
    "/master/Indian_States"
)

# Property key inside the GeoJSON features that holds the state name
_NAME_PROP = "NAME_1"   # used in geohacker dataset; fallback tries "ST_NM"

# ── State-name normalisation for GeoJSON → accident-data matching ─────────────
# Maps GeoJSON display names → canonical names used in the accident DataFrame
_GEOJSON_TO_DATA: dict[str, str] = {
    "Andaman & Nicobar": "Andaman & Nicobar Islands",
    "Andaman and Nicobar": "Andaman & Nicobar Islands",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Bihar": "Bihar",
    "Chandigarh": "Chandigarh",
    "Chhattisgarh": "Chhattisgarh",
    "Dadra and Nagar Haveli": "Dadra & Nagar Haveli",
    "Daman and Diu": "Daman & Diu",
    "Delhi": "Delhi",
    "NCT of Delhi": "Delhi",
    "Goa": "Goa",
    "Gujarat": "Gujarat",
    "Haryana": "Haryana",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu & Kashmir": "Jammu & Kashmir",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Lakshadweep": "Lakshadweep",
    "Madhya Pradesh": "Madhya Pradesh",
    "Maharashtra": "Maharashtra",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Odisha": "Odisha",
    "Orissa": "Odisha",
    "Pondicherry": "Puducherry",
    "Puducherry": "Puducherry",
    "Punjab": "Punjab",
    "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim",
    "Tamil Nadu": "Tamil Nadu",
    "Telangana": "Telangana",
    "Tripura": "Tripura",
    "Uttar Pradesh": "Uttar Pradesh",
    "Uttarakhand": "Uttarakhand",
    "Uttaranchal": "Uttarakhand",
    "West Bengal": "West Bengal",
}


def _fetch_geojson() -> tuple[Optional[dict], Optional[str], str]:
    """
    Fetch GeoJSON from URL.
    Returns (geojson_dict, name_property_key, error_message).
    error_message is empty string on success.
    """
    for url, name_key in [(_GEOJSON_URL, _NAME_PROP), (_GEOJSON_URL_ALT, "ST_NM")]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SafeRoute-AI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            # Validate it looks like a FeatureCollection
            if "features" in data and len(data["features"]) > 0:
                # Try to detect the correct name property
                props = data["features"][0].get("properties", {})
                if name_key not in props:
                    # Try common alternatives
                    for alt in ("NAME_1", "ST_NM", "state", "State", "NAME"):
                        if alt in props:
                            name_key = alt
                            break
                return data, name_key, ""
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            continue

    return None, None, (
        "Could not fetch India states GeoJSON automatically.\n\n"
        "**Manual fix:** Download the file from:\n"
        "https://github.com/geohacker/india/blob/master/state/india_telengana.geojson\n\n"
        "Save it to: `SafeRoute-AI/data/india_states.geojson`\n\n"
        "The map will then load from disk automatically."
    )


def _load_geojson_disk() -> tuple[Optional[dict], Optional[str]]:
    """Try loading from disk fallback path."""
    import pathlib
    disk_path = pathlib.Path(__file__).parent.parent / "data" / "india_states.geojson"
    if disk_path.exists():
        with open(disk_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        props = data["features"][0].get("properties", {}) if data.get("features") else {}
        for name_key in ("NAME_1", "ST_NM", "state", "State", "NAME"):
            if name_key in props:
                return data, name_key
    return None, None


def build_map(ranking: pd.DataFrame) -> tuple:
    """
    Build a Plotly choropleth figure for India states coloured by risk_tier.

    Parameters
    ----------
    ranking : DataFrame from modeling.state_risk_ranking()
              Must have columns: state, total_accidents, fatal_rate, risk_tier, composite_score.

    Returns
    -------
    (fig, unmatched_states, error_message)
    fig              : plotly.graph_objects.Figure or None
    unmatched_states : list of state names in accident data that had no GeoJSON match
    error_message    : non-empty string if an error prevented map creation
    """
    import plotly.express as px

    if ranking is None or ranking.empty:
        return None, [], "No ranking data available."

    # ── Load GeoJSON ─────────────────────────────────────────────────────────
    geojson, name_key = _load_geojson_disk()
    if geojson is None:
        geojson, name_key, err = _fetch_geojson()
        if geojson is None:
            return None, [], err

    # ── Build state→normalised name lookup from GeoJSON ───────────────────────
    geojson_states: dict[str, str] = {}  # raw geojson name → normalised
    for feat in geojson["features"]:
        raw = feat["properties"].get(name_key, "")
        normalised = _GEOJSON_TO_DATA.get(raw, raw)  # use map or keep as-is
        feat["properties"]["_state_norm"] = normalised
        geojson_states[raw] = normalised

    # ── Normalise ranking state names ─────────────────────────────────────────
    ranking = ranking.copy()
    ranking["state_norm"] = ranking["state"]  # already normalised by data_loader

    # Find which data states have no GeoJSON match
    geojson_normalised_set = set(geojson_states.values())
    unmatched = [
        s for s in ranking["state_norm"].unique()
        if s not in geojson_normalised_set
    ]
    if unmatched:
        print(f"[risk_map] WARNING — states in accident data not matched in GeoJSON: {unmatched}")

    # Merge: only matched states will appear coloured; unmatched states still
    # appear in the DataFrame but get "No Data" tier
    merged = ranking.copy()

    # ── Prepare hover data ─────────────────────────────────────────────────────
    merged["fatal_rate_pct"] = (merged["fatal_rate"] * 100).round(1)
    merged["total_accidents_str"] = merged["total_accidents"].apply(lambda x: f"{int(x):,}")

    # Tier colour scale (categorical)
    tier_color_map = {
        "High": "#d62728",
        "Medium": "#ff7f0e",
        "Low": "#2ca02c",
        "No Data": "#cccccc",
    }
    merged["risk_tier"] = merged["risk_tier"].astype(str)

    # ── Build featureidkey ────────────────────────────────────────────────────
    # We need to match GeoJSON features to DataFrame rows via the normalised name
    # So we update GeoJSON to use _state_norm as the key property
    for feat in geojson["features"]:
        feat["properties"]["_state_norm"] = feat["properties"].get("_state_norm", "")

    fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="state_norm",
        featureidkey=f"properties._state_norm",
        color="risk_tier",
        color_discrete_map=tier_color_map,
        category_orders={"risk_tier": ["High", "Medium", "Low", "No Data"]},
        hover_name="state",
        hover_data={
            "total_accidents_str": True,
            "fatal_rate_pct": True,
            "risk_tier": True,
            "state_norm": False,
            "total_accidents": False,
            "fatal_rate": False,
        },
        labels={
            "total_accidents_str": "Total Accidents",
            "fatal_rate_pct": "Fatal Rate (%)",
            "risk_tier": "Accident Freq. Tier",
        },
        title="India — Accident Frequency Tier by State",
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="white",
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend_title_text="Accident Frequency Tier",
        height=520,
        paper_bgcolor="white",
        font_family="-apple-system, 'Segoe UI', system-ui, sans-serif",
    )

    return fig, unmatched, ""
