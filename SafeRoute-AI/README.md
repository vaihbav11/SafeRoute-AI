# SafeRoute AI 🛣️
**Road Safety Accident-Frequency Analysis · UN SDG 3.6 & 11.2**

---

## Project Overview

SafeRoute AI is a lightweight, fully open-source road-safety analytics tool that
surfaces accident-frequency patterns across Indian states from per-crash records.
It is aligned with two UN Sustainable Development Goals:

| SDG | Target |
|-----|--------|
| **3.6** | By 2030, halve the number of global deaths and injuries from road traffic accidents |
| **11.2** | By 2030, provide access to safe, affordable, accessible and sustainable transport systems |

---

## Assumptions (made in lieu of clarifying questions)

1. **Single CSV, per-crash records** — The workspace contained one file
   (`indian_roads_dataset.csv`, 20,000 rows × 24 columns) rather than multiple
   aggregated state/year CSV files described in the prompt.  All analysis is done
   on this per-crash dataset; state-year aggregations are computed on the fly.

2. **Latitude/longitude present** — The dataset does include `latitude` and
   `longitude` columns.  However, as directed, no GeoPandas/Folium map was built
   (fragile on Streamlit Cloud; out of MVP scope).

3. **`festival` column sparsity** — ~99.4 % null; filled with `"none"` as a
   categorical value rather than dropped, so it can still be used as a feature.

4. **Classification target** — `accident_severity` (`minor / major / fatal`) is
   mapped to a three-tier `severity_tier` label (`Low / Medium / High`) used as
   the Random Forest target.  This is defensible: the severity label is a direct
   outcome of the crash, not a proxy.

5. **`risk_score` in source data** — The CSV already contains a `risk_score`
   column (range 0–1).  It is used as a feature in the classifier and as one
   component of the composite state-ranking formula, but its derivation is
   unknown; we do not claim it as our own output.

6. **No multi-file merge needed** — Only one CSV was present; no cross-file
   state-name reconciliation was required.  The normalisation map is still in
   place for future data additions.

---

## State-Name Normalisation

The following alias mappings are applied in `src/data_loader.py`:

| Raw spelling | Normalised to |
|---|---|
| `Orissa` | Odisha |
| `Pondicherry` / `Pondichery` | Puducherry |
| `Uttaranchal` | Uttarakhand |
| `Jammu and Kashmir` | Jammu & Kashmir |
| `NCT of Delhi` / `Delhi (NCT)` | Delhi |
| `Andaman and Nicobar Islands` | Andaman & Nicobar Islands |
| `Dadra & Nagar Haveli` variants | Dadra & Nagar Haveli |
| `Daman & Diu` variants | Daman & Diu |

In this dataset, state names were already clean; none of the above remappings
were actually triggered.  They remain active for future CSV additions.

---

## Project Structure

```
SafeRoute-AI/
├── app.py                  # Streamlit UI (4 tabs, ~330 lines)
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py      # Load, normalise, clean, validate
│   ├── analysis.py         # EDA — 8 chart functions
│   └── modeling.py         # RF classifier + composite risk ranking
├── data/
│   └── raw/
│       └── indian_roads_dataset.csv
└── models/
    └── model.pkl           # Auto-generated on first run
```

---

## Quick Start (Windows)

```powershell
# 1. Clone / navigate into the project
cd SafeRoute-AI

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "SafeRoute AI MVP"
   git remote add origin https://github.com/<your-username>/saferoute-ai.git
   git push -u origin main
   ```

2. **Connect on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your repo → branch `main` → **Main file path**: `SafeRoute-AI/app.py`
     *(or move `app.py` to the repo root and set path to `app.py`)*
   - Click **Deploy**

3. **No secrets or env vars needed** — the app uses only relative paths and
   reads from `data/raw/` at runtime.

> **Note:** `models/model.pkl` is generated automatically on first run.
> Commit it to the repo OR add `models/` to `.gitignore` and let Streamlit
> Cloud regenerate it on each cold start (adds ~5 s to first load).

---

## Streamlit App — Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | SDG context, dataset metrics, raw data preview |
| **Data Quality** | Null audit (highlighted by severity), dtype table, numeric summary |
| **EDA** | Monthly trend, yearly by state, state comparison bar, cause breakdown, weather pie, severity × road type, risk score histogram, hour × day heatmap |
| **Model & Risk Ranking** | RF accuracy / precision / recall / F1, confusion matrix, feature importance, composite state risk-ranking table + bar chart |

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Target | `severity_tier`: Low / Medium / High |
| Features | 15 (see `src/modeling.py → FEATURE_COLS`) |
| Train/test split | 80 / 20, stratified |
| Class weighting | `balanced` (handles class imbalance) |
| Saved to | `models/model.pkl` (joblib) |

**Composite Risk Ranking formula** (transparent, no black-box):

```
composite_score = 0.35 × norm(avg_risk_score)
                + 0.30 × norm(fatal_rate)
                + 0.20 × norm(avg_casualties_per_accident)
                + 0.15 × norm(avg_vehicles_involved)
```

All four components are normalised to [0, 1] within the filtered dataset.

---

## Limitations

1. **No traffic-volume / exposure data** — metrics are raw accident frequency
   counts.  A state with more registered vehicles will naturally rank higher.
   Do not interpret rankings as true accident *rates*.

2. **7 states only** — the source CSV covers only Delhi, Karnataka, Maharashtra,
   Punjab, Tamil Nadu, Telangana, and West Bengal.  National coverage is absent.

3. **Synthetic-looking data** — the dataset appears algorithmically generated
   (perfectly round distributions, 19,885/20,000 null `festival` values, date
   range extending to April 2025).  Treat outputs as illustrative rather than
   policy-grade.

4. **No geographic map** — lat/long columns exist but a choropleth / pin map
   was intentionally excluded (fragile on Streamlit Cloud, out of MVP scope).

5. **No causal inference** — all findings are observational correlations.

6. **`risk_score` provenance unknown** — the column is present in the source
   CSV but its derivation is not documented.  It is used as a feature, not
   as a validated output.

---

## Smoke Tests

```powershell
# Run from inside SafeRoute-AI/
python src/data_loader.py    # ✓ load + validate
python src/analysis.py       # ✓ 8 chart functions
python src/modeling.py       # ✓ train + rank
```

---

## What Was Cut

GeoPandas / Folium choropleth map — excluded because (a) state-level data does
not benefit from pin maps, (b) both libraries are fragile on Streamlit Cloud
deploys, and (c) they were explicitly out of scope in the brief.

---

*Made with ❤️ for UN SDG 3.6 & 11.2*
