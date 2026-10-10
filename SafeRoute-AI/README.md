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

## Why This Matters

India recorded approximately **1.73 lakh (173,000) road deaths in 2023** (NCRB Road
Accidents in India report), equivalent to **14.6 deaths per 100,000 population** — more
than six times the rate in Denmark (2.3 per 100,000) and well above the global average.
Road traffic crashes are now the leading cause of death for Indians aged 15–29.

SafeRoute AI exists to make the pattern-finding work — identifying which states, causes,
road types, and conditions are overrepresented — accessible without a data-science
background, so that policy conversations can start from evidence rather than anecdote.

---

## Limitations & Methodology

| Limitation | Detail |
|---|---|
| **Frequency, not rate** | All metrics are accident *frequency counts*. No traffic-volume or vehicle-km denominator is available. States with more vehicles or better crash reporting appear worse. Do not interpret rankings as true risk rates. |
| **State/district-level aggregates** | The dataset is per-crash records aggregated to state level for analysis. There is no per-road-segment GPS data or route-level granularity. |
| **Correlation ≠ causation** | All findings (recommendations included) are observational. A high cause share in a state means that cause is disproportionately recorded there — it does not prove a causal mechanism or guarantee that an intervention will reduce crashes. |
| **7 states only** | The current dataset covers Delhi, Karnataka, Maharashtra, Punjab, Tamil Nadu, Telangana, and West Bengal. National-level conclusions should be drawn cautiously. |
| **Synthetic data** | The source CSV appears algorithmically generated (round distributions, date range to 2025). Treat outputs as illustrative of the methodology, not as policy-grade statistics. |
| **`risk_score` provenance** | The column exists in the source CSV; its derivation is undocumented. It is used as a model feature, not a validated output. |

---

## Future Work *(aspirational — not built)*

The following capabilities are noted for future development and are **not present in
this release**:

- **Route-level analysis** — per-road-segment crash data (e.g. from OpenStreetMap +
  MoRTH crash databases) would enable true route-safety scoring between cities.
- **Real-time traffic & weather integration** — connecting to live APIs (NOAA,
  Google Maps, HERE) to adjust risk scores dynamically.
- **Multilingual UI** — Hindi and regional-language interfaces to reach traffic
  enforcement officers and local government decision-makers directly.
- **Exposure-adjusted rates** — linking to VAHAN vehicle registration data to
  compute deaths per 10,000 registered vehicles per state.

---

## Project Structure

```
SafeRoute-AI/
├── app.py                  # Streamlit UI (7 tabs)
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py      # Load, normalise, clean, validate
│   ├── analysis.py         # EDA — 8 chart functions
│   ├── modeling.py         # RF classifier + composite risk ranking
│   ├── risk_map.py         # Plotly choropleth map (Tab 5)
│   ├── recommendations.py  # Rule-based policy panel + brief (Tab 6)
│   └── chat_assistant.py   # Groq-powered grounded chat (Tab 7)
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

## Quick Start — with Groq Chat

```powershell
# Set your Groq API key (get a free key at https://console.groq.com)
$env:GROQ_API_KEY = "gsk_your_key_here"

# Run the app
streamlit run app.py
```

Without the key the chat tab shows a setup message; all other tabs work normally.

---

## Deploying to Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "SafeRoute AI v2 — map + recommendations + chat"
   git remote add origin https://github.com/<your-username>/saferoute-ai.git
   git push -u origin main
   ```

2. **Connect on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click **New app**
   - Select your repo → branch `main` → **Main file path**: `SafeRoute-AI/app.py`
     *(or move `app.py` to the repo root and set path to `app.py`)*
   - Click **Deploy**

3. **Add the Groq secret (for chat tab)**
   - In your deployed app → **Settings → Secrets**
   - Add:
     ```toml
     GROQ_API_KEY = "gsk_your_key_here"
     ```
   - Save and reboot the app.

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
| **Risk Map** | Plotly choropleth — India states coloured by accident-frequency tier (same tiers as Model tab). Caption clearly states no exposure adjustment. |
| **Recommendations** | Rule-based policy panel for a user-selected state. 2–4 if/then recommendations with triggering values. Downloadable Markdown safety brief. |
| **Ask SafeRoute** | Groq LLM chat grounded in a computed data summary — no RAG, no invented stats. Requires `GROQ_API_KEY`. |

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

## Groq API Key Setup

| Environment | How to set |
|---|---|
| **Local** | `$env:GROQ_API_KEY = "gsk_..."` (PowerShell) or `export GROQ_API_KEY=gsk_...` (bash) before `streamlit run app.py` |
| **Streamlit Cloud** | App Settings → Secrets → add `GROQ_API_KEY = "gsk_..."` |

Secret key name: **`GROQ_API_KEY`** (exact string, no variation).
Model used: `llama-3.3-70b-versatile` (Groq-hosted Llama 3.3).
Free tier at [console.groq.com](https://console.groq.com) is sufficient for demo use.

---

## Smoke Tests

```powershell
# Run from inside SafeRoute-AI/
python src/data_loader.py    # ✓ load + validate
python src/analysis.py       # ✓ 8 chart functions
python src/modeling.py       # ✓ train + rank
python -c "from src.risk_map import build_map; print('risk_map OK')"
python -c "from src.recommendations import get_recommendations; print('recommendations OK')"
python -c "from src.chat_assistant import build_data_summary; print('chat_assistant OK')"
```

---

## What Was Cut / Scoped

- **Two-wheeler / pedestrian breakdowns** — the current dataset has no `vehicle_type` or
  `road_user_category` column; these recommendation rules were scoped to columns that
  actually exist (`cause`, `weather`, `road_type`, `accident_severity`).
- **PDF export** — the brief uses `st.download_button` with Markdown (`.md`) format per
  the spec; no new PDF library was added.

---

*Made with ❤️ for UN SDG 3.6 & 11.2*
