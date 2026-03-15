# Proactive Affective Agent

Proactive Affective Agent for just-in-time adaptive interventions (JITAI) in cancer survivorship. Uses passive mobile sensing data and diary text to predict emotional states and intervention receptivity.

## Key Idea

Unlike reactive approaches (e.g., CALLM) that require users to write diary entries, this system uses **passive sensing data** (accelerometer, GPS, sleep, app usage, etc.) to proactively predict:
1. **Emotional state** (PANAS positive/negative affect, ER desire, binary states)
2. **Receptivity** to intervention (availability flag)

Each EMA entry becomes an opportunity for an AI agent to autonomously investigate behavioral signals and infer affective states — even when the participant writes nothing.

---

## Research Hypotheses

**H1 — Sensing augments diary-based prediction.**
Adding passive sensing data to diary entries improves prediction over diary alone (CALLM). Behavioral signals (mobility, screen time, activity) carry complementary information beyond what participants self-report. *V3, V4 vs. CALLM.*

**H2 — LLM-empowered agents outperform ML baselines on sensing data.**
Large language models bring contextual, cross-modal reasoning that tabular ML (RF, XGBoost) cannot — especially for interpreting sparse, heterogeneous sensor streams. *V1, V2 vs. ML.*

**H3 — Passive sensing alone enables meaningful prediction.**
A sensing-only agent can predict emotional states and intervention receptivity without requiring any diary entry, making JITAI deployment feasible for participants who consistently skip self-report. *V1, V2 vs. CALLM (sensing-only condition).*

**H4 — Autonomous agentic investigation outperforms structured pipelines.**
An agent that autonomously queries raw sensor data via tool-use (V2, V4) outperforms agents that receive pre-formatted feature summaries (V1, V3), across:
- **(a) Predictive accuracy** — querying informative signals over irrelevant ones reduces noise
- **(b) Token efficiency** — selective investigation uses fewer tokens than unconditional feature dumps
- **(c) Interpretability** — explicit belief-update traces expose what evidence drove each prediction

---

## Architecture: 2×2 Design Space + Baselines

The key research question: does **agentic investigation** (autonomous tool-use queries) outperform **structured pipelines** (fixed pre-formatted summaries)? We test this across two data conditions: sensing-only and multimodal (diary + sensing).

|  | **Structured** (fixed pipeline, **most existing work**) | **Agentic** (autonomous tool-use) |
|---|---|---|
| **Sensing-only** | V1 | **V2** |
| **Multimodal** (diary + sensing) | V3 | **V4** ← key contribution |

Plus: CALLM (diary+RAG reactive baseline, CHI 2025), ML baselines (RF/XGBoost/LogReg), AR autocorrelation baseline.

### Model Taxonomy

| Name | Diary | Sensing | Style | Novelty |
|------|-------|---------|-------|---------|
| **ML** | ❌ | ✅ | RF/XGBoost/LogReg | Traditional baseline |
| **CALLM** | ✅ | ❌ | Reactive (diary required) | CHI 2025 baseline |
| **V1** | ❌ | ✅ | Fixed 5-step pipeline | LLM-structured sensing only |
| **V2** | ❌ | ✅ | Autonomous tool-use | Detective loop, sensing only |
| **V3** | ✅ | ✅ | Multimodal RAG | Diary + hourly sensing context |
| **V4** | ✅ | ✅ | Autonomous tool-use | **Key contribution** |
| **V5** | ❌ | ✅ (filtered) | Agentic + data quality | V2 + filtered sensing |
| **V6** | ✅ | ✅ (filtered) | Agentic + data quality | V4 + filtered sensing |

### Agentic Investigation Loop (V2, V4)

Instead of pre-computed feature summaries, the agent uses **Anthropic SDK tool use** to query a Parquet-backed sensing database autonomously:

```
EMA timestamp received
       ↓
LLM calls query_sensing(modality="gps", hours_before_ema=4)
LLM calls compare_to_baseline(feature="screen_on_min", current_value=45)
LLM calls find_similar_days(n=5)
LLM calls get_receptivity_history(n_days=7)
       ↓
LLM reasons: "Less movement than usual + high screen time +
              3 similar days all had negative affect..."
       ↓
Structured prediction output (JSON)
```

The agent builds up evidence like a behavioral data scientist detective — it decides **what to look at, not just what to predict**. Token efficiency is a key metric: the hypothesis is that agentic agents (V2, V4) query fewer, more informative signals than structured agents (V1, V3) which receive all features unconditionally.

**Real-world deployment note**: The `get_receptivity_history` tool reflects what a deployed JITAI system would actually know — past intervention accept/reject events — not high-frequency EMA labels. For the research comparison we also return mood patterns from the study context.

### Example: Agent Investigation Trace

**Scenario:** Participant 071, 2024-01-15, 2:00 PM EMA.
**Profile:** 62-year-old female, breast cancer stage II, 4 years post-treatment. PHQ-8 = 8 (mild depression), extraversion = 3.2/7, low social support.
**Receptivity log:** Past 7 days: 4 accepted, 2 rejected. Yesterday 2pm slot: rejected.

> *Agent:* "I need to figure out this person's emotional state around 2pm today. She has mild depression and is not very extraverted. Let me start with a big-picture look at today."

`→ CALL: get_daily_summary(date="2024-01-15")`
```
GPS: stayed home all day, 0 km travel. Screen: 3.4h total (high). Motion: 98% stationary.
Keyboard: 12 words typed (very low). Music: no listening.
```
> *Belief update:* She barely moved, barely typed, barely left home. Baseline for this person? I need to check.

> *Agent:* "The home confinement and inactivity stand out. Let me check if this is unusual for her — maybe she works from home and this is normal."

`→ CALL: compare_to_baseline(metric="gps_at_home_pct", lookback_days=14)`
```
Her 14-day baseline: typically home 52% of the time. Today: 100%.
Percentile: 97th. z-score: +2.8. This is extremely unusual for her.
```
> *Belief update:* Strong anomaly signal — much more confined than usual, not just "she works from home."

> *Agent:* "97th percentile home confinement is a red flag. Paired with very low keyboard use and no music — possible social withdrawal or low energy. Let me check if similar patterns predict negative affect."

`→ CALL: find_similar_days(top_k=3)`
```
Similar day 2024-01-08: (home all day, high screen, no music) → PANAS_Neg=5, ER_desire=6  ✗ rejected
Similar day 2023-12-22: (home, holiday)                       → PANAS_Neg=3, ER_desire=2  ✓ accepted
Similar day 2024-01-02: (home, low GPS)                       → PANAS_Neg=5, ER_desire=5  ✓ accepted
```
> *Belief update:* The 2024-01-08 match (no music, low typing, home all day — same pattern as today) showed the highest negative affect AND a rejected intervention. Concerning match.

> *Agent:* "The 2024-01-08 match is concerning. Let me check recent receptivity history — she rejected yesterday, which may signal fatigue with interventions right now."

`→ CALL: get_receptivity_history(n=7)`
```
Last 7 days: 4 accepted, 2 rejected.
Yesterday 2pm slot: rejected (low ER_desire).
Today is a Monday — she often has lower affect on Mondays (pattern in history).
```
> *Belief update:* Yesterday's rejection and the Monday pattern converge. Elevated NA likely, plus intervention fatigue. Enough evidence.

**Final prediction:** `PANAS_Neg = 5` (elevated, 69th personal percentile), `ER_desire = 5` (moderate), `Individual_level_NA_State = True`, availability = *yes but with caution* (receptivity fatigue — suggest lighter intervention or delay).

**Reasoning summary:** Extreme home confinement (97th pct), very low typing, behavioral match to a past high-NA day (2024-01-08), plus yesterday's rejection → elevated negative affect. Recommend lower-intensity contact.

---

## Data: BUCS Cancer Survivorship Study

**418 participants** (297 iOS, 121 Android), ~5-week study.
- **EMA**: 3×/day (8–10am, 1–3pm, 7–9pm), ~15,000+ entries
- **Passive sensing**: 8 modalities at hourly resolution
- **Labels**: PANAS Pos/Neg, ER desire, 12 binary affect states, availability
- **Evaluation subset**: EMA entries where diary was also written (for apple-to-apple CALLM comparison)

### BUCS Sensing Pipeline (Phase 0 → Phase 1)

```
data/bucs-data/                        (raw, ~114 GB)
    Accelerometer/                     (1 Hz, 105 GB)
    GPS/                               (variable rate)
    MotionActivity/ ActivityTransition/ MOTION/
    ScreenOnTime/ APPUSAGE/
    FleksyKeyInput/
    MUS/
    LIGHT/
    AndroidSleep/

scripts/offline/                       (offline batch processors)
    run_phase0.sh                      → participant roster + home locations
    run_phase1.sh                      → all light modalities (~30 min)
    process_accel.py                   → actigraphy (run overnight)
    process_gps.py                     → mobility features (run overnight)

data/processed/hourly/                 (output, ~2 GB Parquet)
    participant_platform.parquet       (418 participants, iOS/Android flag)
    home_locations.parquet             (359/413 home locations, median radius 25m)
    motion/ screen/ keyinput/ mus/ light/
    accel/ gps/                        (heavy, run separately)
```

**Note**: 114 GB raw → 2 GB Parquet = semantic feature aggregation, not lossy compression. Raw 1 Hz accelerometer → 4 hourly statistics (activity counts, mean/std magnitude, coverage %). Agent tools in `src/sense/query_tools.py` can query at hourly granularity; raw data remains accessible on disk.

**Missing data taxonomy** (4 classes, handled explicitly):
- `OBSERVED`: data present and valid
- `STRUCTURAL_MISSING`: platform doesn't support this sensor (e.g., no app-level data on iOS)
- `DEVICE_MISSING`: phone off / not carried (coverage < 5%)
- `PARTICIPANT_MISSING`: phone on, sensor active, but no activity

---

## Pilot Results (11 users, 956 entries/version)

### LLM Agent Versions

| Metric | CALLM | V1 | V2 | V3 | V4 | V5 | **V6** |
|--------|-------|----|----|----|----|----|----|
| Mean MAE ↓ | 3.661 | 5.502 | 4.881 | 3.947 | 4.353 | 4.878 | **4.220** |
| Mean BA ↑ | 0.624 | 0.523 | 0.601 | 0.638 | 0.673 | 0.602 | **0.675** |
| Mean F1 ↑ | 0.611 | 0.465 | 0.594 | 0.632 | 0.667 | 0.596 | **0.669** |

AR autocorrelation baseline: Mean BA = 0.658. V4 and V6 both exceed this ceiling.

### ML/DL Baselines (sensing-only, 5-fold CV)

| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| RF | 5.923 | 0.501 | 0.365 |
| XGBoost | 9.374 | 0.502 | 0.391 |
| MiniLM (text) | 3.898 | 0.629 | 0.588 |
| Combined (RF) | 3.935 | 0.620 | 0.568 |

### Key Findings
- **Multimodal agentic (V4/V6) beats AR baseline** — agentic tool-use on sensing+diary surpasses autocorrelation ceiling
- **Agentic > Structured** — V2 > V1, V4 > V3 across BA and F1
- **Multimodal > Sensing-only** — diary context provides substantial lift
- **Sensing-only ML near chance** — traditional ML on sensor features alone (BA ~0.50) far below LLM agents
- **LLM agents outperform text baselines** — V4/V6 (BA 0.67) > MiniLM (BA 0.63) despite MiniLM having diary access

---

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy .env.example to .env and fill in API keys
cp .env.example .env

# Run Phase 0 (fast, <5 min)
bash scripts/offline/run_phase0.sh

# Run Phase 1 light modalities (motion/screen/key/music/light, ~30 min)
bash scripts/offline/run_phase1.sh

# Run Phase 1 heavy modalities (accel + GPS, overnight)
python scripts/offline/process_accel.py
python scripts/offline/process_gps.py
```

---

## Project Structure

```
src/
├── agent/
│   ├── personal_agent.py         # PersonalAgent dispatcher (CALLM/V1-V6)
│   ├── structured.py             # V1: sensing-only, fixed pipeline
│   ├── structured_full.py        # V3: multimodal (diary+sensing), fixed pipeline
│   └── cc_agent.py               # V2/V4/V5/V6: agentic agent via claude --print + MCP
├── sense/
│   ├── query_tools.py         # SensingQueryEngine + SENSING_TOOLS (SDK format)
│   └── features.py            # HourlyFeatureLoader and feature extractors
├── data/
│   ├── hourly_features.py     # HourlySensingWindow dataclass + SensingContext
│   └── schema.py              # EMAResponse, SensingDay, PredictionOutput
├── think/                     # LLM client, prompts, response parser
├── remember/                  # TF-IDF + MultiModalRetriever (diary+sensing RAG)
├── baselines/                 # ML baselines (RF, XGBoost, LogReg, Ridge)
├── evaluation/                # Metrics, reporting, cross-method comparison
└── utils/                     # Config, constants, mappings

scripts/
├── offline/                   # Phase 0/1 batch processors (run once)
│   ├── run_phase0.sh          # Roster + home locations
│   ├── run_phase1.sh          # Light modality batch
│   ├── process_motion.py      # iOS MotionActivity + Android ActivityTransition/MOTION
│   ├── process_screen_app.py  # iOS ScreenOnTime + Android APPUSAGE
│   ├── process_keyinput.py    # FleksyKeyInput → typed words, sentiment
│   ├── process_mus.py         # MUS → listening hours
│   ├── process_light.py       # LIGHT → ambient lux (Android only)
│   ├── process_accel.py       # Accelerometer → actigraphy (heavy)
│   └── process_gps.py         # GPS → mobility features (heavy)
├── run_pilot.py               # LLM experiment runner (all versions)
├── queue_runner.py            # Parallel queue (5 workers, auto-resume)
├── evaluate_pilot.py          # Compute MAE/BA/F1 from checkpoints
├── results_dashboard.py       # Live results dashboard (auto-refresh, reads checkpoints)
├── generate_dashboard.py      # Static HTML dashboard generator
├── build_filtered_data.py     # Build filtered datasets for V5/V6
└── run_ml_baselines_new.py    # ML/DL/text baselines

docs/
├── design.md                  # System design document
└── advisor-sync-architecture.html  # Architecture overview for advisor meetings
```

---

## Running Experiments

### Queue Runner (recommended)

```bash
# Preview what needs running
python3 scripts/queue_runner.py --dry-run

# Run with 5 workers (default)
python3 scripts/queue_runner.py

# Run with fewer workers if rate-limited
python3 scripts/queue_runner.py --workers 3
```

### Individual Versions

```bash
# All versions for specific users
python3 scripts/run_pilot.py --version all --users 71,164,119 --model sonnet

# Single version
python3 scripts/run_pilot.py --version v3 --users 71,164,119 --model sonnet
```

### Evaluation & Dashboard

```bash
# Compute metrics from checkpoints
PYTHONPATH=. python3 scripts/evaluate_pilot.py

# Live results dashboard (auto-refreshes, reads checkpoints on each load)
python3 scripts/results_dashboard.py
# → http://localhost:8877
```

### ML Baselines

```bash
python3 scripts/run_ml_baselines_new.py
```

---

## Data

Raw BUCS data is gitignored (~114 GB). Processed Parquet outputs (~2 GB) also gitignored. See `data/README.md` for expected directory structure.

Paper draft on Overleaf: `https://www.overleaf.com/project/6999d011b24a9f1d4e6e53e8`

Design doc (Google Docs): https://docs.google.com/document/d/1BJ8P81Zcy3fKQYyQXNr9wU_es1tjkUGCdEblBvemskQ/edit?usp=sharing

Architecture overview (advisor sync HTML): https://barneslab.github.io/proactive-affective-agent/docs/advisor-sync-architecture.html

Progress & next steps: `PROGRESS.md`
