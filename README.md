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
Adding passive sensing data to diary entries improves prediction over diary alone (CALLM). Behavioral signals (mobility, screen time, activity) carry complementary information beyond what participants self-report. *V4-structured/agentic vs. CALLM.*

**H2 — LLM-empowered agents outperform ML baselines on sensing data.**
Large language models bring contextual, cross-modal reasoning that tabular ML (RF, XGBoost) cannot — especially for interpreting sparse, heterogeneous sensor streams. *V2-structured/agentic vs. RF/XGBoost/LogReg.*

**H3 — Passive sensing alone enables meaningful prediction.**
A sensing-only agent can predict emotional states and intervention receptivity without requiring any diary entry, making JITAI deployment feasible for participants who consistently skip self-report. *V2-structured/agentic vs. CALLM (sensing-only condition).*

**H4 — Autonomous agentic investigation outperforms structured pipelines.**
An agent that autonomously queries raw sensor data via tool-use (V2/V4-agentic) outperforms agents that receive pre-formatted feature summaries (V2/V4-structured), across:
- **(a) Predictive accuracy** — querying informative signals over irrelevant ones reduces noise
- **(b) Token efficiency** — selective investigation uses fewer tokens than unconditional feature dumps
- **(c) Interpretability** — explicit belief-update traces expose what evidence drove each prediction

---

## Architecture: 2×2 Design Space + Baselines

The key research question: does **agentic investigation** (autonomous tool-use queries) outperform **structured pipelines** (fixed pre-formatted summaries)? We test this across two data conditions: sensing-only and multimodal (diary + sensing).

|  | **Structured** (fixed pipeline) | **Agentic** (autonomous tool-use) |
|---|---|---|
| **Sensing-only** | V2-structured | **V2-agentic** |
| **Multimodal** (diary + sensing) | V4-structured | **V4-agentic** ← key contribution |

Plus: CALLM (diary+RAG reactive baseline, CHI 2025), ML baselines (RF/XGBoost/LogReg/Ridge).

### Agent Taxonomy

| Name | Diary | Sensing | Style | Novelty |
|------|-------|---------|-------|---------|
| **CALLM** | ✅ | ❌ | Reactive (diary required) | CHI 2025 baseline |
| **V2-structured** | ❌ | ✅ | Fixed 5-step pipeline | LLM-structured sensing only |
| **V2-agentic** | ❌ | ✅ | Autonomous tool-use | Detective loop, sensing only |
| **V4-structured** | ✅ | ✅ | Multimodal RAG | Diary + hourly sensing context |
| **V4-agentic** | ✅ | ✅ | Autonomous tool-use | **Key contribution** |
| **ML** | ❌ | ✅ | RF/XGBoost/LogReg | Traditional baseline |

### Agentic Investigation Loop (V2-agentic, V4-agentic)

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

The agent builds up evidence like a behavioral data scientist detective — it decides **what to look at, not just what to predict**. Token efficiency is a key metric: the hypothesis is that agentic agents query fewer, more informative signals than V2/V4-structured which receive all features unconditionally.

**Real-world deployment note**: The `get_receptivity_history` tool reflects what a deployed JITAI system would actually know — past intervention accept/reject events — not high-frequency EMA labels. For the research comparison we also return mood patterns from the study context.

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

## Pilot Results

| Metric | CALLM | V2-structured | V2-agentic | V4-structured | V4-agentic | ML |
|--------|-------|---------------|------------|---------------|------------|----|
| Mean MAE ↓ | **~1.16** | ~high | *pending* | *pending* | *pending* | *pending* |
| Mean BA ↑ | **~0.63** | ~0.52 | *pending* | *pending* | *pending* | *pending* |
| Mean F1 ↑ | **~0.44** | ~0.19 | *pending* | *pending* | *pending* | *pending* |

CALLM (diary+RAG) dominates sensing-only approaches. Full BUCS evaluation pending (Phase 1 heavy modalities + experiments).

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
│   ├── personal_agent.py      # PersonalAgent dispatcher
│   ├── structured.py          # V2-structured: sensing → fixed 5-step pipeline
│   ├── autonomous.py          # V2-structured (autonomous variant)
│   ├── structured_full.py     # V4-structured: diary + sensing + RAG → structured
│   ├── autonomous_full.py     # V4-structured (autonomous variant)
│   └── agentic_sensing.py     # V2/V4-agentic: Anthropic tool-use detective loop
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
│   ├── build_participant_roster.py
│   ├── compute_home_locations.py
│   ├── process_motion.py      # iOS MotionActivity + Android ActivityTransition/MOTION
│   ├── process_screen_app.py  # iOS ScreenOnTime + Android APPUSAGE
│   ├── process_keyinput.py    # FleksyKeyInput → typed words, sentiment
│   ├── process_mus.py         # MUS → listening hours
│   ├── process_light.py       # LIGHT → ambient lux (Android only)
│   ├── process_accel.py       # Accelerometer → actigraphy (heavy)
│   ├── process_gps.py         # GPS → mobility features (heavy)
│   └── utils.py               # Shared timezone/epoch helpers
├── run_pilot.py               # LLM experiment runner (V2/V4-structured)
├── run_agentic_pilot.py       # Agentic evaluation runner (V2/V4-agentic, tool-use loop)
├── run_ml_baselines.py        # ML baselines
└── run_parallel.sh            # Parallel experiment launcher

docs/
├── design.md                  # System design document
├── PROGRESS.md                # Auto-updated progress tracker
└── advisor-sync-architecture.html  # Architecture overview for advisor meetings
```

---

## Running Experiments

### V2/V4-Agentic (tool-use loop)

```bash
# Dry run — test tool-use loop without LLM calls
python scripts/run_agentic_pilot.py --users 71 --dry-run

# Single user evaluation
python scripts/run_agentic_pilot.py --users 71 --model claude-opus-4-6

# Multiple users (diary-present EMA entries only, for CALLM comparison)
python scripts/run_agentic_pilot.py --users 71,164,119 --model claude-sonnet-4-6
```

### V2/V4-Structured LLM Versions

```bash
python scripts/run_pilot.py --version all --users 71,164,119,458,310 --dry-run
python scripts/run_pilot.py --version v4 --users 71,164,119,458,310
bash scripts/run_parallel.sh   # all versions in parallel
```

### ML Baselines

```bash
python scripts/run_ml_baselines.py
python scripts/run_ml_baselines.py --models rf,xgboost
```

---

## Data

Raw BUCS data is gitignored (~114 GB). Processed Parquet outputs (~2 GB) also gitignored. See `data/README.md` for expected directory structure.

Paper draft on Overleaf: `https://www.overleaf.com/project/6999d011b24a9f1d4e6e53e8`

Design doc (Google Docs): https://docs.google.com/document/d/1BJ8P81Zcy3fKQYyQXNr9wU_es1tjkUGCdEblBvemskQ/edit?usp=sharing

Progress & next steps: `docs/PROGRESS.md`
