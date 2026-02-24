# Proactive Affective Agent

Proactive Affective Agent for just-in-time adaptive interventions (JITAI) in cancer survivorship. Uses passive mobile sensing data and diary text to predict emotional states and intervention receptivity.

## Key Idea

Unlike reactive approaches (e.g., CALLM) that require users to write diary entries, this system uses **passive sensing data** (accelerometer, GPS, sleep, app usage, etc.) to proactively predict:
1. **Emotional state** (PANAS positive/negative affect, ER desire, binary states)
2. **Receptivity** to intervention (availability flag)

Each EMA entry becomes an opportunity for an AI agent to autonomously investigate behavioral signals and infer affective states — even when the participant writes nothing.

---

## Architecture: Six Agent Versions + ML Baselines

| Version | Diary | Sensing | RAG | Style | Novelty |
|---------|-------|---------|-----|-------|---------|
| **CALLM** | ✅ | ❌ | TF-IDF diary only | Reactive | CHI 2025 baseline |
| **V1** | ❌ | ✅ | memory doc only | Structured | Fixed 5-step pipeline |
| **V2** | ❌ | ✅ | memory doc only | Autonomous | LLM decides steps |
| **V3** | ✅ | ✅ | diary+sensing RAG | Structured | Multimodal RAG |
| **V4** | ✅ | ✅ | diary+sensing RAG | Autonomous | Full autonomy |
| **V5** | ✅ | ✅ | tool-use queries | **Agentic** | Detective-style investigation |
| **ML** | ❌ | ✅ | N/A | RF/XGBoost | Traditional baseline |

### V5: Agentic Sensing Investigation (Key Contribution)

V5 is the primary novel contribution. Instead of pre-computed feature summaries, the agent uses **Anthropic SDK tool use** to query a Parquet-backed sensing database autonomously:

```
EMA timestamp received
       ↓
LLM calls query_sensing(modality="gps", hours_before_ema=4)
LLM calls compare_to_baseline(metric="screen_on_min", ...)
LLM calls find_similar_days(top_k=3)
LLM calls get_ema_history(n_days=7)
       ↓
LLM reasons: "Less movement than usual + high screen time +
              3 similar days all had negative affect..."
       ↓
Structured prediction output
```

The agent builds up evidence like a behavioral data scientist detective — it decides **what to look at, not just what to predict**.

---

## Data: BUCS Cancer Survivorship Study

**418 participants** (297 iOS, 121 Android), ~5-week study.
- **EMA**: 3×/day (8–10am, 1–3pm, 7–9pm), ~15,000+ entries
- **Passive sensing**: 8 modalities at hourly resolution
- **Labels**: PANAS Pos/Neg, ER desire, 12 binary affect states, availability

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
    run_phase1.sh                      → all light modalities (30 min)
    process_accel.py                   → actigraphy (run overnight)
    process_gps.py                     → mobility features (run overnight)

data/processed/hourly/                 (output, ~2 GB)
    participant_platform.parquet       (418 participants, iOS/Android flag)
    home_locations.parquet             (362/413 home locations, median radius 25m)
    motion/ screen/ keyinput/ mus/ light/
    accel/ gps/                        (heavy, run separately)
```

**Missing data taxonomy** (4 classes, handled explicitly):
- `OBSERVED`: data present and valid
- `STRUCTURAL_MISSING`: platform doesn't support this sensor (e.g., no app-level data on iOS)
- `DEVICE_MISSING`: phone off / not carried (coverage < 5%)
- `PARTICIPANT_MISSING`: phone on, sensor active, but no activity

---

## Pilot Results (existing V1–V4)

| Metric | CALLM | V1 | V2 | V3 | V4 | ML |
|--------|-------|----|----|----|----|-----|
| Mean MAE ↓ | **~1.16** | ~high | 7.06 | *pending* | *pending* | *pending* |
| Mean BA ↑ | **~0.63** | ~0.52 | 0.52 | *pending* | *pending* | *pending* |
| Mean F1 ↑ | **~0.44** | ~low | 0.19 | *pending* | *pending* | *pending* |

CALLM (diary+RAG) dominates sensing-only V1/V2. V3/V4/V5/ML evaluation pending on BUCS processed data.

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
│   ├── personal_agent.py      # PersonalAgent dispatcher (routes to V1–V5)
│   ├── structured.py          # V1: sensing → structured 5-step pipeline
│   ├── autonomous.py          # V2: sensing → LLM-autonomous reasoning
│   ├── structured_full.py     # V3: diary + sensing + RAG → structured
│   ├── autonomous_full.py     # V4: diary + sensing + RAG → autonomous
│   └── agentic_sensing.py     # V5: Anthropic tool-use detective loop
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
├── run_pilot.py               # LLM experiment runner (V1–V4)
├── run_agentic_pilot.py       # V5 evaluation runner (tool-use loop)
├── run_ml_baselines.py        # ML baselines
└── run_parallel.sh            # Parallel 5-version experiment launcher

docs/
├── design.md                  # System design document
└── next-steps-waiting-for-data.md
```

---

## Running Experiments

### V5 Agentic Agent

```bash
# Dry run — test tool-use loop without LLM calls
python scripts/run_agentic_pilot.py --users 71 --dry-run

# Single user evaluation
python scripts/run_agentic_pilot.py --users 71 --model claude-opus-4-6

# Multiple users
python scripts/run_agentic_pilot.py --users 71,164,119 --model claude-sonnet-4-6
```

### V1–V4 LLM Versions

```bash
python scripts/run_pilot.py --version all --users 71,164,119,458,310 --dry-run
python scripts/run_pilot.py --version v3 --users 71,164,119,458,310
bash scripts/run_parallel.sh   # all 5 versions × 5 users in parallel
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
