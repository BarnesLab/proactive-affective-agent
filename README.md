# Proactive Affective Agent

Proactive Affective Agent for just-in-time adaptive interventions (JITAI) in cancer survivorship. Uses passive mobile sensing data and diary text to predict emotional states and intervention receptivity.

## Key Idea

Unlike reactive approaches (e.g., CALLM) that require users to write diary entries, this system uses **passive sensing data** (accelerometer, GPS, sleep, app usage, etc.) to proactively predict:
1. **Emotional state** (valence, arousal, stress, loneliness)
2. **Receptivity** to intervention (Desire ∧ Availability)

Each participant gets a **PersonalAgent** that learns and adapts over time through self-evaluation against EMA ground truth.

## Five Agent Versions + ML Baselines

| Version | Diary | Sensing | RAG | Pipeline Style | LLM Calls |
|---------|-------|---------|-----|----------------|-----------|
| **CALLM** | ✅ | ❌ | TF-IDF diary only (CHI baseline) | Reactive | 1 |
| **V1** | ❌ | ✅ | memory doc only | Structured | 1 |
| **V2** | ❌ | ✅ | memory doc only | Autonomous | 1 |
| **V3** | ✅ | ✅ | diary search → diary+sensing | Structured | 1 |
| **V4** | ✅ | ✅ | diary search → diary+sensing | Autonomous | 1 |
| **ML** | ❌ | ✅ | N/A | RF / XGBoost / LogReg | 0 |

- **CALLM**: CHI paper baseline — diary text + TF-IDF RAG
- **V1 vs V2**: Structured (fixed 5-step) vs Autonomous (LLM decides what matters), sensing only
- **V3 vs V4**: Same distinction, but with diary + sensing + enhanced multimodal RAG
- **ML baselines**: Traditional models on sensor feature vectors (no LLM calls needed)

## Phase 1 Pilot Results (5 users, 427 EMA entries)

| Metric | CALLM | V1 | V2 | V3 | V4 | ML |
|--------|-------|----|----|----|----|-----|
| Mean MAE ↓ | **~1.16** | ~high | 7.06 | *pending* | *pending* | *pending* |
| Mean BA ↑ | **~0.63** | ~0.52 | 0.52 | *pending* | *pending* | *pending* |
| Mean F1 ↑ | **~0.44** | ~low | 0.19 | *pending* | *pending* | *pending* |
| PT BA ↑ | **~0.72** | — | 0.54 | *pending* | *pending* | — |

CALLM (diary+RAG) dominates V1/V2 (sensing only). V3/V4 and ML baselines are ready to run — waiting for raw hourly sensing data from collaborator.

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy .env.example to .env and fill in API keys
cp .env.example .env
```

## Project Structure

```
src/
├── simulation/    # Data replay engine (PilotSimulator)
├── agent/         # PersonalAgent + 4 workflow classes
│   ├── structured.py       # V1: sensing → structured pipeline
│   ├── autonomous.py       # V2: sensing → autonomous reasoning
│   ├── structured_full.py  # V3: diary + sensing + RAG → structured
│   └── autonomous_full.py  # V4: diary + sensing + RAG → autonomous
├── think/         # LLM client (Claude CLI), prompts, response parser
├── remember/      # TF-IDF retriever + MultiModalRetriever (diary+sensing RAG)
├── baselines/     # ML baselines (RF, XGBoost, LogReg, Ridge)
├── data/          # Data loading, preprocessing, hourly features (placeholder)
├── evaluation/    # Metrics, reporting, unified cross-method comparison
└── utils/         # Config & constants

scripts/
├── run_pilot.py           # Main LLM experiment runner (5 versions)
├── run_parallel.sh        # Launch 25 parallel processes (5 versions × 5 users)
├── run_ml_baselines.py    # ML baselines (no LLM calls)
├── generate_dashboard.py  # HTML dashboard with comparison charts
├── select_pilot_users.py  # Select users by data coverage
└── sync_overleaf.py       # Overleaf ↔ local sync

docs/
├── design.md                      # System design document
└── next-steps-waiting-for-data.md # Action plan for when raw data arrives

draft/                     # Paper draft (synced from Overleaf)
```

## Data

Raw data is from the BUCS cancer survivorship study (399 users, 15,984 EMA entries). Includes:
- 8 passive sensing streams (accelerometer, GPS, sleep, screen, app usage, motion, key input)
- Daily & weekly EMA surveys (emotional states, intervention receptivity)
- Baseline trait questionnaires (346 features)
- 756 pre-generated memory documents

Data files are gitignored due to size. See `data/` for expected structure.

## Running Experiments

### LLM Versions (CALLM / V1 / V2 / V3 / V4)

```bash
# Dry run (test pipeline, no LLM calls)
python scripts/run_pilot.py --version all --users 71,164,119,458,310 --dry-run

# Run single version
python scripts/run_pilot.py --version v3 --users 71,164,119,458,310
python scripts/run_pilot.py --version v4 --users 71,164,119,458,310

# Run all 5 versions × 5 users in parallel (25 processes)
bash scripts/run_parallel.sh
```

### ML Baselines (no LLM calls)

```bash
# Run all models with daily features
python scripts/run_ml_baselines.py

# Specific models only
python scripts/run_ml_baselines.py --models rf,xgboost
```

### Evaluation

Two evaluation approaches:
1. **Regression**: MAE on continuous targets (PANAS_Pos, PANAS_Neg, ER_desire)
2. **CHI paper personal threshold**: Per-user mean ± SD on predicted values → binary classification → Balanced Accuracy & F1

## Paper Draft

The paper draft is maintained on Overleaf:
**[Overleaf Project](https://www.overleaf.com/project/6999d011b24a9f1d4e6e53e8)**

> Need access? Contact Zhiyuan for permissions.
