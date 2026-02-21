# Proactive Affective Agent

Proactive Affective Agent for just-in-time adaptive interventions (JITAI) in cancer survivorship. Uses passive mobile sensing data to predict emotional states and intervention receptivity *before* user self-report.

## Key Idea

Unlike reactive approaches (e.g., CALLM) that require users to write diary entries, this system uses **passive sensing data** (accelerometer, GPS, sleep, app usage, etc.) to proactively predict:
1. **Emotional state** (valence, arousal, stress, loneliness)
2. **Receptivity** to intervention (Desire ∧ Availability)

Each participant gets a **PersonalAgent** that learns and adapts over time through self-evaluation against EMA ground truth.

## Three Agent Versions

| Version | Input | RAG | LLM Calls/Entry | Description |
|---------|-------|-----|-----------------|-------------|
| **CALLM** | diary text (`emotion_driver`) | TF-IDF top-20 similar cases | 1 | CHI paper baseline (reactive) |
| **V1 Structured** | passive sensing data | memory doc only | 1 | Fixed pipeline: Sense → Retrieve → Reason → Decide |
| **V2 Autonomous** | passive sensing data | memory doc + on-demand | 2-3 | ReAct-style: LLM decides what to examine |

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
├── agent/         # PersonalAgent (CALLM + V1 structured + V2 autonomous)
├── think/         # LLM client (Claude CLI), prompts, response parser
├── remember/      # TF-IDF retriever for RAG
├── data/          # Data loading & preprocessing
├── evaluation/    # Metrics (MAE, BA, F1, personal threshold) & reporting
└── utils/         # Config & constants

scripts/
├── run_pilot.py           # Main experiment runner
├── select_pilot_users.py  # Select users by data coverage
├── monitor_experiment.py  # Telegram progress notifications
└── sync_overleaf.py       # Overleaf ↔ local sync docs

draft/                     # Paper draft (synced from Overleaf)
```

## Data

Raw data is from the BUCS cancer survivorship study (399 users, 15,984 EMA entries). Includes:
- 8 passive sensing streams (accelerometer, GPS, sleep, screen, app usage, motion, key input)
- Daily & weekly EMA surveys (emotional states, intervention receptivity)
- Baseline trait questionnaires (346 features)
- 756 pre-generated memory documents

Data files are gitignored due to size. See `data/` for expected structure.

## Paper Draft

The paper draft is maintained on Overleaf:
**[Overleaf Project](https://www.overleaf.com/project/6999d011b24a9f1d4e6e53e8)**

> Need access? Contact Zhiyuan for permissions.

The draft is also synced to `draft/` in this repository via the Overleaf Git bridge.

## Pilot Study

Compare **CALLM baseline** (CHI paper, diary text) vs **V1 Structured** (sensing) vs **V2 Autonomous** (sensing) on 5 users with all their EMA entries.

```bash
# Select best pilot users by data coverage
python scripts/select_pilot_users.py

# Dry run (test pipeline, no LLM calls)
python scripts/run_pilot.py --version all --users 71,164,119,458,310 --dry-run

# Run single version (for parallel execution)
python scripts/run_pilot.py --version callm --users 71,164,119,458,310
python scripts/run_pilot.py --version v1 --users 71,164,119,458,310
python scripts/run_pilot.py --version v2 --users 71,164,119,458,310

# Run all versions sequentially
python scripts/run_pilot.py --version all --users 71,164,119,458,310
```

### Evaluation

Two evaluation approaches:
1. **Regression**: MAE on continuous targets (PANAS_Pos, PANAS_Neg, ER_desire)
2. **CHI paper personal threshold**: Per-user mean ± SD on predicted values → binary classification → Balanced Accuracy & F1
