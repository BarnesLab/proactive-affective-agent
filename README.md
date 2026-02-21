# Proactive Affective Agent

Proactive Affective Agent for just-in-time adaptive interventions (JITAI) in cancer survivorship. Uses passive mobile sensing data to predict emotional states and intervention receptivity *before* user self-report.

## Key Idea

Unlike reactive approaches (e.g., CALLM) that require users to write diary entries, this system uses **passive sensing data** (accelerometer, GPS, sleep, app usage, etc.) to proactively predict:
1. **Emotional state** (valence, arousal, stress, loneliness)
2. **Receptivity** to intervention (Desire ∧ Availability)

Each participant gets a **PersonalAgent** that learns and adapts over time through self-evaluation against EMA ground truth.

## Two Agent Versions

- **V1 (Structured)**: Fixed pipeline — Sense → Retrieve Memory → Reason → Decide
- **V2 (Autonomous)**: LLM-driven agent with tools — decides what data to examine and how to act

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy .env.example to .env and fill in API keys
cp .env.example .env

# Run single-user simulation
python scripts/run_single_user.py --user_id <ID>

# Run full simulation
python scripts/run_simulation.py
```

## Project Structure

```
src/
├── simulation/    # Data replay engine
├── agent/         # PersonalAgent (V1 structured + V2 autonomous)
├── sense/         # Sensing data processing & feature extraction
├── think/         # LLM reasoning & prompt management
├── remember/      # Per-user memory & RAG retrieval
├── act/           # Decision-making & intervention delivery
├── learn/         # Self-evaluation & threshold adaptation
├── data/          # Data loading & preprocessing
├── evaluation/    # Metrics & reporting
└── utils/         # Config & constants
```

## Data

Raw data is from the BUCS cancer survivorship study. Includes:
- 8 passive sensing streams (accelerometer, GPS, sleep, screen, app usage, etc.)
- Daily & weekly EMA surveys (emotional states, intervention receptivity)
- Baseline trait questionnaires

Data files are gitignored due to size. See `data/` for expected structure.

## Paper Draft

The paper draft is maintained on Overleaf:
**[Overleaf Project](https://www.overleaf.com/project/6999d011b24a9f1d4e6e53e8)**

> Need access? Contact Zhiyuan for permissions.

The draft is also synced to `draft/` in this repository via the Overleaf Git bridge.

## Pilot Study

Compare **CALLM baseline** (CHI paper, diary text) vs **V1 Structured** (sensing) vs **V2 Autonomous** (sensing) on 5 users.

```bash
# Select best pilot users by data coverage
python scripts/select_pilot_users.py --group 1

# Dry run (test pipeline, no LLM calls)
python scripts/run_pilot.py --version all --dry-run

# Full pilot run (~5.5 hours)
python scripts/run_pilot.py --version all --group 1 --max-ema 30
```
