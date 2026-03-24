# PULSE: Agentic Sensing for Proactive Intervention in Cancer Survivorship

> **Paper:** Under review at IMWUT 2026
>
> **Results dashboard:** [Interactive results](docs/results-dashboard.html)

LLM agents that autonomously investigate passive mobile sensing data to predict emotional states and intervention receptivity in cancer survivors — without requiring any user input.

## The Problem

Existing emotion-aware systems (e.g., [CALLM, CHI 2025](https://dl.acm.org/doi/10.1145/3706598.3714882)) are **reactive**: they require users to write diary entries. But participants skip self-report precisely when they need help most — during high stress, low motivation, or withdrawal. We need systems that act **proactively**, using passive sensing to anticipate emotional states.

## The Approach

We design LLM agents that use **Anthropic SDK tool-use** to autonomously query a Parquet-backed sensing database (GPS, accelerometer, screen time, keyboard, music, light, sleep) and predict emotional states at each EMA prompt — like a behavioral data scientist detective.

The core research question: does **agentic investigation** (autonomous tool-use) outperform **structured pipelines** (fixed pre-formatted summaries)?

### 2x2 Design Space

|  | **Structured** (fixed pipeline) | **Agentic** (autonomous tool-use) |
|---|---|---|
| **Sensing-only** | V1 (BA: 0.521) | V2 (BA: 0.598) |
| **Multimodal** (diary + sensing) | V3 (BA: 0.607) | **V4 (BA: 0.666)** |

Plus: V5/V6 (agentic + data quality filtering), CALLM baseline (diary-only, BA: 0.626), ML baselines (RF/XGBoost/MiniLM).

**Best system: V6** (agentic + filtered multimodal) — **BA: 0.669, F1: 0.664**.

## Key Findings

1. **Agentic > Structured:** Agents that autonomously decide *what to investigate* outperform structured pipelines. V4 BA=0.666 vs V3 BA=0.607 (+9.7%). V2 BA=0.598 vs V1 BA=0.521 (+14.8%).

2. **LLM agents beat all ML baselines:** V6 (BA=0.669) outperforms MiniLM sentence embeddings (BA=0.629), combined RF (BA=0.620), and sensing-only ML (BA~0.50).

3. **Sensing augments diary:** Adding passive sensing to diary text improves over diary-alone CALLM by +6.9% BA.

4. **Proactive prediction is feasible:** Sensing-only agents (V2 BA=0.598, V5 BA=0.601) predict well above chance without any diary entry.

## Results

### LLM Agent Systems (18-user evaluation set, all Claude Sonnet)

| System | Input | Mean BA | Mean F1 |
|--------|-------|---------|---------|
| CALLM | Diary + TF-IDF RAG | 0.626 | 0.618 |
| V1 (Structured) | Sensing only | 0.521 | 0.453 |
| V2 (Agentic) | Sensing only | 0.598 | 0.591 |
| V3 (Structured) | Sensing + diary | 0.607 | 0.590 |
| **V4 (Agentic)** | **Sensing + diary** | **0.666** | **0.661** |
| V5 (Agentic+filtered) | Sensing only (filtered) | 0.601 | 0.596 |
| **V6 (Agentic+filtered)** | **Sensing + diary (filtered)** | **0.669** | **0.664** |

### ML/DL Baselines (5-fold CV, 399 users)

| Model | Input | Mean BA | Mean F1 |
|-------|-------|---------|---------|
| RF | Sensing | 0.501 | 0.365 |
| XGBoost | Sensing | 0.502 | 0.391 |
| MiniLM | Diary embeddings | 0.629 | 0.588 |
| Combined RF | Sensing + diary | 0.620 | 0.568 |

### Hypothesis Evaluation

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Sensing enhances diary-based prediction | **Supported** | V6 BA=0.669 > CALLM BA=0.626 |
| H2: LLM agents outperform ML baselines | **Supported** | V6 BA=0.669 > MiniLM BA=0.629 |
| H3: Passive sensing alone enables prediction | **Partially supported** | V2 BA=0.598 > chance, but < multimodal |
| H4: Agentic > structured | **Supported** | V4 BA=0.666 > V3 BA=0.607 |
| H5: Diary adds value beyond sensing | **Supported** | V4 BA=0.666 > V2 BA=0.598 |

## Architecture

```
Passive Sensing (8 modalities, 114 GB raw)
    → Hourly Feature Store (2 GB Parquet)
    → LLM Agent (Claude Sonnet, 15+ tools via Anthropic SDK)
    → Structured Prediction (PANAS, 12 binary affect states, receptivity)
    → Intervention Decision (desire ∧ availability)
```

### Agent Versions

| Version | Diary | Sensing | Style | Description |
|---------|-------|---------|-------|-------------|
| **CALLM** | Yes | No | Reactive | CHI 2025 baseline — diary + TF-IDF RAG |
| **V1** | No | Yes | Structured | Fixed 5-step pipeline over pre-formatted sensing |
| **V2** | No | Yes | Agentic | Autonomous tool-use queries on sensing data |
| **V3** | Yes | Yes | Structured | Multimodal RAG with diary + sensing context |
| **V4** | Yes | Yes | Agentic | Full agentic investigation — **key contribution** |
| **V5** | No | Yes (filtered) | Agentic | V2 + data quality filtering |
| **V6** | Yes | Yes (filtered) | Agentic | V4 + data quality filtering — **best system** |

### Agentic Investigation Loop (V2, V4, V5, V6)

The agent uses Anthropic SDK tool-use to query a Parquet-backed sensing database autonomously:

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

The agent decides **what to look at, not just what to predict** — querying selective signals and building explicit belief-update traces.

### Example: Agent Investigation Trace

**Scenario:** Participant 071, 2024-01-15, 2:00 PM EMA.
**Profile:** 62-year-old female, breast cancer stage II, 4 years post-treatment. PHQ-8 = 8 (mild depression).

> *Agent:* "I need to figure out this person's emotional state. Let me start with a big-picture look at today."

`CALL: get_daily_summary(date="2024-01-15")`
```
GPS: stayed home all day, 0 km travel. Screen: 3.4h total (high).
Motion: 98% stationary. Keyboard: 12 words typed (very low).
```
> *Belief update:* She barely moved, barely typed, barely left home. Is this unusual?

`CALL: compare_to_baseline(metric="gps_at_home_pct", lookback_days=14)`
```
14-day baseline: typically home 52%. Today: 100%.
Percentile: 97th. z-score: +2.8. Extremely unusual.
```
> *Belief update:* Strong anomaly — much more confined than usual.

`CALL: find_similar_days(top_k=3)`
```
2024-01-08: (home, high screen, no music) → PANAS_Neg=5, ER_desire=6
2023-12-22: (home, holiday)               → PANAS_Neg=3, ER_desire=2
2024-01-02: (home, low GPS)               → PANAS_Neg=5, ER_desire=5
```
> *Belief update:* The 2024-01-08 match showed highest negative affect. Concerning.

`CALL: get_receptivity_history(n=7)`
```
Last 7 days: 4 accepted, 2 rejected. Yesterday 2pm: rejected.
```

**Final prediction:** PANAS_Neg=5 (elevated), NA_State=True, availability=yes but with caution.
**Reasoning:** Extreme home confinement (97th pct) + behavioral match to past high-NA day + yesterday's rejection → elevated negative affect. Recommend lower-intensity contact.

## Dataset: BUCS Cancer Survivorship Study

- **418 participants** (297 iOS, 121 Android), ~5-week study
- **EMA**: 3x/day (morning, afternoon, evening), ~15,000+ entries
- **Passive sensing**: 8 modalities at hourly resolution (accelerometer, GPS, motion activity, screen time, keyboard input, music, ambient light, sleep)
- **Labels**: PANAS Pos/Neg, ER desire, 12 binary affect states, intervention availability
- **Data scale**: 114 GB raw sensor data → 2 GB processed Parquet features

Missing data handled via 4-class taxonomy: `OBSERVED`, `STRUCTURAL_MISSING` (platform limitation), `DEVICE_MISSING` (phone off), `PARTICIPANT_MISSING` (no activity).

## Project Structure

```
src/
├── agent/              # PersonalAgent dispatcher + V1-V6 implementations
│   ├── personal_agent.py   # Version routing (CALLM/V1-V6)
│   ├── structured.py       # V1: sensing-only structured pipeline
│   ├── structured_full.py  # V3: multimodal structured pipeline
│   └── cc_agent.py         # V2/V4/V5/V6: agentic via Anthropic SDK tool-use
├── sense/              # Sensing data engine (15+ query tools)
├── data/               # Data structures, preprocessing, schemas
├── think/              # LLM client, prompt templates, response parsing
├── remember/           # TF-IDF + multimodal retriever (diary+sensing RAG)
├── baselines/          # ML/DL baselines (RF, XGBoost, LogReg, MLP, LSTM)
├── evaluation/         # Metrics (BA, F1, MAE), reporting
└── utils/              # Config, constants, mappings

scripts/
├── run_pilot.py            # Main experiment runner (all versions)
├── evaluate_pilot.py       # Compute metrics from checkpoints
├── build_filtered_data.py  # Data quality filtering for V5/V6
├── baselines_50user.py     # ML baseline evaluation
└── offline/                # Sensing data processing pipeline

docs/
├── results-dashboard.html  # Interactive results dashboard
└── design.md               # System design document

tests/                      # 170+ tests
```

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env  # Add Anthropic API key

# Process sensing data
bash scripts/offline/run_phase0.sh    # Participant roster + home locations
bash scripts/offline/run_phase1.sh    # Light modalities (~30 min)
python scripts/offline/process_accel.py  # Accelerometer (heavy, run overnight)
python scripts/offline/process_gps.py    # GPS mobility (heavy, run overnight)
```

## Running Experiments

```bash
# Run all versions for specific users
python3 scripts/run_pilot.py --version all --users 71,164,119 --model sonnet

# Evaluate results
PYTHONPATH=. python3 scripts/evaluate_pilot.py

# ML baselines (5-fold CV)
python3 scripts/baselines_50user.py
```

## License

Research code. Contact authors for use.
