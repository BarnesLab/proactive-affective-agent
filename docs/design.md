# Proactive Affective Agent — Design Document

> Living document. Updated as the system evolves.

## 1. Motivation

CALLM (CHI'26) demonstrated that LLMs can infer emotional states from diary text — but it is **reactive**: the user must actively write. Many users skip diary entries when they need help most (high stress, low motivation). We need a system that acts **proactively**, using passive sensing to anticipate emotional states and intervention needs.

## 2. Core Concepts

### 2.1 Receptivity = Desire ∧ Availability

Previous work treats "willingness to receive intervention" and "availability to engage" as separate constructs. We unify them:

- **Desire** (`ER_desire_State`): Does the user *want* emotional regulation support right now?
- **Availability** (`INT_availability`): Is the user *able* to engage with an intervention right now?
- **Receptivity**: Both must be true. An available but uninterested user won't engage. An interested but busy user can't engage.

This is the agent's primary prediction target: **"Is this user receptive to intervention right now?"**

### 2.2 PersonalAgent

Each participant gets a dedicated agent instance that:
- Maintains persistent memory (personal patterns, prediction history, learned thresholds)
- Evolves through self-evaluation (compares predictions to EMA ground truth)
- Can query shared knowledge (population norms, peer cases)

This mirrors clinical practice: a therapist builds a model of each patient over time.

### 2.3 Sensing → Affect → Receptivity Pipeline

```
Passive Sensing Data (accelerometer, GPS, sleep, screen, app usage, ...)
        ↓
Feature Extraction (activity level, location entropy, sleep quality, ...)
        ↓
LLM Reasoning (interpret features in context of user history)
        ↓
Predictions (emotional state + receptivity)
        ↓
Decision (intervene / wait / observe)
```

## 3. Five Agent Versions + ML Baselines

### 3.0 CALLM Baseline

CHI paper reactive approach — uses diary text (`emotion_driver`) as input:

1. **Retrieve**: TF-IDF similarity search over training set to find top-20 similar past cases
2. **Reason**: LLM reads diary text + similar cases + user memory doc + trait profile → predicts

This is the baseline to beat. It has a natural advantage on text-rich entries since the diary text directly reveals emotional state.

### 3.1 V1: Structured (Sensing Only)

Fixed pipeline using passive sensing data only (no diary text):

1. **Sleep Analysis**: What do the sleep metrics suggest about rest quality?
2. **Mobility & Activity**: What do GPS, motion, and screen data reveal?
3. **Social Signals**: What do typing patterns and app usage suggest?
4. **Pattern Integration**: How do these signals combine?
5. **User Context**: Given this user's history and traits, predict.

**Advantages**: Reproducible, debuggable, consistent across runs.
**Disadvantages**: Cannot adapt its process; no diary input.

### 3.2 V2: Autonomous (Sensing Only)

Single-call autonomous agent — receives all sensing data + memory, freely reasons about which signals matter most.

**Advantages**: Flexible — can focus on the most informative signals for this user/context.
**Disadvantages**: Less predictable; no diary input.

### 3.3 V3: Structured Full (Diary + Sensing + RAG)

Combines all three data modalities with a fixed 5-step pipeline:

1. **Diary Analysis**: Identify emotional themes, coping language, distress indicators
2. **Sensing Analysis**: Interpret sleep, mobility, screen, typing patterns
3. **Cross-Modal Consistency**: Do diary and sensing tell a consistent story?
4. **Similar Case Comparison**: How do RAG-retrieved cases (diary + sensing) compare?
5. **Integrated Prediction**: Synthesize all evidence

Uses `MultiModalRetriever`: TF-IDF search on diary text, but returns matched cases with their sensing data attached.

### 3.4 V4: Autonomous Full (Diary + Sensing + RAG)

Same data as V3, but the LLM autonomously decides how to weigh and reason about all evidence. No fixed pipeline.

### 3.5 ML Baselines

Traditional ML models on sensor feature vectors (no LLM calls):

- **RandomForest**: Non-linear, handles missing features well
- **XGBoost**: Gradient boosted trees, strong on tabular data
- **LogisticRegression**: Linear baseline for binary targets (class-weighted)
- **Ridge**: Linear baseline for continuous targets

5-fold CV using the same train/test splits as LLM versions. Features: 20 daily aggregate sensing features (sleep, mobility, activity, screen, typing, apps).

### 3.6 Version Comparison Matrix

| Version | Diary | Sensing | RAG | Pipeline | LLM |
|---------|-------|---------|-----|----------|-----|
| CALLM | ✅ | ❌ | TF-IDF diary | Reactive | 1 call |
| V1 | ❌ | ✅ | memory only | Structured | 1 call |
| V2 | ❌ | ✅ | memory only | Autonomous | 1 call |
| V3 | ✅ | ✅ | diary→diary+sensing | Structured | 1 call |
| V4 | ✅ | ✅ | diary→diary+sensing | Autonomous | 1 call |
| ML | ❌ | ✅ | N/A | Traditional | 0 |

## 4. Simulation Design

### 4.1 Retrospective Replay

We don't have real-time access to participants. Instead, we **replay historical data chronologically**:

```
For each user:
    For each day in study period:
        For each EMA window (morning, afternoon, evening):
            1. Gather sensing data BEFORE this window (agent only sees past data)
            2. Agent makes predictions
            3. Reveal EMA ground truth
            4. Agent self-evaluates and updates memory
```

### 4.2 Information Boundary

Critical constraint: **The agent must never see future data.** At each decision point:
- Sensing data: only data timestamped *before* the current EMA window
- EMA data: only *past* EMA responses (not the current one being predicted)
- Memory: accumulated from past interactions only

### 4.3 Data

- 399 users, 15,984 EMA entries across 5 non-overlapping test splits (combined for evaluation)
- 8 sensing streams as **daily aggregates** (one row per user per day, not raw streams)
  - Hourly/minute-level raw data expected from collaborator (will enable richer features)
- 756 pre-generated memory documents
- Baseline trait questionnaire (346 features)
- LLM backend: Claude Code CLI (`claude -p --model sonnet`), Max subscription

## 5. Memory Architecture

### 5.1 Per-User Memory (`user_memory.md`)

Markdown document maintained per user, containing:
- **Baseline profile**: Trait data, demographics (from baseline survey)
- **Patterns**: Learned behavioral patterns ("this user's sleep drops before high-stress days")
- **Prediction log**: Recent predictions and their outcomes
- **Thresholds**: Personalized confidence thresholds (adapted over time)

### 5.2 Shared Knowledge

- **Peer case library**: Anonymized cases from other users with similar profiles
- **Population norms**: Statistical summaries across all users (mean sleep, typical activity, etc.)
- **Cross-user patterns**: Patterns that generalize (e.g., "poor sleep → higher stress next day" across most users)

### 5.3 RAG (Retrieval-Augmented Generation)

Two retriever classes:

- **TFIDFRetriever**: TF-IDF over training set `emotion_driver` texts. Used by CALLM to find similar cases by cosine similarity. Returns diary text + outcomes.
- **MultiModalRetriever** (extends TFIDFRetriever): Same TF-IDF search, but enriches results with the matched participant's sensing data for that date. Used by V3/V4. Returns diary text + sensing data + outcomes.

(Original OpenAI embeddings from CHI paper are incompatible with Claude CLI backend.)

## 6. Self-Learning Loop

After each EMA arrives (ground truth), the agent:

1. **Self-evaluates**: Was my prediction correct? By how much was I off?
2. **Updates memory**: Records the outcome, notes any surprising patterns
3. **Adjusts thresholds**: If consistently over/under-confident, tunes thresholds

This creates a learning loop where the agent improves over a user's study participation period.

## 7. Evaluation Strategy

### 7.1 Metrics

Two evaluation approaches:

**Regression (continuous targets)**:
- **MAE**: For PANAS_Pos (0-30), PANAS_Neg (0-30), ER_desire (0-10)

**Classification (binary states)**:
- **Balanced Accuracy (BA)** and **F1**: For 15 Individual_level_*_State targets and INT_availability
- **CHI paper personal threshold**: Per-user mean ± SD on predicted continuous values → binary classification → BA & F1

### 7.2 Comparisons

- 5 LLM versions: CALLM vs V1 vs V2 vs V3 vs V4
- ML baselines: RF, XGBoost, LogReg, Ridge (5-fold CV, no LLM)
- Per-user analysis across 5 pilot users (71, 164, 119, 458, 310)
- Personal threshold evaluation matching CHI paper methodology
- Unified comparison with markdown + LaTeX table generation

## 8. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM as reasoner | Yes | Sensing features need contextual interpretation; statistical models alone miss nuance |
| Per-user memory | Markdown files | Simple, inspectable, version-controllable. No need for a database at this scale |
| Receptivity definition | Desire ∧ Availability | Unifies two constructs into one actionable signal |
| Simulation approach | Retrospective replay | Only option given we have historical data, not real-time access |
| Five agent versions | CALLM + V1-V4 | Tests effect of data modalities (diary vs sensing vs both) and reasoning style (structured vs autonomous) |
| ML baselines | RF/XGBoost/LogReg | Establishes non-LLM reference point; tests whether LLM reasoning adds value over feature engineering |
| MultiModal RAG | TF-IDF diary → return diary+sensing | Enriches retrieved cases with sensing context for V3/V4 |

## 9. Phase 1 Findings

- **CALLM >> V1/V2**: Diary text + RAG (MAE ~1.16) crushes daily aggregate sensing only (MAE ~7.06)
- **V1 ≈ V2**: Structured vs autonomous makes little difference when sensing data is too coarse
- **Implication**: Daily aggregate sensing lacks signal. Need hourly/minute-level features

## 10. Open Questions

- ~~Sensing data is daily aggregates — is sub-day granularity worth pursuing?~~ **Yes — Phase 1 showed daily aggregates are insufficient**
- Will hourly features close the gap between sensing-based versions and CALLM?
- Can V3/V4 (diary + sensing + RAG) surpass CALLM by adding sensing context?
- Do ML baselines outperform LLM-based sensing-only versions (V1/V2)?
- How to handle missing sensing data gracefully? (Common in real-world mobile sensing)
- Does personal threshold evaluation favor certain prediction patterns?
