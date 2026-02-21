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

## 3. Three Agent Versions (Pilot)

### 3.0 CALLM Baseline

CHI paper reactive approach — uses diary text (`emotion_driver`) as input:

1. **Retrieve**: TF-IDF similarity search over training set to find top-20 similar past cases
2. **Reason**: LLM reads diary text + similar cases + user memory doc + trait profile → predicts

This is the baseline to beat. It has a natural advantage on text-rich entries since the diary text directly reveals emotional state.

### 3.1 V1: Structured Agent

Fixed pipeline using passive sensing data only (no diary text):

1. **Sense**: Extract and summarize available sensing features for the current time window
2. **Retrieve Memory**: Query user's personal memory for relevant past patterns
3. **Reason**: LLM interprets current sensing + memory context → predicts emotional state and receptivity
4. **Decide**: Based on predictions and confidence, decide whether to intervene

**Advantages**: Reproducible, debuggable, consistent across runs.
**Disadvantages**: Cannot adapt its *process* — always follows the same steps regardless of context.

### 3.2 V2: Autonomous Agent

ReAct-style agent with tools:

- `query_sensing(sensor, window)` — Retrieve specific sensing data
- `read_memory(query)` — Search user's personal memory
- `check_peers(pattern)` — Find similar patterns in other users
- `retrieve_rag(query)` — Semantic search over memory documents
- `predict_affect(evidence)` — Make an emotional state prediction
- `check_history(user_id)` — Review past prediction accuracy
- `intervene(type, content)` — Deliver an intervention

The LLM decides which tools to use, in what order, and when to stop.

**Advantages**: Flexible — can skip unnecessary steps, deep-dive on anomalies, adapt process to context.
**Disadvantages**: Less predictable, harder to evaluate systematically, higher API cost.

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

TF-IDF retriever over training set `emotion_driver` texts. Used for CALLM version to find similar past cases by cosine similarity. (Original OpenAI embeddings are incompatible with Claude CLI backend.)

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

- CALLM (diary text, reactive) vs V1 (sensing, structured) vs V2 (sensing, autonomous)
- Per-user analysis across 5 selected users (highest data coverage)
- Personal threshold evaluation matching CHI paper methodology

## 8. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM as reasoner | Yes | Sensing features need contextual interpretation; statistical models alone miss nuance |
| Per-user memory | Markdown files | Simple, inspectable, version-controllable. No need for a database at this scale |
| Receptivity definition | Desire ∧ Availability | Unifies two constructs into one actionable signal |
| Simulation approach | Retrospective replay | Only option given we have historical data, not real-time access |
| Two agent versions | V1 + V2 | Tests whether autonomy helps or hurts in this domain |

## 9. Open Questions

- Sensing data is daily aggregates — is sub-day granularity worth pursuing?
- How to handle missing sensing data gracefully? (Common in real-world mobile sensing)
- Can V1/V2 beat CALLM despite lacking direct emotional text input?
- How to fairly compare V1 and V2 given V2's variable compute cost (2-3x calls)?
- Does personal threshold evaluation favor certain prediction patterns?
