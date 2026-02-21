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

## 3. Two Agent Architectures

### 3.1 V1: Structured Agent

Fixed 4-step pipeline, same every time:

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

### 4.3 Train/Test Split

Use pre-existing 5-fold splits. Agent builds memory on train users, then is evaluated on test users. For test users, agent starts cold (or with population priors) and must learn during the evaluation period.

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

Pre-computed FAISS embeddings over 754 memory documents. Used for semantic similarity search — "find past situations similar to the current sensing pattern."

## 6. Self-Learning Loop

After each EMA arrives (ground truth), the agent:

1. **Self-evaluates**: Was my prediction correct? By how much was I off?
2. **Updates memory**: Records the outcome, notes any surprising patterns
3. **Adjusts thresholds**: If consistently over/under-confident, tunes thresholds

This creates a learning loop where the agent improves over a user's study participation period.

## 7. Evaluation Strategy

### 7.1 Metrics

- **Balanced Accuracy (BA)**: For categorical predictions (receptivity yes/no, emotion categories)
- **MAE**: For continuous predictions (valence, arousal scales)
- **F1**: For receptivity prediction (class-imbalanced)
- **Calibration**: Are confidence scores well-calibrated?

### 7.2 Comparisons

- V1 (Structured) vs V2 (Autonomous)
- Cold-start vs warm (with accumulated memory)
- With vs without peer knowledge
- Against CALLM baselines (where applicable)

## 8. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM as reasoner | Yes | Sensing features need contextual interpretation; statistical models alone miss nuance |
| Per-user memory | Markdown files | Simple, inspectable, version-controllable. No need for a database at this scale |
| Receptivity definition | Desire ∧ Availability | Unifies two constructs into one actionable signal |
| Simulation approach | Retrospective replay | Only option given we have historical data, not real-time access |
| Two agent versions | V1 + V2 | Tests whether autonomy helps or hurts in this domain |

## 9. Open Questions

- How much sensing data history is optimal per prediction? (Currently: 4h lookback)
- Should V2 agent have access to raw sensing data or only extracted features?
- How to handle missing sensing data gracefully? (Common in real-world mobile sensing)
- What intervention content is most appropriate? (Currently using templates)
- How to fairly compare V1 and V2 given V2's variable compute cost?
