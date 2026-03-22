# Paper Outline: PULSE
## Target: IMWUT, ~25 pages (ACM acmart format)

### Abstract (~250 words)
- Problem: Cancer survivors need proactive mental health support; diary paradox
- Approach: PULSE — agentic LLM investigation of passive sensing via 8 MCP tools
- Evaluation: 2×2 factorial on 50 cancer survivors (~3,900 entries/version)
- Key results: Agentic multimodal BA=0.661 vs structured 0.603; ER_desire 0.751; INT_avail 0.706 from sensing alone
- Contribution: New paradigm of agentic sensing investigation

### 1. Introduction (~2 pages)
- Opening hook: Cancer survivors and mental health burden; JITAI promise
- The diary paradox: most informative data absent when most needed
- Passive sensing opportunity: continuous, unobtrusive, but traditional ML ~0.52 BA
- Core insight: LLM agents that autonomously investigate behavioral data (like a clinician reviewing charts)
- PULSE system overview: 8 sensing tools, 2×2 factorial design
- Contributions list (4 items: paradigm, system, factorial evaluation, clinical findings)
- Paper structure

### 2. Related Work (~3 pages)
- 2.1 Passive Sensing for Mental Health
  - Foundational: StudentLife (Wang et al. 2014), Saeb et al. (2015)
  - Benchmarks: GLOBEM (Xu et al. 2022)
  - Clinical populations: sensing in cancer, chronic disease
  - Gap: modest accuracy, lack of contextual interpretation

- 2.2 LLMs for Affective Computing and Health
  - Health-LLM (Kim et al. 2024), Mental-LLM (Xu et al. 2024)
  - Zhang/Bae et al. (2024) — LLMs for affective states from sensing
  - CALLM (Wang et al. 2025) — direct predecessor
  - GLOSS (Choube et al. 2025) — multi-agent sensemaking
  - PHIA (Nature Comms 2026) — LLM agent for wearable analysis
  - Gap: all use fixed prompts or post-hoc analysis; none give LLM autonomous investigation tools

- 2.3 Agentic AI and Tool Use
  - ReAct (Yao et al. 2023), Toolformer, MCP
  - Gap: agentic paradigm not applied to passive sensing investigation

- 2.4 Just-in-Time Adaptive Interventions
  - Nahum-Shani et al. (2018) — JITAI framework
  - Receptivity: Mishra et al. (2021), Kunzler et al. (2019)
  - Gap: JITAI theory identifies tailoring variables as critical but hard to estimate in real time

- Positioning table: compare PULSE vs 8-10 prior systems on key dimensions

### 3. System Design (~5 pages)
- 3.1 Problem Formulation
  - Inputs: sensing history, optional diary, user profile, memory
  - Outputs: continuous (PANAS-PA/NA, ER_desire) + binary states (16 targets)
  - Focus targets: ER_desire, INT_avail, PA_State, NA_State

- 3.2 Factorial Design (2×2 + baseline)
  - Table: {Structured, Agentic} × {Sensing-only, Multimodal}
  - 7 versions: CALLM, Struct-Sense, Auto-Sense, Struct-Multi, Auto-Multi, Auto-Sense+, Auto-Multi+
  - What each cell tests

- 3.3 BUCS Dataset
  - 418 cancer survivors, ~5 weeks, 3× daily EMA
  - 8 sensing modalities (motion, GPS, screen, keyboard, app, light, music, sleep)
  - Coverage by platform (iOS vs Android)
  - Data splits: 5-fold across-subject CV

- 3.4 Structured Agents (Struct-Sense, Struct-Multi)
  - Fixed pipeline: pre-formatted summary → step-by-step reasoning
  - RAG: TF-IDF (text) + sensing-based (cosine similarity)

- 3.5 Agentic Agents (Auto-Sense, Auto-Multi, Auto-Multi+)
  - MCP sensing query tools (8 tools — describe each)
  - Investigation strategy: agent decides what to examine
  - Multi-turn tool-use loop
  - Session memory: per-user reflections (no ground truth leakage)

- 3.6 Cross-User RAG for Calibration
  - Three modes: text, sensing, tool-based
  - Purpose: empirical grounding for predictions

### 4. Evaluation Methodology (~2 pages)
- 4.1 Evaluation Setup
  - 50 users from test folds, ~3,900 entries per version
  - Representativeness analysis (50 vs 418): in main text, not supplementary
  - Temporal boundary enforcement

- 4.2 Baselines
  - ML: RF, XGBoost, Logistic (5-fold CV, 399 users) — reference points
  - Text: TF-IDF+SVM, BoW+SVM, MiniLM+SVM
  - Note on evaluation asymmetry

- 4.3 Metrics
  - Binary: Balanced Accuracy (primary), Macro F1
  - Focus targets: ER_desire, INT_avail, PA_State, NA_State
  - Statistical tests: Wilcoxon signed-rank, bootstrap CIs, effect sizes

- 4.4 Implementation
  - Claude Sonnet via CLI, max 5 concurrent, 30-90s per prediction

### 5. Results (~5 pages)
- 5.1 Aggregate Performance (Table: all versions + baselines, Mean BA/F1)
  - Auto-Multi+ 0.661 > Auto-Multi 0.660 > CALLM 0.611 > Struct-Multi 0.603 > Auto-Sense 0.589 > Struct-Sense 0.516

- 5.2 Intervention Opportunity Detection
  - ER_desire: Auto-Multi+ 0.751, Auto-Multi 0.745
  - INT_avail: Auto-Multi 0.716, Auto-Sense 0.706
  - Key finding: INT_avail is behavioral — sensing >> diary

- 5.3 Affect State Detection
  - PA_State: Auto-Multi+ 0.733
  - NA_State: Auto-Multi+ 0.722

- 5.4 Agentic vs. Structured (the 2×2 result — LEAD WITH THIS)
  - Table: pairwise comparisons within each modality condition
  - Statistical significance: Wilcoxon, effect sizes, bootstrap CIs
  - Per-user BA distributions (violin/box plots)

- 5.5 Sensing-Only vs. Multimodal
  - Diary addition improves all targets, especially NA
  - INT_avail improvement minimal — sensing sufficient

- 5.6 Calibration Analysis
  - Continuous prediction biases (negativity for NA, mean regression for PA)
  - Binary robust despite continuous miscalibration

- 5.7 Agentic Investigation Patterns (Qualitative)
  - Example traces: which tools called, in what order, what the agent finds
  - Comparison: structured prompt vs. agentic investigation for same user/entry

### 6. Discussion (~3 pages)
- 6.1 Why Agentic Investigation Works
  - Selective attention, anomaly detection, analogical reasoning
  - Connect to BSP framework (Narayanan): contextual behavioral signal interpretation

- 6.2 The Diary Paradox and Graceful Degradation
  - Multimodal when available, sensing-only fallback
  - Practical JITAI deployment implications

- 6.3 INT_availability as a Behavioral Construct
  - Sensing captures what user is doing; diary captures what user is feeling
  - Design implication: separate desire from availability in JITAI systems

- 6.4 Cross-User RAG as Calibration (not knowledge retrieval)
  - Differs from typical RAG: empirical grounding, not factual knowledge

- 6.5 Limitations
  - Retrospective evaluation (not real-time)
  - N=50 (with representativeness analysis)
  - ML baseline asymmetry (399 vs 50)
  - Continuous calibration gap
  - Model dependency (Claude Sonnet)
  - Inference cost (30-90s per prediction)
  - No memory ablation

- 6.6 Implications for JITAI Design
  - Separate desire from availability
  - Use agentic reasoning for state detection
  - Design graceful degradation
  - Leverage population data for calibration

- 6.7 Ethical Considerations
  - IRB, consent, de-identification
  - Privacy: passive sensing + LLM API
  - Clinical safety: prediction layer only
  - Surveillance concerns + user autonomy

### 7. Future Work (~0.5 pages)
- Cross-model comparison (GPT, open-source LLMs)
- Real-time deployment (smartphone app, edge LLMs)
- Longitudinal personalization
- From prediction to intervention generation

### 8. Conclusion (~0.5 pages)
- PULSE introduces agentic sensing investigation for affect prediction
- 2×2 factorial proves agentic reasoning is the key differentiator
- Sensing-only prediction is viable for proactive JITAI
- Paradigm contribution: the community can adopt agentic investigation for behavioral data

### Appendices
- A: Full per-target results table (16 binary targets × 7 versions)
- B: Agentic investigation trace example (full tool-use sequence)
- C: ML baseline details (hyperparameters, features)
- D: Prompt templates for structured agents
