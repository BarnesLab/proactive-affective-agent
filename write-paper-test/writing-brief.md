# PULSE Writing Brief — Section-by-Section Reference

**Paper**: PULSE — Agentic LLM Investigation of Passive Sensing for Proactive Affect Prediction and Intervention Opportunity Detection in Cancer Survivors
**Target**: IMWUT, ~25 pages | **Experts consulted**: 10 (Choudhury, Xu, Nahum-Shani, Campbell, Narayanan, Althoff, Mishra, Barnes, Spring, Kim)

---

## Abstract (~250 words)

**Consensus (5+)**: Open with diary paradox (7/10). State 2x2 factorial as primary evaluation (8/10). Report Auto-Multi+ 0.661 mean BA (all). Name the paradigm "agentic sensing investigation" (7/10).

**Key numbers**: Auto-Multi+ 0.661 mean BA | ER_desire 0.751 | INT_avail 0.706 sensing-only | 50 users, ~3,900 entries/version | 8 MCP tools | ML ceiling ~0.52 BA

**Phrasing**: "agentic sensing investigation" | "the very moments when intervention is most needed are precisely when self-report data is absent"

---

## 1. Introduction (~2 pages)

### Consensus (5+ experts)
- **Diary paradox** as clinical motivation: patients most in need disengage from self-report (Choudhury, Xu, Nahum-Shani, Campbell, Barnes, Mishra, Spring) — 7/10
- **ML ceiling ~0.52 BA** is well-documented and persistent across studies (Choudhury, Xu, Althoff, Campbell, Kim) — 5/10
- **2x2 factorial** cleanly isolates agentic effect from data modality (all 10)
- **PULSE is a prediction layer**, NOT a clinical intervention or full JITAI (Choudhury, Nahum-Shani, Spring, Barnes) — emphasized by 4, but universally implied
- **Paradigm shift**: from feature engineering to agentic investigation (all 10)

### Strong suggestions (3-4)
- Frame through JITAI tailoring variables: ER_desire = vulnerability/opportunity, INT_avail = receptivity (Nahum-Shani, Mishra, Spring)
- Ground in cancer survivorship mental health burden, 20-30% depression rates (Barnes, Spring, Campbell)
- Position in lineage: StudentLife -> GLOBEM -> Health-LLM -> PULSE (Xu, Althoff, Kim, Campbell)

### Individual suggestions
- BSP framing: agentic investigation as contextual behavioral signal interpretation (Narayanan)
- Receptivity reframing: diary paradox is fundamentally a receptivity problem (Mishra)

### Key numbers
- ML baselines: RF 0.518, XGBoost 0.514, Logistic 0.515
- Auto-Multi+ 0.661 vs ML best 0.518
- INT_avail: Auto-Sense 0.706 >> CALLM 0.542
- 418 cancer survivors, ~5 weeks, 3x daily EMA, 8 sensing modalities

### Citations
- Wang2014StudentLife, Wang2016CrossCheck, Saeb2015
- Xu2023GLOBEM, Adler2024BeyondDetection
- NahumShani2018JITAI, Mishra2021Receptivity
- Kim2024HealthLLM, Wang2025CALLM

### Pitfalls
- Do NOT claim real-time deployment readiness (retrospective only)
- Do NOT frame as clinical intervention system — prediction layer only
- Do NOT claim ML comparison is head-to-head (399 vs 50 users)
- Do NOT oversell: "shatters" is too strong; "exceeds" or "surpasses" the ceiling

### Phrasing fragments
- "closing the loop from behavioral sensing to proactive intervention delivery" (Choudhury)
- "the 2x2 factorial design cleanly isolates the contribution of agentic reasoning from data modality" (Althoff)
- "JITAI theory identifies tailoring variables as critical — but estimating these in real-time remains the central challenge" (Spring)
- "agentic investigation enables contextual interpretation that fixed feature pipelines fundamentally cannot provide" (Choudhury)
- "the system need not interrupt the user to determine whether to interrupt the user" (Nahum-Shani)

### Reviewer preemption
- "How is this different from prior sensing work?" -> paradigm shift + 2x2 factorial isolation
- "N=50 is small" -> representativeness analysis in main text (3/4 targets p > 0.05), within-subject design with ~3,900 entries/version

---

## 2. Related Work (~3 pages)

### Consensus (5+)
- **Three converging threads**: (1) passive sensing for MH, (2) LLMs for health/affect, (3) agentic AI + tool use. PULSE occupies the intersection. (Choudhury, Xu, Althoff, Kim, Mishra) — 5/10
- **Progression narrative**: fixed features -> LLM prompts -> agentic tools (Xu, Althoff, Kim, Campbell, Mishra) — 5/10
- **GLOBEM ceiling** as diagnostic of the field's limitations (Xu, Althoff, Choudhury, Kim) — 4/10 but strong
- **Do NOT dismiss prior work** — PULSE builds on it (Choudhury, Xu, Althoff, Kim, Barnes) — 5/10
- **Include positioning table** comparing PULSE vs 8-10 systems (all)

### Strong suggestions (3-4)
- CALLM is predecessor, not competitor — same team (Xu, Nahum-Shani, Kim)
- GLOSS vs PULSE distinction: sensemaking vs prediction (Mishra, Kim, Xu)
- JITAI literature: tailoring variables theorized but hard to operationalize (Nahum-Shani, Spring, Mishra)
- Health-LLM -> PHIA -> PULSE arc for LLM-for-health (Kim, Xu, Althoff)

### Individual suggestions
- BSP provides theoretical framework for why agentic works (Narayanan)
- Clinical populations underrepresented vs student populations (Barnes)
- MindScape comparison: journaling vs prediction (Campbell)
- LENS: narrative synthesis is complementary (Kim, Xu)

### Citations
- **2.1 Sensing**: Wang2014StudentLife, Saeb2015, Wang2016CrossCheck, Xu2023GLOBEM, Adler2022Generalization, Adler2024BeyondDetection, Huckins2020COVID, Nepal2024CollegeExperience, McClaine2024Engagement
- **2.2 LLMs**: Kim2024HealthLLM, Xu2024MentalLLM, Zhang2024LLMAffect, Wang2025CALLM, Choube2025GLOSS, Merrill2026PHIA, Nepal2024MindScape, Xu2025LENS
- **2.3 Agentic**: Yao2023ReAct, Schick2023Toolformer, Anthropic2024MCP
- **2.4 JITAIs**: NahumShani2018JITAI, NahumShani2023Vulnerability, Mishra2021Receptivity, Kunzler2019Receptivity, Klasnja2015MRT
- **Clinical**: Boukhechba2018SocialAnxiety, Chow2017MobileSensing, Cai2020BreastCancer
- **BSP/Affect**: Narayanan2013BSP, Watson1988PANAS, Gross2015EmotionRegulation

### Pitfalls
- Do NOT conflate GLOBEM (platform) with the ML models tested within it
- Do NOT claim PULSE "solves" generalization — not yet tested cross-dataset
- Do NOT frame LLMs as replacing clinical judgment — augmenting workflows
- Do NOT position against Health-LLM/GLOSS as competitors — next step / complementary

### Phrasing fragments
- "a decade of mobile sensing research has established feasibility of detecting behavioral correlates of mental health, yet translation to clinical action remains elusive" (Choudhury)
- "prior benchmarks reveal a generalization ceiling for fixed-feature approaches" (Althoff)
- "PULSE shifts the locus of intelligence from feature engineering to autonomous investigation" (Althoff/Xu)
- "while JITAI theory identifies tailoring variables as the lynchpin of adaptive intervention, most implementations rely on self-report" (Nahum-Shani)

### Positioning table dimensions
System | Year | Data Source | LLM? | Agentic? | Tools? | Clinical Pop? | Prediction Targets

---

## 3. System Design (~5 pages)

### Consensus (5+)
- **2x2 factorial** is a major methodological contribution — cleanly isolates architecture from modality (all 10)
- **8 MCP tools mirror clinical chart review** — selective investigation, not exhaustive feature extraction (Choudhury, Narayanan, Mishra, Campbell, Kim) — 5/10
- **Cross-user RAG = calibration** (empirical grounding), NOT knowledge retrieval (Choudhury, Xu, Althoff, Spring, Narayanan) — 5/10
- **Session memory**: per-user reflections without ground truth leakage (Choudhury, Mishra, Campbell) — 3+ explicit

### Strong suggestions (3-4)
- Factorial maps to JITAI design question: what data + reasoning best estimates tailoring variables (Nahum-Shani, Spring, Mishra)
- Tools should be presented as reusable toolkit for community (Xu, Kim, Campbell)
- 5-fold across-subject CV + temporal boundary enforcement for rigor (Althoff, Xu, Barnes)
- ER_desire = emotional/diary-aided; INT_avail = behavioral/sensing-captured — theoretically grounded distinction (Nahum-Shani, Spring, Mishra)

### Individual suggestions
- Each MCP tool = "channel" in BSP terms; agent does dynamic channel selection (Narayanan)
- Factorial echoes MOST framework for component optimization (Spring)
- Platform-specific coverage (iOS vs Android) is deployment reality; agent adapts (Campbell, Barnes)

### Key numbers
- 418 cancer survivors, ~5 weeks, 3x daily EMA
- 8 sensing modalities: motion, GPS, screen, keyboard, app (Android), light (Android), music, sleep
- 7 versions: CALLM, Struct-Sense, Auto-Sense, Struct-Multi, Auto-Multi, Auto-Sense+, Auto-Multi+
- 8 MCP tools: daily summary, behavioral timeline, targeted hourly query, raw events, baseline comparison, receptivity history, similar days, peer cases
- 3 RAG modes: text-based, sensing-based, tool-based
- 5-fold across-subject CV

### Citations
- NahumShani2018JITAI (tailoring variable framing)
- Yao2023ReAct, Anthropic2024MCP (agentic architecture)
- Wang2025CALLM (predecessor/baseline)

### Pitfalls
- Do NOT anthropomorphize agent excessively — "tool-augmented reasoning," not "thinking like a clinician"
- Do NOT understate MCP technical contribution — these are novel domain-specific tools
- Do NOT oversell RAG as "knowledge retrieval"
- Do NOT gloss over platform differences — affects ~25% of users
- Do NOT overclaim MOST alignment — optimizes prediction components, not intervention

### Phrasing fragments
- "the agent's investigation strategy emerges from the interaction between its clinical reasoning and available tools, rather than being pre-specified by the system designer" (Choudhury)
- "the factorial design enables clean causal attribution — the agentic architecture, not merely richer data, drives the performance gain" (Althoff)
- "cross-user RAG operationalizes the intuition behind population-level calibration" (Xu)
- "domain-specific tools provide guardrails: the agent cannot accidentally access future data" (Kim)

### Reviewer preemption
- "Data leakage through memory?" -> Reflections only, no ground truth stored
- "Why MCP not custom API?" -> Open standard, reproducibility
- "Why not multi-agent like GLOSS?" -> Single-agent + tools is simpler, sufficient for prediction
- "Why not code generation like PHIA?" -> Tools enforce temporal boundaries, safer for clinical

---

## 4. Evaluation Methodology (~2 pages)

### Consensus (5+)
- **Balanced accuracy** as primary metric — essential for imbalanced health data (Althoff, Xu, Mishra, Barnes, Kim) — 5/10
- **Representativeness analysis in MAIN TEXT**, not supplementary (Choudhury, Althoff, Barnes, Campbell) — 4+ strong
- **ML baselines = reference points**, NOT head-to-head (Choudhury, Xu, Althoff, Kim, Nahum-Shani) — 5/10
- **Per-user BA distributions** essential — show advantage is consistent, not outlier-driven (Althoff, Xu, Campbell, Narayanan) — 4/10
- **Effect sizes + bootstrap CIs** alongside p-values (Althoff, Xu, Narayanan) — 3+ strong

### Strong suggestions (3-4)
- Acknowledge 50 vs 399 asymmetry explicitly but note factorial is apples-to-apples (Choudhury, Althoff, Xu, Kim)
- Wilcoxon signed-rank appropriate for paired, non-normal data (Althoff, Xu)
- Per-user evaluation prevents high-compliance users from dominating (Xu)

### Individual suggestions
- PANAS is gold standard for momentary affect measurement (Narayanan)
- Binarization loses info but necessary for clinical decisions (Narayanan)
- Focus targets grounded in JITAI theory, not just predictability (Spring)

### Key numbers
- 50 users from test folds, ~3,900 entries/version, ~27,300 total LLM inferences
- Representativeness: PA_State, ER_desire, INT_avail p > 0.05 vs full 418; NA_State p = 0.028 (small effect r = 0.19)
- EMA count: 82 vs 34 mean (by design — high-compliance selection)
- Claude Sonnet via CLI, max 5 concurrent, 30-90s per prediction

### Citations
- Watson1988PANAS, Hicks2019BestPractices
- Xu2023GLOBEM (methodological rigor standard)

### Pitfalls
- Do NOT bury evaluation asymmetry — state clearly
- Do NOT rely solely on p-values — effect sizes mandatory
- Do NOT compare LLM and ML as apples-to-apples
- Do NOT conflate statistical significance with clinical meaningfulness
- Do NOT bury representativeness in supplementary

### Phrasing fragments
- "ML baselines serve as calibration reference points for the sensing feature ceiling, not as direct comparisons" (Choudhury/Althoff)
- "per-user balanced accuracy distributions reveal that the agentic advantage is consistent, not driven by outliers" (Althoff)
- "we report effect sizes (r) alongside p-values, following best practices for behavioral data analysis" (Althoff)

### Reviewer preemption
- "Unfair ML comparison" -> Acknowledged; factorial is the primary evaluation
- "N=50 is small" -> Representativeness + within-subject + ~3,900 entries/version
- "Why only 50?" -> Cost: ~27,300 LLM inferences for 7 versions
- "Is the 2x2 causal?" -> Quasi-experimental; strongest design possible at this scale

---

## 5. Results (~5 pages)

### Consensus (5+)
- **LEAD with 2x2 factorial**: Agentic >> Structured is the headline (all 10)
- **INT_avail from sensing is the standout clinical finding** (Choudhury, Nahum-Shani, Mishra, Spring, Barnes, Kim) — 6/10
- **Per-user distributions** alongside aggregates (Althoff, Xu, Campbell, Narayanan, Barnes) — 5/10
- **Filtering is negligible**: Auto-Multi+ 0.661 ~ Auto-Multi 0.660, do NOT emphasize (Choudhury, Althoff, Xu) — 3+ but consensus to de-emphasize
- **Contextualize 0.66 BA** — these are 16-target averages; individual targets reach 0.75 (Choudhury, Xu, Barnes)

### Strong suggestions (3-4)
- Effect sizes r > 0.9 are "exceptionally large" for behavioral data — contextualize (Althoff, Xu, Narayanan)
- Desire-availability dissociation: ER_desire benefits from diary, INT_avail from sensing (Nahum-Shani, Mishra, Spring)
- Qualitative investigation traces are essential for contribution clarity (Choudhury, Campbell, Narayanan)
- Report tool usage statistics alongside example traces to address cherry-picking (Campbell, Narayanan)

### Individual suggestions
- Auto-Multi+ 0.661 is highest BA ever for sensing-based affect prediction in clinical population (Xu)
- NA is harder to predict — people mask distress; 0.722 is notable (Narayanan)
- Clinical framing: ER_desire 0.751 means detecting 3/4 moments when support is wanted (Barnes, Spring)

### Key numbers
- **2x2 factorial**: Auto-Multi 0.660 vs Struct-Multi 0.603; Auto-Sense 0.589 vs Struct-Sense 0.516
- **Statistical**: p < 10^-10, r > 0.9 (Wilcoxon signed-rank)
- **ER_desire**: Auto-Multi+ 0.751, Auto-Multi 0.745, CALLM 0.632
- **INT_avail**: Auto-Multi 0.716, Auto-Sense 0.706, CALLM 0.542, Struct-Multi 0.551
- **Affect**: PA_State 0.733, NA_State 0.722 (Auto-Multi+)
- **Sensing-only**: PA 0.598, NA 0.592 (Auto-Sense)
- **INT_avail diary gain**: 0.716 vs 0.706 — marginal (0.01)
- **ER_desire diary gain**: 0.751 vs 0.653 — substantial
- **Filtering**: 0.661 vs 0.660 — negligible
- **CALLM**: 0.611 overall
- **ML**: RF 0.518, XGBoost 0.514, Logistic 0.515

### Citations
- Hicks2019BestPractices (reporting standards)
- Comparative references for effect size context

### Pitfalls
- Do NOT overclaim filtering adds value
- Do NOT present aggregate BA without per-user distributions
- Do NOT present absolute numbers without context (baselines + effect sizes)
- Do NOT translate BA directly to sensitivity/specificity without per-class analysis
- Do NOT directly compare PULSE BA to Health-LLM accuracy (different metrics/tasks)

### Phrasing fragments
- "the agentic investigation paradigm is the primary driver of prediction accuracy, independent of data modality" (Choudhury)
- "the effect size (r > 0.9) is exceptionally large by standards of behavioral sensing research, where r = 0.3-0.5 is typical" (Althoff)
- "INT_avail is fundamentally behavioral — best captured by passive sensing, not self-report" (Mishra)
- "the negligible improvement from diary for INT_avail (0.716 vs 0.706), contrasted with substantial improvement for ER_desire (0.751 vs 0.653), reveals a clean dissociation between behavioral and emotional constructs" (Mishra)
- "the minimal gain from filtering (0.661 vs 0.660) suggests agentic agents are inherently robust to noisy input" (Althoff)

### Reviewer preemption
- "0.66 is modest" -> 16-target average; individual targets reach 0.75; compare to ML ~0.52
- "Effect sizes inflated?" -> Per-user distributions show consistent advantage, not outlier-driven
- "Only one LLM" -> Architecture contribution is model-agnostic; cross-model is future work
- "Traces are cherry-picked" -> Summary statistics of tool usage patterns alongside examples

---

## 6. Discussion (~3 pages)

### Consensus (5+)
- **6.1 Why agentic works**: selective attention, anomaly detection, contextual interpretation — like clinician chart review (Choudhury, Campbell, Narayanan, Mishra, Kim) — 5/10
- **6.2 Diary paradox -> graceful degradation**: multimodal when available, sensing-only fallback (Choudhury, Xu, Spring, Barnes, Mishra) — 5/10
- **6.3 INT_avail as behavioral construct**: sensing >> diary; separate desire from availability (Choudhury, Nahum-Shani, Mishra, Spring, Barnes) — 5/10
- **6.5 Limitations — be transparent**: retrospective, N=50, ML asymmetry, model dependency, cost, no memory ablation (all 10)
- **Do NOT claim deployment readiness or generalizability** (all 10)

### Strong suggestions (3-4)
- Connect to BSP framework: contextual behavioral signal interpretation (Narayanan, Choudhury)
- Connect to Adler2024 "Beyond Detection" — PULSE operationalizes actionable sensing (Choudhury)
- Cross-user RAG as calibration distinct from typical RAG (Choudhury, Xu, Althoff)
- Four JITAI design implications: (1) separate desire/availability, (2) agentic > fixed rules, (3) graceful degradation, (4) population calibration (Nahum-Shani, Spring, Mishra)
- "This is just prompt engineering" rebuttal: 2x2 proves architecture, not prompt (Choudhury, Xu)

### Individual suggestions
- Calibration: negativity bias for NA, mean regression for PA — addressable via post-hoc calibration (Narayanan)
- Binary robust despite continuous miscalibration (Narayanan, Althoff)
- RAG as empirical Bayes estimation analogue (Althoff)
- Batch processing (every few hours) is feasible even with 30-90s latency (Campbell)
- Integration with Spring's health-promoting system framework (Spring)
- False positives (unnecessary check-in) = low cost; false negatives (missed distress) = high cost (Barnes)

### Key numbers
- INT_avail: Auto-Sense 0.706 vs CALLM 0.542
- Graceful degradation: 0.660 -> 0.589 overall; 0.716 -> 0.706 for INT_avail
- Struct-Sense 0.516 is NOT viable; Auto-Sense 0.589 IS viable
- 30-90s inference latency per prediction

### Citations
- Narayanan2013BSP, Adler2024BeyondDetection
- NahumShani2018JITAI, NahumShani2023Vulnerability
- Spring2019CancerSurvivors
- Gross2015EmotionRegulation

### Pitfalls
- Do NOT claim generalizability beyond cancer survivors
- Do NOT speculate about real-time deployment without acknowledging latency
- Do NOT claim PULSE is first to reason contextually — BSP has always emphasized context
- Do NOT overclaim on clinical integration without discussing regulatory barriers
- Do NOT claim system replaces clinical assessment
- Do NOT conflate prediction accuracy with intervention effectiveness

### Phrasing fragments
- "the sensing-to-intervention gap can be narrowed not by better sensors, but by smarter reasoning over existing data streams" (Choudhury)
- "the field has focused on whether sensing data contains mental health signals; PULSE shifts the question to how an intelligent agent can investigate those signals in context" (Choudhury)
- "JITAI designers should model intervention desire and intervention availability as separate constructs, using different data sources for each" (Mishra/Nahum-Shani)
- "a two-channel system: passive sensing continuously estimates behavioral availability, while brief check-ins assess emotional desire" (Spring)
- "the key insight from BSP is that contextual interpretation, not just extraction, is the bottleneck for affect prediction from in-the-wild data" (Narayanan)
- "graceful degradation from multimodal (0.660) to sensing-only (0.589) ensures the system remains clinically useful even during diary gaps that characterize acute distress" (Barnes)

### Reviewer preemption
- "This is just prompt engineering" -> 2x2 factorial proves architecture is the differentiator
- "Model dependency on Claude" -> Paradigm is model-agnostic; cross-model is future work
- "LLM is a black box" -> Agentic traces provide observability; MORE interpretable than black-box ML
- "Latency prohibits JITAI" -> Edge LLMs, distillation; insight valid regardless of current latency
- "Privacy with LLM API" -> De-identified data; future: on-device LLMs
- "No memory ablation" -> Acknowledged; future work

---

## 7. Future Work (~0.5 pages)

### Consensus (5+)
- **Cross-model comparison** (GPT, open-source LLMs) (Xu, Althoff, Kim, Narayanan, Campbell) — 5/10
- **Real-time deployment** / edge LLMs (Choudhury, Campbell, Kim, Spring, Mishra) — 5/10

### Strong suggestions (3-4)
- Prospective evaluation in JITAI framework / micro-randomized trial (Nahum-Shani, Spring, Mishra)
- Cross-population testing (students, other chronic illness) (Campbell, Barnes, Xu)
- From prediction to intervention generation — close the full loop (Choudhury, Nahum-Shani)

### Individual suggestions
- Richer behavioral signals: speech prosody, wearable physiology as additional MCP tools (Narayanan)
- Combine Health-LLM's fine-tuning efficiency with PULSE's agentic reasoning (Kim)
- Longitudinal personalization (outline)
- Model distillation for deployment efficiency (Kim, Spring)

### Pitfalls
- Keep brief — 0.5 pages max
- Do NOT promise what cannot be delivered

---

## 8. Conclusion (~0.5 pages)

### Consensus (5+)
- **Paradigm contribution**: agentic sensing investigation is adoptable by the community (all 10)
- **2x2 factorial proves agentic reasoning is the key differentiator** (all 10)
- **Sensing-only is viable** for proactive JITAI (Choudhury, Mishra, Spring, Barnes, Xu) — 5/10

### Phrasing fragments
- "agentic sensing investigation represents a new paradigm for behavioral signal processing" (Narayanan)
- "the community can adopt agentic investigation for behavioral data beyond cancer survivors" (Choudhury)

---

## Ethics Section (within 6.7)

### Consensus (5+)
- IRB, consent, de-identification — state clearly (all)
- **Passive sensing + LLM API = sensitive pipeline** (Narayanan, Barnes, Spring) — 3+ explicit
- Clinical safety: prediction layer only, not diagnostic (Choudhury, Barnes, Spring)

### Individual suggestions
- Surveillance vs benefit distinction; cancer survivors may be more accepting when monitoring serves their care (Barnes)
- User autonomy and consent for ongoing passive monitoring (Narayanan)

---

## Universal Reviewer Preemption Summary

These 8 concerns will arise regardless. Prepare responses:

1. **"LLM expensive"** -> 30-90s/prediction; proof-of-concept; costs decreasing
2. **"Only Claude"** -> Architecture is model-agnostic; cross-model future work; MCP is open protocol
3. **"N=50"** -> Representativeness (3/4 p > 0.05); within-subject; ~3,900 entries/version
4. **"Retrospective"** -> Necessary first step; prospective future work
5. **"Privacy + LLM API"** -> De-identified; on-device LLMs future work
6. **"Unfair ML"** -> Reference points, not head-to-head; factorial is apples-to-apples
7. **"No memory ablation"** -> Acknowledged limitation
8. **"Why not fine-tune?"** -> Agentic investigation requires tool use + multi-turn; cannot be collapsed to fine-tuning

---

## Do NOT Overclaim Checklist (from framing-consensus)

- [ ] Real-time deployment readiness
- [ ] Generalizability beyond cancer survivors
- [ ] Fair ML baseline comparison
- [ ] Filtering adds meaningful value (0.661 ~ 0.660)
- [ ] Cost efficiency or scalability
- [ ] Memory system improves performance (no ablation)
- [ ] CALLM is "surpassed" — PULSE extends it
- [ ] Clinical intervention effectiveness
- [ ] System replaces clinical assessment
