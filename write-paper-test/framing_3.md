# Framing Proposal #3 — Inbal Nahum-Shani Perspective

*Perspective: Intervention science, JITAI design, receptivity prediction, adaptive treatment*

---

## 1. Title Options (Ranked)

1. **PULSE: Proactive Prediction of Intervention Receptivity via Agentic LLM Investigation of Passive Smartphone Sensing**
   - Best because it foregrounds the *proactive* and *intervention receptivity* framing, which is the core scientific contribution. "Agentic LLM investigation" signals the novel methodology. "Passive smartphone sensing" grounds it in the IMWUT sensing tradition.

2. **From Reactive Diaries to Proactive Sensing: LLM Agents that Autonomously Investigate Behavioral Data to Predict Emotional States and Intervention Opportunities**
   - Stronger narrative framing (reactive-to-proactive arc), but long. Works if IMWUT allows it.

3. **Agentic Sensing: Autonomous LLM Investigation of Smartphone Behavioral Data for Just-in-Time Affective State Prediction in Cancer Survivors**
   - Most descriptive. Anchors to the clinical population and the JITAI literature. Slightly less punchy.

---

## 2. Core Contribution Framing

This paper introduces and validates *agentic sensing investigation* — the idea that an LLM agent, equipped with purpose-built query tools over passive smartphone sensing streams, can autonomously decide what behavioral evidence to examine, how far back to look, and which cross-user comparisons to make, in order to predict a person's emotional state and receptivity to intervention. The key scientific story is not merely "LLMs are better than ML baselines" but rather that **the agent's capacity for autonomous, contextualized investigation of heterogeneous behavioral signals produces fundamentally different (and superior) predictions compared to structured pipelines operating on the same data** — and that this capacity is precisely what is needed to solve the JITAI tailoring variable estimation problem at the individual level, in real time, from passive data alone.

The secondary story — equally important from an intervention science standpoint — is the **diary paradox**: the most informative data source (self-report diary) is absent precisely when users are most distressed or unavailable, which is exactly when proactive intervention is most needed. PULSE demonstrates that autonomous sensing investigation can approach multimodal performance even without diary text, offering a path to truly passive, always-on intervention timing systems.

---

## 3. Positioning

### The gap this paper fills

The JITAI framework (Nahum-Shani et al., 2018) identifies **tailoring variables** — the time-varying states that should trigger intervention delivery — as a critical design component. The framework is clear that these variables must be estimable in real time from available data. Yet the field faces a fundamental bottleneck: the most common approach to estimating tailoring variables (EMA/self-report) imposes respondent burden, suffers from systematic missingness when states are most clinically relevant, and cannot scale to continuous monitoring. Passive sensing offers a theoretical alternative, but traditional ML approaches to sensing-based prediction have shown modest performance (typically BA around 0.50-0.55 on individual-level states), insufficient for clinical deployment.

Meanwhile, the LLM-for-health literature has demonstrated text-based prediction capabilities but has not solved the passive-sensing prediction problem, and has not connected to the JITAI design framework at all.

PULSE fills the intersection: it is the first system to demonstrate that LLM agents can autonomously investigate passive sensing data to predict JITAI-relevant tailoring variables (affect states, emotion regulation desire, intervention availability) with clinically meaningful accuracy, validated on a real cancer survivor population.

### Key comparison papers (all verified)

**JITAI Framework & Receptivity:**
- **Nahum-Shani et al. (2018).** "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support." *Annals of Behavioral Medicine*, 52(6), 446-462. — The foundational JITAI framework paper. PULSE directly addresses the tailoring variable estimation problem this paper identifies.
- **Nahum-Shani et al. (2014).** "Just-in-Time Adaptive Interventions (JITAIs): An Organizing Framework for Ongoing Health Behavior Support." *Methodology Center Technical Report*, Penn State. — The original six-component JITAI model (distal outcomes, proximal outcomes, tailoring variables, decision points, decision rules, intervention options). PULSE operationalizes the tailoring variable component.
- **Klasnja et al. (2015).** "Microrandomized Trials: An Experimental Design for Developing Just-in-Time Adaptive Interventions." *Health Psychology*, 34(Suppl), 1220-1228. — Establishes the MRT methodology. PULSE provides the predictive engine that would feed into an MRT's decision rules.
- **Kunzler et al. (2019).** "Exploring the State-of-Receptivity for mHealth Interventions." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 3(4), Article 140. — 189 participants, 6-week study predicting receptivity from contextual sensing. Achieved 77% improvement in F1 over random. PULSE targets the same construct (receptivity = desire AND availability) but uses LLM-based reasoning rather than traditional ML features.
- **Mishra, Kunzler et al. (2021).** "Detecting Receptivity for mHealth Interventions in the Natural Environment." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 5(2), Article 74. — Deployed ML-based receptivity detection in-the-wild, showing up to 40% improvement with adaptive models. Key comparison for receptivity prediction performance.

**Passive Sensing for Mental Health:**
- **Wang et al. (2014).** "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones." *Proceedings of the ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)*. Winner of the 2024 UbiComp 10-Year Impact Award. — Foundational passive sensing study. PULSE extends this tradition by replacing handcrafted features + correlation analysis with autonomous LLM investigation.
- **Saeb et al. (2015).** "Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study." *Journal of Medical Internet Research*, 17(7), e175. — Early demonstration that GPS and phone usage features correlate with depression. Illustrative of the traditional feature-engineering approach that PULSE's agentic investigation supersedes.

**LLMs for Mental Health and Sensing:**
- **Wang et al. (2025).** "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models." *arXiv:2503.10707* (CHI 2025/2026). — The direct predecessor to PULSE. Uses LLM + RAG on diary text. PULSE extends CALLM from reactive (diary-dependent) to proactive (sensing-based) prediction on the same BUCS dataset.
- **Xu et al. (2024).** "Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 8(1). — Demonstrates LLM capabilities for mental health prediction from text. Key comparison for showing that PULSE works from *sensing data*, not just text.
- **Zhang et al. (2024).** "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features." *Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '24)*. — First work using LLMs for affective prediction from smartphone sensing. However, uses a simple prompt-based approach (not agentic investigation). PULSE's key differentiation: autonomous tool use vs. static prompting.
- **Choube et al. (2025).** "GLOSS: Group of LLMs for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 9(3), Article 76. — Multi-LLM sensemaking system for passive sensing. Closest methodological comparison: also uses LLMs to reason over sensing data. But GLOSS is a sensemaking tool (open-ended queries), not a prediction system with ground truth evaluation. PULSE is validated on prediction accuracy.
- **Nepal et al. (2024).** "MindScape Study: Integrating LLM and Behavioral Sensing for Personalized AI-Driven Journaling Experiences." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 2024. — LLM + behavioral sensing for journaling. Intervention-focused rather than prediction-focused. Complementary work.

**Agent Architecture:**
- **Yao et al. (2023).** "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. — The reasoning-and-acting paradigm that PULSE's autonomous agents instantiate: interleaving reasoning traces with tool calls (sensing queries) to build an investigation strategy.

---

## 4. Key Claims (What to Claim vs. Not Overclaim)

### Can claim (supported by evidence):

1. **Agentic investigation substantially outperforms structured pipelines on identical data.** Auto-Multi (BA 0.660) vs. Struct-Multi (0.603); Auto-Sense (0.589) vs. Struct-Sense (0.516). This is the cleanest result — same data, same LLM, different reasoning approach. The 2x2 factorial isolates the agentic effect.

2. **Multimodal input (diary + sensing) outperforms sensing-only.** Auto-Multi (0.660) vs. Auto-Sense (0.589). Expected but important to quantify.

3. **LLM-based sensing investigation outperforms traditional ML on the same sensing features.** Auto-Sense (0.589) vs. best ML baseline RF (0.518). The LLM extracts more signal from the same behavioral data.

4. **Intervention availability is primarily a behavioral (not textual) signal.** Auto-Sense achieves 0.706 on INT_avail vs. CALLM's 0.542. Sensing outperforms diary for this construct. This is a clinically significant finding — availability is about what you are *doing*, not what you are *writing about*.

5. **Sensing-only prediction is viable for proactive intervention.** Auto-Sense (0.589 mean BA) approaches diary-based CALLM (0.611), and for specific constructs (INT_avail: 0.706 vs. 0.542) surpasses it. This addresses the diary paradox.

6. **The receptivity decomposition (desire AND availability) reveals construct-specific sensing signatures.** Availability is sensing-driven; desire is diary-augmented. Different prediction strategies may be optimal for each component.

### Should NOT overclaim:

1. **Do not claim real-time deployment readiness.** This is retrospective replay on historical data. The temporal information boundary is maintained, but there are no latency, battery, or user experience results.

2. **Do not claim generalizability beyond cancer survivors.** N=50 from a single study (BUCS). The pilot sample is higher-compliance than the full cohort by design. Results need replication in other populations.

3. **Do not claim the agent "understands" emotional states.** The LLM reasons over behavioral patterns; it does not have access to the person's subjective experience. Frame as *prediction from behavioral correlates*, not emotional understanding.

4. **Do not overclaim the ML baseline comparison.** The ML baselines are 5-fold CV on 399 users (not the same 50). This is an apples-to-oranges comparison. Present it as context, not as the primary comparison. The primary comparison should be within the 2x2 factorial on the same 50 users.

5. **Do not claim cost-free inference is a general solution.** The Claude Max subscription approach is a practical choice, not a reproducibility guarantee. API costs could be substantial at scale.

6. **Do not overstate the filtering result.** Auto-Multi+ (0.661) vs. Auto-Multi (0.660) is essentially equivalent. Filtering does not help. Frame this as "the agent is robust to data quality variation" rather than as a positive result.

---

## 5. Narrative Arc

### Problem (2-3 pages)
Open with the JITAI vision: the right intervention, at the right time, to the right person. Then immediately identify the bottleneck: **we cannot deliver timely interventions if we cannot predict when someone is receptive**. Current approaches rely on EMA (burden, missingness, reactive) or simple sensing features (low accuracy). Cancer survivors are a compelling population because (a) emotional well-being is clinically critical for recovery, (b) they have high variability in daily emotional states, and (c) intervention timing matters enormously — a well-timed check-in can prevent a crisis; a poorly timed one adds burden.

Introduce the *diary paradox*: the most informative self-report data is absent when users most need support. This motivates the shift from reactive (diary-dependent) to proactive (sensing-based) prediction.

### Approach (5-7 pages)
Frame the system design through the JITAI lens:
- **Decision points**: Each EMA window (morning, afternoon, evening) — these are the moments when the system must decide whether to intervene.
- **Tailoring variables**: The four prediction targets (PA_State, NA_State, ER_desire, INT_availability) and their conjunction into receptivity.
- **The prediction engine**: PULSE agents with purpose-built MCP sensing tools.

Present the 2x2 factorial (structured vs. autonomous x sensing-only vs. multimodal) as a *systematic investigation of what makes prediction work*. The factorial is not just an evaluation design — it is the contribution. It isolates the value of autonomous investigation vs. structured reasoning, and the value of multimodal vs. sensing-only input.

Detail the agent architecture: per-user memory, cross-user RAG, 8 MCP sensing tools (daily summary, behavioral timeline, hourly query, raw events, baseline comparison, receptivity history, similar days, peer cases). Emphasize that the agent *decides its own investigation strategy* — it is not a fixed pipeline.

### Results (5-7 pages)
Lead with the 2x2 factorial result (the cleanest scientific finding). Then:
1. Agentic >> Structured (the autonomy effect)
2. Multimodal >> Sensing-only (the modality effect)
3. Per-construct analysis revealing that INT_avail is behavioral while ER_desire benefits from diary
4. Comparison to CALLM (proactive vs. reactive)
5. Comparison to ML baselines (LLM reasoning vs. feature engineering)
6. Representativeness analysis (50 vs. 418)

### Discussion (3-4 pages)
- **Implications for JITAI design**: PULSE demonstrates that LLM agents can estimate tailoring variables from passive data with sufficient accuracy to support intervention timing decisions. This opens a path to fully passive JITAIs that do not require user self-report to make timing decisions.
- **The construct-specific sensing story**: Different tailoring variables have different optimal sensing strategies. Intervention availability is best predicted from behavioral data alone. Emotion regulation desire benefits from diary text. A deployed JITAI should use construct-specific prediction pipelines.
- **Limitations and future work**: Retrospective design, single population, no real-time deployment, LLM cost/latency considerations, need for prospective validation with actual intervention delivery.

---

## 6. What to Emphasize vs. De-Emphasize

### Emphasize:

- **The 2x2 factorial design.** This is what makes the paper rigorous. It is not just "we built a system and it works." It is a systematic experimental design that isolates the contributions of autonomous reasoning and multimodal input. Reviewers will respect this.
- **The agentic investigation concept.** The agent's tool-use behavior (choosing which sensing modalities to query, deciding temporal scope, comparing across users) is a genuinely new contribution to the ubicomp sensing pipeline. Show examples of agent investigation traces.
- **INT_avail as a behavioral signal.** The finding that sensing outperforms diary for intervention availability is a clean, novel, clinically relevant result. This should be a highlighted finding.
- **The diary paradox and its resolution.** Frame sensing-only performance (BA 0.589) not as a limitation but as a strength: the system works precisely when diary data is unavailable. This is the proactive advantage.
- **Receptivity decomposition.** Treating receptivity as desire AND availability, and showing that each component has different sensing signatures, is a contribution to the JITAI theory, not just to the system.
- **Per-user analysis and variance.** Show the distribution of per-user BA, not just means. This matters for clinical deployment: does the system work for *most* users, or is it carried by a few high-performers?

### De-Emphasize:

- **Raw accuracy numbers in isolation.** BA of 0.66 sounds modest outside the clinical sensing context. Always contextualize: (a) these are individual-level, momentary predictions from passive data alone, (b) traditional ML achieves ~0.52, (c) even CALLM with diary text achieves 0.61.
- **The filtering variants (Auto-Sense+, Auto-Multi+).** These add minimal value (0.661 vs. 0.660). Mention briefly as evidence that the agent is robust, but do not make it a major result.
- **The ML baseline comparison.** Because it is on different user sets (399 vs. 50), it is weaker evidence. Present it, but make the 2x2 factorial the primary comparison.
- **Cost/infrastructure details.** The Claude Max subscription is a practical convenience, not a contribution. Mention it in a methods subsection, not in the framing.
- **LLM model identity.** The contribution is the *agentic investigation approach*, not the specific LLM. Avoid making this a "Claude is good at prediction" paper. The architecture should be model-agnostic in principle.

---

## 7. Target Reader

### Primary audience: IMWUT/UbiComp researchers working on:
- **Mobile sensing for health/wellbeing** — they care about whether LLM agents can extract more signal from sensing data than traditional ML pipelines. They want to see rigorous evaluation, not just demo systems.
- **JITAI and intervention timing** — they care about whether this moves the needle on predicting when to intervene. They want to see the connection to the JITAI framework, receptivity constructs, and clinical populations.
- **LLM + sensing integration** — a growing subfield. They want to understand the architecture, the tool design, and the comparison to simpler LLM approaches (structured prompting, RAG).

### Secondary audience:
- **Clinical researchers in cancer survivorship** — they care about whether passive sensing can support emotional well-being monitoring without adding burden.
- **AI/NLP researchers interested in agentic tool use** — they care about the MCP tool design and the comparison between structured and autonomous investigation strategies.

### What the primary reader cares about:
1. Does the agentic approach actually work better, or is it just more complex?
2. Is the evaluation rigorous (proper controls, statistical tests, representativeness)?
3. Can I use this approach in my own sensing studies?
4. What are the practical constraints (cost, latency, reliability)?

---

## 8. Potential Reviewer Objections & Preemptive Strategies

### Objection 1: "Retrospective replay is not real-time deployment"
**Severity**: High. IMWUT values in-the-wild deployment.
**Preemption**: Acknowledge this clearly in limitations. Emphasize the strict temporal information boundary (agent never sees future data). Frame the contribution as establishing the *predictive capability* that would feed into a deployed JITAI, not as a deployed system. Cite precedent: many IMWUT papers establish prediction models on historical data before deployment (Kunzler et al. 2019 did exactly this).

### Objection 2: "N=50 is too small and not representative"
**Severity**: High-Medium. The pilot is by design high-compliance.
**Preemption**: (a) Present the representativeness analysis (PA_State, ER_desire, INT_avail base rates show no significant difference). (b) Acknowledge the NA_State difference (p=0.028, small effect). (c) Emphasize that N=50 x ~78 EMA entries/user = ~3,900 prediction instances per version, which is substantial. (d) Frame the 2x2 factorial as within-subject (same 50 users across all conditions), which controls for individual differences.

### Objection 3: "The ML baseline comparison is unfair (different user sets)"
**Severity**: Medium-High.
**Preemption**: Acknowledge this directly. Present it as contextual evidence, not as a primary claim. The primary claim is the 2x2 factorial (same users, same data, different reasoning approach). If possible, run ML baselines on the same 50 users before submission.

### Objection 4: "How do we know the LLM is not just pattern-matching on the prompt format?"
**Severity**: Medium. Structured vs. autonomous comparison partially addresses this (same LLM, different prompting strategy, different results). But reviewers may want ablations.
**Preemption**: (a) Show agent investigation traces demonstrating genuine adaptive behavior (e.g., querying different modalities for different users, changing temporal scope based on data availability). (b) The per-construct analysis (INT_avail vs. ER_desire) shows the agent differentiates between constructs. (c) The cross-user RAG comparison shows the agent uses population-level evidence appropriately.

### Objection 5: "Balanced accuracy of 0.66 is not impressive"
**Severity**: Medium. Especially from ML-focused reviewers.
**Preemption**: Contextualize aggressively. (a) These are individual-level, momentary predictions, not population-level screening. (b) The base rates vary substantially across targets and users. (c) Traditional ML on the same features achieves ~0.52. (d) Even human clinicians struggle with momentary affect prediction from behavioral data. (e) The clinical utility threshold is not perfect prediction — it is "better than random timing" for intervention delivery.

### Objection 6: "LLM inference is too expensive/slow for real-time JITAI deployment"
**Severity**: Medium.
**Preemption**: (a) Acknowledge as a limitation. (b) Note that JITAIs do not require millisecond latency — a prediction every few hours is sufficient for intervention timing. (c) The cost landscape is rapidly changing (smaller models, distillation, local inference). (d) Frame this as a *capability demonstration* — if the agentic approach produces better predictions, the engineering problem of making it fast/cheap is tractable.

### Objection 7: "No comparison to fine-tuned models or specialized architectures"
**Severity**: Medium.
**Preemption**: The contribution is not "LLMs beat everything" but rather "autonomous investigation of heterogeneous behavioral data produces better predictions than structured approaches." A fine-tuned model would need labeled training data per user/construct, which is the very burden PULSE avoids. Zero-shot/few-shot generalization to new users is a feature, not a limitation.

### Objection 8: "The paper conflates two contributions: agentic reasoning AND LLM-based sensing interpretation"
**Severity**: Low-Medium.
**Preemption**: The 2x2 factorial design directly addresses this. Structured-agentic versions use the same LLM for sensing interpretation but without autonomous tool use. The comparison isolates the agentic contribution. Be very explicit about this in the evaluation design section.

---

## Summary of Key Recommendations

1. **Lead with the JITAI tailoring variable problem**, not with "we used LLMs." The contribution is solving a real intervention science problem.
2. **The 2x2 factorial is your strongest methodological asset.** Build the paper around it.
3. **The diary paradox is your most compelling motivating story.** Self-report fails when it is most needed; passive sensing with agentic investigation fills the gap.
4. **INT_avail as a behavioral signal is your cleanest novel finding.** Highlight it.
5. **Be honest about limitations** (retrospective, N=50, ML baseline mismatch) — but frame them correctly. The contribution is the *approach and its evaluation*, not a deployed clinical tool.
6. **Show agent behavior**, not just aggregate numbers. Investigation traces, per-user variation, construct-specific strategies — these are what make the paper rich and convincing.
7. **Connect back to JITAI deployment** in the discussion. What would a deployed system using PULSE look like? What decision rules would it feed? This grounds the work in clinical utility.
