# Framing Proposal: PULSE Paper for IMWUT

**Advisor**: Yubin Kim (MIT Media Lab) — Lead author, Health-LLM (PMLR 2024)
**Date**: 2026-03-21

---

## 1. Title Options (Ranked)

1. **PULSE: LLM Agents as Behavioral Investigators for Proactive Affect Prediction from Passive Smartphone Sensing**
   - Rationale: Foregrounds the key novelty (agentic investigation via tools), the application domain (affect prediction), and the data modality (passive sensing). "Behavioral investigators" conveys the agent's autonomous data-exploration role without requiring jargon.

2. **From Passive Sensing to Proactive Care: Agentic LLM Reasoning over Smartphone Data for Cancer Survivors' Emotional States**
   - Rationale: Emphasizes the clinical motivation and the passive-to-proactive narrative. Cancer survivors grounds the work in a meaningful health context.

3. **Agentic Sensing: How LLM Agents with Tool Access Outperform Structured Pipelines for Affective State Prediction**
   - Rationale: Most technically direct. Highlights the 2x2 factorial finding (agentic >> structured). Less evocative but sharply scoped.

**My recommendation**: Title 1. It is specific enough for IMWUT reviewers to immediately grasp the contribution (LLM agents + tool-based sensing investigation + affect prediction), memorable, and accurate.

---

## 2. Core Contribution Framing

**THE story of this paper (2-3 sentences):**

Existing approaches to LLM-based health prediction feed pre-formatted feature summaries to models in a single prompt-response turn, treating the LLM as a pattern matcher rather than an investigator. PULSE introduces a fundamentally different paradigm: LLM agents equipped with sensing query tools that autonomously decide what behavioral data to examine, how far back to look, and which cross-user comparisons to make — conducting a structured investigation for each prediction. A rigorous 2x2 factorial evaluation on 50 cancer survivors (~3,900 entries per condition) demonstrates that this agentic investigation paradigm yields 0.660 mean balanced accuracy across 16 binary affective targets, substantially outperforming structured single-call LLM pipelines (0.603), traditional ML baselines (0.518), and a state-of-the-art diary-based LLM system (0.611).

---

## 3. Positioning

### 3.1 The Gap This Paper Fills

There is a growing body of work applying LLMs to health prediction from sensor data, but a critical gap persists: **all existing approaches treat LLMs as passive processors of pre-curated feature summaries.** No prior work gives LLMs autonomous tool access to query, explore, and investigate raw behavioral data streams. This paper fills that gap by introducing the *agentic sensing investigation* paradigm, where the LLM actively drives the data exploration process — analogous to how a clinician conducts an assessment by selectively gathering information rather than passively reading a chart.

### 3.2 Key Comparison Papers

**Direct comparisons (same problem space — LLMs + sensor data for health/affect):**

- **Health-LLM** (Kim, Xu, McDuff, Breazeal, Park; PMLR 2024 — Proceedings of the Conference on Health, Inference, and Learning). Evaluated 12 LLMs on 10 health prediction tasks from wearable sensor data; created HealthAlpaca fine-tuned model. Key difference from PULSE: Health-LLM feeds pre-formatted features to LLMs in a single prompt — no tool use, no agentic investigation. PULSE shows that giving the LLM agency over data exploration dramatically outperforms this passive paradigm.
  *Verified: [PMLR proceedings](https://proceedings.mlr.press/v248/kim24b.html)*

- **Mental-LLM** (Xu, Yao, et al.; IMWUT 2024, Vol 8, Issue 1). Evaluates LLMs for mental health prediction from online text data, including zero-shot, few-shot, and fine-tuning. Focuses on text (Reddit/social media), not passive sensing. PULSE differs by using passive sensor data as the primary modality and employing agentic tool-use rather than prompt engineering.
  *Verified: [ACM DL](https://dl.acm.org/doi/abs/10.1145/3643540)*

- **Malgaroli et al., "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features"** (UbiComp Companion 2024). Most directly comparable: uses LLMs (Gemini 1.5 Pro) with smartphone sensing data for I-PANAS-SF prediction. But it is a small-scale workshop paper (10 students), uses zero/few-shot prompting with pre-aggregated weekly summaries, and achieves modest results. PULSE scales to 50 cancer survivors with ~3,900 entries per condition and introduces tool-use agentic investigation rather than static prompting.
  *Verified: [ACM DL](https://dl.acm.org/doi/10.1145/3675094.3678420)*

- **GLOSS** (Choube, Le, Li, Ji, Das Swain, Mishra; IMWUT 2025, Vol 9, Issue 3). Uses a group of LLMs for open-ended sensemaking of passive sensing data. Closest in spirit to PULSE's tool-based approach but focuses on answering natural language queries about behavior, not on making closed-form predictions. PULSE's contribution is specifically about prediction with clinical targets and factorial evaluation of agentic vs. structured paradigms.
  *Verified: [ACM DL](https://dl.acm.org/doi/10.1145/3749474)*

- **CALLM** (Wang et al.; arXiv 2025, under review). The direct predecessor to this work — uses LLMs with diary text and RAG for cancer survivors' emotion prediction. PULSE extends CALLM by adding passive sensing, agentic tool use, and the 2x2 factorial design. CALLM serves as one of the baseline conditions within PULSE's evaluation.
  *Verified: [arXiv 2503.10707](https://arxiv.org/abs/2503.10707)*

**Foundational references (sensing + affect prediction):**

- **StudentLife** (Wang, Chen, et al.; UbiComp 2014, 10-Year Impact Award 2024). Foundational mobile sensing study linking smartphone data to mental health in college students. Establishes that passive sensing can predict affective states — PULSE builds on this with LLM-based reasoning.
  *Verified: [ACM DL](https://dl.acm.org/doi/10.1145/2632048.2632054)*

- **GLOBEM** (Wang et al.; IMWUT 2022/2023, Distinguished Paper Award). Cross-dataset generalization benchmark for longitudinal behavior modeling from mobile sensing. Demonstrates that ML models struggle with generalization across populations — PULSE's LLM approach sidesteps feature engineering entirely.
  *Verified: [ACM DL](https://dl.acm.org/doi/10.1145/3569485)*

- **Meegahapola et al., "Generalization and Personalization of Mobile Sensing-Based Mood Inference Models"** (IMWUT 2022, Vol 6, Issue 4; Distinguished Paper Award at UbiComp 2023). Analyzes mood inference generalization across 8 countries. Shows personalization matters — PULSE's per-user memory and cross-user RAG address this via LLM-based personalization.
  *Verified: [ACM DL](https://dl.acm.org/doi/10.1145/3569483)*

**Foundational references (receptivity + JITAI):**

- **Nahum-Shani et al., "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health"** (Annals of Behavioral Medicine, 2018). Defines the JITAI framework. PULSE predicts receptivity (desire AND availability), which is a core JITAI decision variable.
  *Verified: [Oxford Academic](https://academic.oup.com/abm/article/52/6/446/4733473)*

- **Mishra et al., "Exploring the State-of-Receptivity for mHealth Interventions"** (IMWUT 2019, Vol 3, Issue 4). Introduces receptivity detection via mobile sensing with ML models. PULSE replaces these ML models with LLM agents that can reason about context.
  *Verified: [ACM DL](https://dl.acm.org/doi/10.1145/3369805)*

- **Mishra et al., "Detecting Receptivity for mHealth Interventions in the Natural Environment"** (IMWUT 2021, Vol 5, Issue 2; Distinguished Paper Award). Deploys receptivity prediction in the wild. PULSE's intervention availability target (INT_avail) directly maps to this work's definition of receptivity.
  *Verified: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8680205/)*

**Foundational references (LLM agents + tool use):**

- **ReAct** (Yao et al.; ICLR 2023). Introduces the synergy of reasoning and acting in LLMs — the foundational paradigm that PULSE's agentic investigation builds upon. PULSE applies ReAct-style interleaved reasoning + tool-use to a new domain (behavioral sensing data).
  *Verified: [arXiv](https://arxiv.org/abs/2210.03629)*

---

## 4. Key Claims

### What the paper CAN claim (supported by evidence):

1. **Agentic investigation >> structured prompting for sensing data.** The 2x2 factorial cleanly isolates this: Auto-Multi (0.660) >> Struct-Multi (0.603); Auto-Sense (0.589) >> Struct-Sense (0.516). This is the paper's strongest and most novel claim. The agent's ability to adaptively choose which data to examine, how to compare to baselines, and which peer cases to retrieve yields a consistent advantage over fixed pipelines given the same underlying data.

2. **Multimodal >> sensing-only, but sensing alone is viable.** Auto-Multi (0.660) >> Auto-Sense (0.589), confirming diary text adds significant value. But Auto-Sense (0.589) substantially outperforms Struct-Sense (0.516) and ML baselines (0.518), showing that agentic reasoning can extract meaningful signal from passive sensing alone — important because diary entries are often missing when users need help most.

3. **LLM agents >> traditional ML baselines on the same sensing features.** Auto-Multi+ (0.661) vs. best ML (RF 0.518). The LLM's contextual reasoning, longitudinal memory, and cross-user calibration provide substantial lift over feature-engineered models.

4. **Intervention availability is behaviorally detectable.** Auto-Sense achieves 0.706 BA on INT_avail, outperforming even CALLM's diary-based 0.542. This suggests that behavioral patterns (screen use, activity, location) are more informative than self-reported diary text for determining when someone is available for an intervention — a clinically actionable finding for JITAI design.

5. **Cross-user RAG provides empirical calibration.** The peer case retrieval mechanism grounds predictions in population-level evidence rather than relying solely on LLM world knowledge, addressing a key concern about LLM reliability in clinical contexts.

### What the paper should NOT overclaim:

1. **Do not claim real-time deployment readiness.** This is a retrospective replay study, not a live deployment. The agent never actually intervened. Frame as "simulation study" or "retrospective evaluation" throughout.

2. **Do not claim generalizability beyond this population.** N=50 cancer survivors from one study (BUCS). The representativeness analysis shows the 50 are largely representative of the 418, but this is still a single cohort.

3. **Do not claim the agentic approach is cost-efficient at scale.** Each prediction involves multiple LLM calls with tool use. Acknowledge compute requirements (though the paper cleverly uses Claude Max subscription for zero marginal cost).

4. **Do not overclaim on ML baseline comparison.** ML baselines were run on all 399 users with 5-fold CV, while LLM experiments used 50 users. The comparison is directionally informative but not perfectly controlled. Be transparent about this.

5. **Do not claim the filtering variants add meaningful value.** Auto-Multi+ (0.661) vs. Auto-Multi (0.660) is negligible. Present this as evidence that the agentic approach is robust to data noise, not as a benefit of filtering.

---

## 5. Narrative Arc

### Problem (Sections 1-2)
Cancer survivors experience fluctuating emotional states that benefit from just-in-time adaptive interventions, but current approaches face a fundamental paradox: the most informative signal (diary text) is least available when users need help most. Passive smartphone sensing is always-on but historically underperforms self-report for affect prediction. Recent work has shown LLMs can reason about health data, but existing approaches treat LLMs as passive feature processors — feeding pre-formatted summaries for single-turn predictions. This misses the core strength of modern LLMs: their ability to autonomously investigate, reason across evidence, and synthesize from multiple data sources.

### Approach (Sections 3-4)
PULSE introduces *agentic sensing investigation*: LLM agents equipped with 8 purpose-built sensing query tools (daily summary, behavioral timeline, targeted hourly query, raw events, baseline comparison, receptivity history, similar days, peer cases) that autonomously investigate behavioral data streams for each prediction. The agent decides what to look at, how far back to examine, and which cross-user comparisons to make. A 2x2 factorial design (structured vs. autonomous x sensing-only vs. multimodal) isolates the effect of agentic reasoning from data modality. Per-user longitudinal memory enables personalized reasoning, and cross-user RAG provides empirical calibration anchors.

### Results (Sections 5-6)
The factorial evaluation on 50 cancer survivors (~3,900 entries per condition) reveals three key findings: (1) agentic investigation consistently outperforms structured pipelines regardless of data modality, (2) multimodal data with agentic reasoning achieves the best performance (0.660 BA), substantially outperforming both diary-only LLM baselines and traditional ML, and (3) passive sensing alone with agentic investigation (0.589 BA) is clinically viable and outperforms ML baselines that use the same features. Per-target analysis reveals that intervention availability is best predicted by behavioral data, not diary text — a finding with direct implications for JITAI design.

### Impact (Section 7)
PULSE demonstrates that the paradigm of *how* LLMs interact with sensor data matters more than the data itself. The shift from passive processing to active investigation unlocks performance gains that neither better features nor better prompts alone can achieve. This has implications beyond cancer survivorship: any domain where contextual behavioral data informs health decisions — from mental health monitoring to chronic disease management — can benefit from agentic sensing investigation. We release the full system architecture and evaluation framework to support replication and extension.

---

## 6. What to Emphasize vs. De-emphasize

### EMPHASIZE:

- **The 2x2 factorial design.** This is methodologically rigorous and rare in LLM health papers. It cleanly isolates agentic benefit from data modality benefit. Make this a centerpiece.
- **Agentic vs. structured gap.** The ~6 percentage point BA improvement from Auto-Multi over Struct-Multi (same data, different reasoning paradigm) is the most novel finding. Show example agent traces demonstrating different investigation strategies.
- **INT_avail from sensing alone (0.706).** This is a striking result with clear JITAI implications. Sensing outperforms diary for availability prediction — this is counterintuitive and actionable.
- **The "diary paradox."** Frame the finding that diary is most informative but least available as motivation for the sensing-first approach. This resonates with IMWUT reviewers who understand real-world deployment constraints.
- **Qualitative agent investigation traces.** Show how the agent reasons through a case — what tools it calls, what evidence it weighs, how it synthesizes. This makes the contribution visceral and distinguishes it from black-box ML.
- **Per-target analysis.** Some targets (emotion regulation desire, positive/negative affect) show large agentic advantages. Others may not. Honest per-target reporting strengthens credibility.

### DE-EMPHASIZE:

- **Filtering variants (V5/V6).** Auto-Multi+ (0.661) vs. Auto-Multi (0.660) shows filtering adds nothing meaningful. Mention briefly as evidence of robustness, not as a contribution.
- **Absolute BA numbers in isolation.** 0.660 BA sounds modest out of context. Always present relative to baselines and emphasize the difficulty of the task (predicting affect from behavioral proxies, 16 heterogeneous binary targets, real-world noisy data).
- **ML baseline comparison details.** Since ML baselines are on 399 users (not the same 50), treat them as a reference point rather than a primary comparison. The LLM-vs-LLM comparisons (agentic vs. structured, multimodal vs. sensing-only) are more controlled.
- **Cost/compute analysis.** The Claude Max subscription approach is pragmatic but not generalizable. Mention it in the method but don't dwell on it. Focus on the paradigm, not the engineering.
- **CALLM as a competitor.** Position CALLM as a respected baseline that PULSE extends, not something to "beat." The real contribution is not outperforming CALLM but demonstrating the agentic paradigm's value.

---

## 7. Target Reader

### Primary audience: IMWUT/UbiComp researchers in mobile sensing and digital health

These readers care about:
- **Does passive sensing actually work for affect prediction?** (Yes, with agentic LLM reasoning.)
- **What is the practical path from sensing data to intervention decisions?** (LLM agents that investigate data, not just classify features.)
- **How does this compare to the ML models I already know?** (Substantially better, and the comparison is fair.)
- **Can I replicate or extend this?** (Open architecture, clear factorial design.)

### Secondary audience: LLM/AI researchers interested in agentic tool use

These readers care about:
- **Does agentic tool use actually help in a real application?** (Yes, 2x2 factorial proves it.)
- **What makes the tools useful?** (Designed for behavioral data investigation, not generic.)
- **Is the agentic advantage consistent or task-dependent?** (Consistent across modality conditions.)

### Tertiary audience: Clinical/behavioral health researchers

These readers care about:
- **Is this clinically meaningful?** (Predicting receptivity = desire AND availability, directly actionable for JITAIs.)
- **Does it work for my population?** (Cancer survivors, but the paradigm generalizes.)
- **When is passive sensing good enough?** (For availability, it is better than diary.)

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "N=50 is too small for IMWUT."
**Preemption**: 50 users x ~80 EMA entries each = ~3,900 data points per condition. The analysis is at the entry level with user-level aggregation for statistical tests. The representativeness analysis (Section X) shows these 50 are statistically representative of the full 418-person BUCS cohort on all key target base rates (p > 0.05 for 3 of 4 focus targets). Per-user BA distributions and Wilcoxon signed-rank tests with bootstrap CIs provide rigorous statistical backing. Additionally, each of the 7 conditions is run independently on all 50 users — this is a within-subjects design, which has far greater statistical power than between-subjects. Acknowledge the limitation but contextualize: this is the largest LLM-based affect prediction study from passive sensing to date (Malgaroli et al. used 10 students).

### Objection 2: "ML baselines are unfairly compared (different N, different splits)."
**Preemption**: Be transparent upfront. State clearly that ML baselines are computed on the full 399-user cohort with 5-fold CV, while LLM experiments use 50 users. The ML numbers serve as a reference point for the difficulty of the task, not as a controlled head-to-head comparison. The primary contribution is the LLM-vs-LLM factorial comparison, which is perfectly controlled.

### Objection 3: "This is a retrospective simulation, not a real deployment."
**Preemption**: Acknowledge explicitly and frame it positively. Retrospective replay with strict information boundaries (agent only sees data before the prediction timestamp) is the standard evaluation paradigm in JITAI prediction research (cite Mishra et al. 2021, Nahum-Shani et al. 2018). The simulation enables the 2x2 factorial that would be impractical in a live study (running 7 parallel conditions per user). Position the next step as a micro-randomized trial.

### Objection 4: "The LLM is a black box — how do we know it is reasoning correctly about sensing data?"
**Preemption**: Include an extensive qualitative analysis section with agent investigation traces. Show concrete examples of the agent's reasoning chains: which tools it calls, what patterns it identifies, how it weighs conflicting evidence. The structured vs. agentic comparison itself provides interpretability — when giving the agent freedom helps, it suggests the LLM is identifying useful investigation strategies. Also, the cross-user RAG mechanism grounds predictions in empirical evidence, not just LLM "intuition."

### Objection 5: "Balanced accuracy of 0.66 is not that impressive."
**Preemption**: Context is everything. (a) This is predicting momentary affective states from behavioral proxies, which is inherently noisy — 0.66 BA across 16 heterogeneous binary targets is strong. (b) For comparison, Health-LLM (Kim et al., PMLR 2024) reports similar ranges on simpler tasks with richer physiological data (heart rate, etc.). (c) The individual focus targets show much higher BA (e.g., ER_desire at 0.751, INT_avail at 0.716, PA_State at 0.733) — the mean is pulled down by inherently harder targets like interaction quality and pain. (d) The consistent improvement over well-controlled baselines is more informative than the absolute number.

### Objection 6: "Why not fine-tune an LLM on this data instead of using tool-based agentic approach?"
**Preemption**: Fine-tuning requires labeled training data in the target format, loses the LLM's ability to reason about novel patterns, and doesn't scale to new populations without retraining. The agentic approach is zero-shot by design — it leverages the LLM's general reasoning capability and grounds it in domain-specific data through tools and RAG. This makes it immediately applicable to new populations and sensing setups without retraining. Health-LLM showed that fine-tuning (HealthAlpaca) can match larger models but remains domain-locked. PULSE's agentic paradigm is fundamentally more flexible.

### Objection 7: "Is the improvement from agentic reasoning, or from making more LLM calls?"
**Preemption**: Good question. The structured pipelines (V1, V3) use a single LLM call with all data provided upfront. The agentic agents (V2, V4) use multiple calls through tool use. However, the structured prompts already contain all the information the agent could discover through tools — the data is the same. The advantage comes from the agent's ability to selectively focus, compare, and synthesize, not from seeing more data. The filtering variants (V5/V6) further show that pre-curating data does not match the agent's own curation. Include a tool-call analysis showing average number of calls and which tools contribute most.

### Objection 8: "How reproducible is this? LLM outputs are stochastic."
**Preemption**: Report on variance across the evaluation. The within-subjects design (same users, same data, different system versions) controls for user-level variance. The ~3,900 entries per condition provide enough volume to estimate per-user and per-target distributions reliably. Wilcoxon signed-rank tests confirm statistical significance. Acknowledge that individual predictions will vary across runs, but aggregate metrics are stable. This is analogous to how clinical assessments by different practitioners vary case-by-case but converge in aggregate accuracy.

---

## Summary

The strongest framing for this paper is: **PULSE introduces the agentic sensing investigation paradigm for health prediction, demonstrating through a rigorous 2x2 factorial design that giving LLM agents autonomous tool access to behavioral data yields consistent and substantial improvements over both structured LLM pipelines and traditional ML approaches.** The key intellectual contribution is not "another LLM for health prediction" but rather the insight that *how* the LLM interacts with sensor data (investigating vs. processing) matters as much as *what* data it sees. This is a paradigm paper, not just an empirical one.
