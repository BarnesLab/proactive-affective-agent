# Paper Framing Advisory — Xuhai "Orson" Xu
## Perspective: Cross-dataset generalization, LLM+sensing systems, computational well-being benchmarks

---

## 1. Title Options (Ranked)

1. **PULSE: Agentic LLM Investigation of Passive Sensing for Proactive Emotional Support in Cancer Survivors**
   - Rationale: Foregrounds the novel mechanism (agentic investigation), the data modality (passive sensing), and the clinical population. "PULSE" is memorable and domain-appropriate.

2. **From Passive Sensing to Proactive Support: LLM Agents that Investigate Behavioral Data for Just-in-Time Emotional Prediction**
   - Rationale: Emphasizes the passive-to-proactive transformation, which is the conceptual leap. More descriptive, less reliant on the system name.

3. **Agentic Reasoning over Mobile Sensing Streams: Predicting Affect and Intervention Receptivity in Cancer Survivors**
   - Rationale: Leads with "agentic reasoning," which is the technical novelty. Crisp for an IMWUT audience that knows sensing but not yet LLM agents.

**My recommendation: Option 1.** It has the right balance of specificity and memorability. IMWUT reviewers will immediately understand what is new (agentic LLM + passive sensing) and what is grounded (cancer survivors, real clinical constructs).

---

## 2. Core Contribution Framing

**The story in 2-3 sentences:**

Existing LLM-for-health systems treat sensor data as static feature vectors fed into prompts — essentially replacing a traditional ML classifier with an LLM while preserving the same fixed-pipeline paradigm. PULSE introduces *agentic sensing investigation*: the LLM autonomously decides which sensing modalities to query, how far back to look, and whether to compare against the user's own baseline or peer population, using purpose-built query tools — mirroring how a clinician interrogates a patient's behavioral history. This agentic approach, evaluated across 50 cancer survivors and ~3,900 EMA entries, yields a mean balanced accuracy of 0.661 on 16 affective and receptivity targets — substantially outperforming both structured LLM pipelines (0.603) and traditional ML baselines (0.518) — demonstrating that *how* an LLM investigates sensor data matters as much as *what* data it sees.

---

## 3. Positioning Relative to Existing Work

### The gap this paper fills

There is a rapidly growing body of work on LLMs for health prediction from sensor data. However, every existing system operates in what I would call "prompt-and-predict" mode: sensor features are pre-extracted, formatted into a textual prompt, and the LLM produces a prediction in a single forward pass. No system lets the LLM *drive the investigation* — choose what data to examine, request comparisons, look at different time windows. This is the gap PULSE fills.

### Key comparison papers (all verified):

**Direct comparisons (LLM + sensor data for health/affect prediction):**

1. **Health-LLM** — Kim, Y., Xu, X., McDuff, D., Breazeal, C., & Park, H.W. "Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data." *Proceedings of the Conference on Health, Inference, and Learning (CHIL)*, 2024. ([PMLR](https://proceedings.mlr.press/v248/kim24b.html))
   - Evaluated 12 LLMs on wearable data across 10 health tasks. Key limitation: single-pass prompting with pre-formatted features. PULSE's contribution is the agentic layer on top.

2. **Cossio et al.** "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features." *Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '24)*. ([ACM DL](https://dl.acm.org/doi/10.1145/3675094.3678420))
   - First work applying LLMs to smartphone sensing for affect prediction. Zero-shot and few-shot only, no tool use or agentic reasoning. PULSE extends this direction dramatically by adding autonomous investigation.

3. **PHIA** — "Transforming Wearable Data into Personal Health Insights using Large Language Model Agents." *Nature Communications*, 2026. ([Nature](https://www.nature.com/articles/s41467-025-67922-y))
   - LLM agent with code generation for wearable data analysis. Closest conceptually to PULSE, but focused on retrospective Q&A (answering user questions about their data) rather than prospective prediction of clinical constructs. PULSE targets real-time JITAI decision-making.

4. **IoT-LLM** — An, X. et al. "IoT-LLM: A Framework for Enhancing Large Language Model Reasoning from Real-World Sensor Data." *arXiv:2410.02429*, 2024. ([arXiv](https://arxiv.org/abs/2410.02429))
   - General framework for LLM + IoT sensor reasoning with chain-of-thought. Evaluated on activity recognition, anomaly detection, etc. Does not address longitudinal personal health prediction or clinical populations.

**Foundational comparisons (mobile sensing + mental health):**

5. **GLOBEM** — Xu, X., Liu, X., Zhang, H., Wang, W., et al. "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 6(4), 2022. Distinguished Paper Award, UbiComp 2023. ([ACM DL](https://dl.acm.org/doi/10.1145/3569485))
   - Established that cross-dataset generalization is the central challenge in behavioral sensing. PULSE sidesteps this by leveraging the LLM's general knowledge + per-user memory rather than training dataset-specific models.

6. **Meegahapola et al.** "Generalization and Personalization of Mobile Sensing-Based Mood Inference Models: An Analysis of College Students in Eight Countries." *IMWUT*, 2023. Distinguished Paper Award, UbiComp 2023.
   - Showed massive variation in mood inference across populations. PULSE's per-user agent with memory is one answer to the personalization challenge this paper raises.

7. **Wang et al.** "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones." *Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)*. ([ACM DL](https://dl.acm.org/doi/10.1145/2632048.2632054))
   - Seminal work on passive sensing for mental health. Established the paradigm that PULSE now extends with LLM agents.

**JITAI and receptivity:**

8. **Nahum-Shani et al.** "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support." *Annals of Behavioral Medicine*, 52(6), 446-462, 2018. ([Oxford Academic](https://academic.oup.com/abm/article/52/6/446/4733473))
   - Foundational JITAI framework. PULSE operationalizes the "decision point" + "tailoring variable" concepts with LLM-based sensing investigation.

9. **Kunzler, F., Mishra, V. et al.** "Exploring the State-of-Receptivity for mHealth Interventions." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 3(4), Article 140, 2019. ([ACM DL](https://dl.acm.org/doi/10.1145/3369805))
   - Defined and modeled receptivity from sensing. PULSE predicts the same construct (desire + availability = receptivity) but using LLM reasoning rather than handcrafted ML features.

**The predecessor system:**

10. **CALLM** — Wang, Z., Daniel, K.E., Barnes, L.E., & Chow, P.I. "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models." *arXiv:2503.10707*, 2025. ([arXiv](https://arxiv.org/abs/2503.10707))
    - PULSE's direct predecessor. Same dataset, same clinical targets. CALLM uses diary text + RAG. PULSE extends to passive sensing + agentic investigation, addressing the fundamental limitation that diary entries are absent when users need support most.

**LLM agent architecture:**

11. **Yao, S. et al.** "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023*. ([arXiv](https://arxiv.org/abs/2210.03629))
    - Foundational agent architecture. PULSE's autonomous mode is conceptually a domain-specialized ReAct agent with sensing-specific tools.

**Penetrative AI:**

12. **Xu, Z. et al.** "Penetrative AI: Making LLMs Comprehend the Physical World." *Proceedings of the 25th International Workshop on Mobile Computing Systems and Applications (HotMobile '24)*, and *Findings of ACL 2024*. ([ACM DL](https://dl.acm.org/doi/10.1145/3638550.3641130))
    - Introduced the concept of LLMs "penetrating" into physical-world sensor data. PULSE is a concrete instantiation of this vision for clinical health sensing.

---

## 4. Key Claims (Supported vs. At-Risk)

### Claims the evidence supports strongly:

1. **Agentic >> Structured (same data)**: Auto-Multi (0.660 BA) significantly outperforms Struct-Multi (0.603 BA). This is the cleanest comparison — same input data, only the reasoning paradigm differs. This is your headline result.

2. **Multimodal >> Sensing-only**: Auto-Multi (0.660) >> Auto-Sense (0.589). Diary text adds substantial signal, confirming CALLM's finding that diary content is informative.

3. **LLM agents >> ML baselines**: Auto-Multi+ (0.661) >> best ML (RF 0.518). The gap is large enough to be meaningful even if ML baselines are on a different user set.

4. **Sensing-only is viable**: Auto-Sense (0.589 BA) substantially outperforms chance and ML baselines. This matters because sensing is always available; diary is not.

5. **Behavioral targets benefit most from sensing**: INT_availability at 0.706 with Auto-Sense vs. 0.542 with CALLM is a striking finding — sensing captures behavioral availability better than diary text.

### Claims to make carefully:

6. **Per-user learning**: The memory mechanism is described but its isolated contribution is not ablated. Mention it as a design component, do not claim "memory improves performance" without an ablation.

7. **Clinical utility**: The retrospective evaluation is rigorous, but real-world JITAI deployment introduces latency, missing data patterns, and user burden considerations not tested here. Frame as "demonstrates feasibility" not "ready for deployment."

### Claims to NOT make:

8. **Do not claim superiority over CALLM on diary-available entries**: CALLM (0.611) is designed for diary-present scenarios. PULSE's advantage is that it works *without* diary, not that it beats CALLM when diary is available (though it does — the comparison is confounded by the agentic architecture).

9. **Do not claim the ML baselines are a fair head-to-head**: They are trained on 399 users (5-fold CV) while PULSE runs on 50 users. Acknowledge this asymmetry explicitly; say you will update with matched baselines.

10. **Do not claim generalizability beyond this population**: 50 cancer survivors with high EMA compliance is a specific sample. The method may generalize, but the evidence does not yet show it.

---

## 5. Narrative Arc

### Problem (1-2 pages)
- Cancer survivors experience volatile emotional states that benefit from just-in-time intervention
- Current approaches require active user input (diaries, EMA responses) — but users disengage precisely when they need help most ("diary paradox")
- Passive sensing captures behavior continuously but traditional ML on sensor features has limited accuracy (GLOBEM shows ~0.52 BA for affect tasks) and does not generalize across populations
- Recent work shows LLMs can interpret sensor data (Health-LLM, Cossio et al.) — but they use static prompt-and-predict, not investigation

### Insight (0.5 pages)
- A clinician does not look at a pre-formatted summary and make a snap judgment. They *investigate*: "How did the patient sleep? Is this unusual for them? What about their activity? Let me compare to similar patients."
- We propose giving LLM agents the same investigative tools and letting them drive the inquiry

### Approach (3-4 pages)
- PULSE architecture: per-user agent with 6 sensing query tools, personal memory, peer comparison
- 2x2 factorial: Structured vs. Autonomous x Sensing-only vs. Multimodal
- This design isolates the contribution of agentic reasoning from the contribution of multimodal data

### Evaluation (5-6 pages)
- 50 cancer survivors, ~3,900 EMA entries per version, 16 binary prediction targets
- 7 system versions + ML baselines
- Wilcoxon signed-rank tests, bootstrap CIs, per-user analysis
- Representativeness analysis (50 vs. 418)

### Results (3-4 pages)
- Lead with the 2x2 factorial: agentic reasoning contributes +0.057 BA (sensing) and +0.057 BA (multimodal)
- Show target-specific findings: behavioral targets (INT_avail) vs. affective targets (ER_desire, PA, NA)
- The "diary paradox" analysis: sensing-only Auto-Sense (0.589) as a fallback when diary is missing

### Discussion (2-3 pages)
- Why agentic investigation helps: tool selection patterns, investigation depth, personalization
- Clinical implications: toward LLM-powered JITAI systems
- Limitations: retrospective, 50 users, cost/latency of LLM inference
- Connection to GLOBEM's generalization challenge: agents may generalize better than trained models because they carry general knowledge

---

## 6. What to Emphasize vs. De-emphasize

### Emphasize:

- **The 2x2 factorial result**: This is methodologically clean and rare in LLM papers. Structured vs. Autonomous is a direct test of whether agentic reasoning matters. Most papers just show "LLM > baseline." You show WHY the LLM wins.
- **Auto-Sense for INT_availability (0.706)**: This is your most clinically actionable finding. Intervention availability is exactly what a JITAI system needs, and sensing predicts it better than diary.
- **The agentic tool design**: The 6 MCP tools are a genuine engineering contribution. Show examples of what the agent investigates — this makes the paper concrete and reproducible.
- **The diary paradox framing**: Sensing-only Auto-Sense (0.589) as the "always-available" mode vs. multimodal Auto-Multi (0.660) when diary exists. This is a practical design point for real JITAI systems.
- **Per-target variation**: Some targets (ER_desire: 0.751, PA_State: 0.733) respond much more to the agentic approach than others. This is rich for discussion.

### De-emphasize:

- **Absolute BA numbers in isolation**: 0.661 mean BA sounds modest. Always present it relative to baselines (0.518 ML, 0.516 Struct-Sense). The LIFT is the story, not the absolute number.
- **Filtering variants (v5/v6)**: Auto-Multi+ (0.661) vs Auto-Multi (0.660) is essentially null. Mention it briefly as evidence that agent reasoning is robust to noise, then move on. Do not make it a major finding.
- **Continuous regression metrics (MAE)**: The project digest mentions these are de-prioritized. Keep the focus on binary classification (BA, F1) which maps to clinical decision-making.
- **The 16 individual emotion targets**: Lead with the 4 clinical focus targets (ER_desire, INT_avail, PA_State, NA_State). Relegate the other 12 to a supplementary table.

---

## 7. Target Reader

### Primary audience: IMWUT researchers working on mobile sensing + health

These readers care about:
- Can this actually work with real sensor data (missing data, noise, device heterogeneity)?
- Is the evaluation rigorous (proper baselines, statistical tests, clinical grounding)?
- Is the system practical (cost, latency, deployability)?
- Does it advance the field's understanding, not just add another system?

### Secondary audience: HCI/health researchers interested in LLM agents

These readers care about:
- What does "agentic" mean concretely? (Show the tool calls, the investigation traces)
- Is this just prompt engineering dressed up as a system paper?
- How does this compare to simpler LLM approaches?

### Tertiary audience: Clinical researchers in cancer survivorship / JITAI

These readers care about:
- Are the prediction targets clinically meaningful?
- Could this actually be deployed in a real intervention study?
- What are the ethical implications of LLM-based emotional prediction?

### Write for the primary audience. The 2x2 factorial design, rigorous baselines, and clinical grounding will satisfy IMWUT reviewers. The agentic architecture and tool design will attract the LLM-agent crowd.

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "N=50 is too small"
**Preemption**: 50 users x ~80 EMA entries each = ~3,900 data points per system version. Show the representativeness analysis (base rates match full cohort on 3/4 targets, p > 0.05). Acknowledge the selection bias toward high-compliance users and discuss what this means for generalization. Note that this is comparable to or larger than evaluation sets in Health-LLM and Cossio et al.

### Objection 2: "ML baselines are unfairly weak / not comparable"
**Preemption**: Acknowledge the asymmetry (399 users for ML vs 50 for LLM) explicitly. The ML baselines use the same feature set available to the LLM. Frame ML as a reference point, not a head-to-head. If possible, re-run ML on the same 50 users before submission.

### Objection 3: "This is just prompt engineering, not a system contribution"
**Preemption**: The 2x2 factorial directly refutes this. Structured (V1/V3) IS prompt engineering — and it performs significantly worse than autonomous (V2/V4) which uses tool-based investigation. The contribution is the investigation paradigm, not the prompt.

### Objection 4: "Retrospective simulation is not real deployment"
**Preemption**: Describe the strict temporal boundary (agent only sees past data). Acknowledge that real deployment introduces additional challenges (latency, user burden, missing data in real-time). Frame this as a necessary precursor study — you cannot ethically deploy an untested JITAI system.

### Objection 5: "LLM inference is too expensive/slow for real-time JITAI"
**Preemption**: Report actual cost and latency per prediction. Note that the agentic mode makes multiple tool calls but LLM costs are falling rapidly. Discuss practical deployment architectures (batch prediction at decision points, not continuous monitoring).

### Objection 6: "No ablation of individual components (memory, RAG, tools)"
**Preemption**: The 2x2 design ablates two major axes (structured vs. agentic, sensing vs. multimodal). Acknowledge that finer-grained ablations (memory alone, RAG alone, individual tools) are future work. The current factorial already isolates the most important factor (agentic reasoning).

### Objection 7: "Balanced accuracy of 0.66 is not clinically useful"
**Preemption**: Frame in context: (a) this is a 16-target MEAN; individual targets reach 0.75 (ER_desire); (b) for passive, always-available prediction, even modest accuracy enables useful triage (surface likely-receptive moments for intervention); (c) prior ML approaches on similar tasks achieve ~0.52. The clinical bar is not perfect prediction but better-than-random triage.

### Objection 8: "Reproducibility — LLM outputs are non-deterministic"
**Preemption**: Report temperature settings. Show per-user variance analysis. The structured vs. autonomous comparison itself demonstrates that the variance in agentic reasoning is beneficial, not harmful. Commit to releasing all prediction outputs for reproducibility even if the LLM inference cannot be exactly replicated.

---

## Summary of Positioning Strategy

This paper sits at the intersection of three active research threads:

1. **LLMs for health sensing** (Health-LLM, Cossio et al., IoT-LLM) — PULSE adds agentic investigation
2. **Mobile sensing for mental health** (GLOBEM, StudentLife, Meegahapola et al.) — PULSE replaces trained ML with LLM reasoning
3. **JITAI and receptivity** (Nahum-Shani, Kunzler/Mishra) — PULSE predicts the constructs these systems need

The unique contribution is at the intersection: **an LLM agent that autonomously investigates behavioral sensing data to predict clinical affect and intervention receptivity**. No prior system does all three: agentic tool use + passive sensing + clinical prediction targets.

Frame PULSE not as "another LLM-for-health paper" but as **the first system where the LLM drives the investigation of behavioral data**, and the 2x2 factorial proves this investigation paradigm is what makes the difference.

---

*Advisory prepared from the perspective of Xuhai "Orson" Xu, simulated for the purpose of paper framing. All referenced papers have been web-search verified.*
