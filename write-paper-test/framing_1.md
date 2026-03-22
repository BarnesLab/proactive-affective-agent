# Paper Framing Proposal — Tanzeem Choudhury Perspective

**Advisor**: Tanzeem Choudhury (Cornell Tech) — ACM Fellow, pioneer in mobile sensing for mental health, founder of HealthRhythms/Dapple, advocate for closing the sensing-to-intervention loop.

**System**: PULSE (Proactive Affective Agent)
**Target venue**: IMWUT (Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)

---

## 1. Title Options (Ranked)

1. **PULSE: Closing the Sensing-to-Prediction Gap with Agentic LLMs for Proactive Affective Computing in Cancer Survivors**
   - Rationale: Directly signals the core contribution (agentic LLMs doing what traditional ML pipelines cannot), names the clinical population, and invokes the well-understood "closing the gap" framing in ubicomp.

2. **From Passive Sensing to Active Reasoning: LLM Agents that Investigate Behavioral Data to Predict Emotional States**
   - Rationale: Emphasizes the paradigm shift — the agent is not a classifier receiving features; it is an investigator deciding what to examine. This is the most technically provocative title.

3. **Agentic Sensing: Autonomous LLM Investigation of Smartphone Behavioral Data for Emotion and Receptivity Prediction**
   - Rationale: Coins the term "agentic sensing" as a category contribution. Risky if reviewers find it premature, but memorable if the paper delivers.

**Recommendation**: Title 1 for safety and clarity at IMWUT. Title 2 as a strong alternative if the authors want to foreground the paradigm shift.

---

## 2. Core Contribution Framing

The story of this paper is:

**Passive smartphone sensing has long promised to bridge the gap between clinic visits and daily life, but the field has been stuck at low-to-moderate prediction accuracy because traditional ML pipelines treat sensing features as fixed-dimensional input vectors, losing the contextual richness of behavioral data. PULSE demonstrates that giving LLM agents the ability to autonomously investigate behavioral data — deciding what sensing streams to examine, how to compare against personal baselines, and when to consult peer cases — unlocks substantially higher prediction accuracy for emotional states and intervention receptivity in cancer survivors (BA=0.66 vs. 0.52 for ML baselines), establishing a new paradigm where the model actively reasons about behavior rather than passively consuming features.**

The secondary story is equally important: **This is the first system to show that agentic reasoning (autonomous tool use) provides a significant, measurable advantage over structured prompting when applied to the same sensing data — the 2x2 factorial design cleanly isolates this effect.**

---

## 3. Positioning Relative to Existing Work

### The Gap This Paper Fills

The ubicomp community has spent a decade building passive sensing pipelines for mental health (sense → extract features → classify). My own work, and the broader field's trajectory, has moved from "can we detect?" to "can we act?" (Adler et al., 2024). But there is a missing middle step: **can we reason?** Traditional ML classifiers and even recent zero-shot LLM approaches treat sensing data as flat feature vectors. PULSE fills the gap between detection and intervention by introducing an intelligent reasoning layer that can investigate, contextualize, and synthesize behavioral evidence autonomously.

### Key Comparison Papers (All Verified)

1. **Wang et al. (2025). "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models." arXiv:2503.10707. [CHI '26 submission].**
   - Authors: Zhiyuan Wang, Katharine E. Daniel, Laura E. Barnes, Philip I. Chow.
   - This is the direct predecessor to PULSE, using diary text + RAG to predict emotional states. PULSE extends CALLM by (a) replacing reactive diary-dependent input with proactive sensing, (b) introducing agentic investigation, and (c) showing that combining both modalities via an autonomous agent (Auto-Multi, BA=0.66) significantly outperforms diary-only (CALLM, BA=0.61).
   - Source: [arXiv](https://arxiv.org/abs/2503.10707)

2. **Zhang et al. (2024). "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features." UbiComp '24 Workshop (Companion Proceedings). ACM.**
   - Authors: Tianyi Zhang, Songyan Teng, Hong Jia, Simon D'Alfonso.
   - The first work to apply LLMs to smartphone sensing for affect prediction. However, it uses zero-shot/few-shot prompting on pre-extracted feature vectors — no agentic investigation, no tool use, no longitudinal memory. PULSE goes substantially beyond by letting the agent decide its investigation strategy.
   - Source: [ACM DL](https://dl.acm.org/doi/10.1145/3675094.3678420)

3. **Feng et al. (2026). "A Comparative Study of Traditional Machine Learning, Deep Learning, and Large Language Models for Mental Health Forecasting using Smartphone Sensing Data." IMWUT.**
   - Authors: Kaidong Feng, Zhu Sun, Roy Ka-Wei Lee, Xun Jiang, Yin-Leng Theng, Yi Ding.
   - The most direct methodological comparison. Their benchmark finds DL (Transformer, F1=0.58) outperforms LLMs on temporal modeling. PULSE's agentic approach (with tool-mediated data access and cross-user RAG) likely addresses the "weaker temporal modeling" limitation they identify in LLMs, because the agent can explicitly query temporal patterns.
   - Source: [arXiv](https://arxiv.org/abs/2601.03603)

4. **Nepal et al. (2024). "MindScape Study: Integrating LLM and Behavioral Sensing for Personalized AI-Driven Journaling Experiences." IMWUT, 8(4).**
   - Authors: Subigya Nepal, Arvind Pillai, William Campbell, et al., Andrew T. Campbell.
   - Uses LLM + behavioral sensing for personalized journaling (intervention side). Complementary to PULSE: MindScape focuses on the intervention delivery; PULSE focuses on the prediction/receptivity assessment that should precede any intervention.
   - Source: [ACM DL](https://dl.acm.org/doi/10.1145/3699761)

5. **Adler et al. (2024). "Beyond Detection: Towards Actionable Sensing Research in Clinical Mental Healthcare." IMWUT, 8(4), Article 160.**
   - Authors: Daniel A. Adler, Yuewen Yang, Thalia Viranda, Xuhai Xu, David C. Mohr, Anna R. Van Meter, Julia C. Tartaglia, Nicholas C. Jacobson, Fei Wang, Deborah Estrin, Tanzeem Choudhury.
   - The conceptual framing paper for the field's next phase. Argues that sensing research must move beyond detection toward clinical action. PULSE is a concrete instantiation of this vision: an agent that reasons about sensing data to determine *when* to intervene.
   - Source: [ACM DL](https://dl.acm.org/doi/10.1145/3699755)

6. **Mishra et al. (2021). "Detecting Receptivity for mHealth Interventions in the Natural Environment." IMWUT, 5(2). [Distinguished Paper Award, UbiComp 2022].**
   - Authors: Varun Mishra, Florian Kunzler, Jan-Niklas Kramer, Elgar Fleisch, Tobias Kowatsch, David Kotz.
   - The gold standard for receptivity prediction using ML on sensor features. Achieved 40% improvement in receptivity detection. PULSE reframes receptivity as Desire AND Availability and uses LLM reasoning rather than handcrafted features.
   - Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8680205/)

7. **Nahum-Shani et al. (2018). "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support." Annals of Behavioral Medicine, 52(6), 446-462.**
   - The conceptual framework for JITAIs. PULSE's receptivity model (Desire AND Availability) directly operationalizes two of the "tailoring variables" from this framework.
   - Source: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5364076/)

8. **Saeb et al. (2015). "Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study." JMIR, 17(7), e175.**
   - Authors: Sohrab Saeb, Mi Zhang, Christopher J. Karr, et al.
   - A foundational study establishing that GPS and phone usage features correlate with depressive symptoms. Represents the "first generation" of sensing-for-mental-health that PULSE builds upon and transcends.
   - Source: [JMIR](https://www.jmir.org/2015/7/e175/)

9. **Wang et al. (2014). "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones." UbiComp '14.**
   - Authors: Rui Wang, Fanglin Chen, Zhenyu Chen, et al., Andrew T. Campbell.
   - The landmark sensing study establishing that passively sensed behavior correlates with mental health outcomes. PULSE represents the evolution from correlation to prediction to reasoning.
   - Source: [ACM DL](https://dl.acm.org/doi/10.1145/2632048.2632054)

---

## 4. Key Claims (What to Claim vs. Not Overclaim)

### Supported Claims

1. **Agentic reasoning significantly outperforms structured prompting on identical data.** Auto-Multi (BA=0.660) >> Struct-Multi (BA=0.603); Auto-Sense (BA=0.589) >> Struct-Sense (BA=0.516). The 2x2 design makes this clean and defensible. This is the paper's strongest claim.

2. **LLM agents outperform traditional ML baselines for affect prediction from sensing data.** Auto-Multi+ (BA=0.661) >> RF (BA=0.518) / XGBoost (BA=0.514). Even sensing-only agentic (BA=0.589) beats all ML baselines.

3. **Multimodal input (diary + sensing) with agentic reasoning achieves the best performance.** Auto-Multi (BA=0.660) >> Auto-Sense (BA=0.589) >> CALLM diary-only (BA=0.611). Diary and sensing are complementary when an agent can reason across both.

4. **Sensing-only prediction is viable and outperforms chance substantially.** Auto-Sense at BA=0.589 across 16 targets, with standout performance on INT_availability (BA=0.706). This is clinically meaningful because sensing does not require user effort.

5. **Intervention availability is more behavioral than emotional.** Auto-Sense (BA=0.706) >> CALLM (BA=0.542) for INT_availability. Sensing captures availability better than self-report. This is a novel, clinically interesting finding.

6. **The "diary paradox"**: diary text is the most informative single modality, but it is absent exactly when users most need support. Sensing provides a fallback that maintains meaningful accuracy.

### Do NOT Overclaim

- **Do not claim this replaces clinical assessment.** BA of 0.66 is promising but not sufficient for autonomous clinical decision-making.
- **Do not claim generalizability beyond cancer survivors.** The population is specific; the method is general, but the results are tied to BUCS.
- **Do not overclaim the 50-user pilot as a full validation.** Be transparent that this is a pilot on 50 high-compliance users from a 418-person dataset. The representativeness analysis shows small differences in NA_State base rates and platform distribution.
- **Do not claim real-time deployment readiness.** This is retrospective replay, not a live study. The temporal information boundary is carefully maintained, but real deployment introduces latency, missing data, and user interaction effects not captured here.
- **Do not claim the agentic approach is cost-efficient.** Each prediction involves multiple LLM calls with tool use. Acknowledge this as a limitation and discuss the cost-accuracy tradeoff.

---

## 5. Narrative Arc

### Problem (2-3 pages)
Open with the clinical reality: cancer survivors experience fluctuating emotional states that are poorly served by periodic clinic visits. Mobile sensing promises continuous monitoring, but a decade of work has shown that traditional ML on extracted sensor features yields only modest prediction accuracy (BA~0.51-0.52). Meanwhile, LLMs have shown promise for text-based affect analysis (CALLM), but diary-based approaches fail when users do not write — precisely when they most need help. **We need a system that can proactively reason about passively collected behavioral data.**

### Approach (5-7 pages)
Introduce PULSE as an agentic sensing system. Three key ideas:
1. **Agent-as-investigator**: Instead of feeding pre-extracted features to a model, give the LLM tools to query behavioral data streams. The agent decides what to examine, forming and testing hypotheses.
2. **2x2 factorial design**: Cleanly isolate the effects of (a) agentic vs. structured reasoning and (b) sensing-only vs. multimodal input.
3. **Cross-user calibration via RAG**: The agent grounds predictions in empirical peer cases, not just LLM intuition.

Include a detailed system architecture section (MCP tools, memory architecture, information boundary constraints).

### Results (5-7 pages)
Present in this order:
1. **Main finding**: Agentic >> Structured (the paradigm shift result)
2. **Modality analysis**: Multimodal >> Sensing-only >> Diary-only baseline patterns
3. **Clinical target deep-dive**: Focus on the 4 key targets (ER_desire, INT_availability, PA_State, NA_State)
4. **ML baseline comparison**: LLM agents >> traditional ML
5. **Per-user analysis**: Show variance, identify when the agent struggles
6. **The diary paradox and INT_availability findings**
7. **Agent behavior analysis**: What does the agent actually do differently in agentic vs. structured mode? (qualitative examples of investigation traces)

### Discussion (3-4 pages)
1. **What agentic reasoning buys you**: The agent's advantage comes from adaptive investigation — it allocates attention where the signal is, rather than processing all features uniformly.
2. **Clinical implications**: Receptivity = Desire AND Availability; sensing captures availability better than self-report.
3. **Toward the sensing-to-intervention loop**: PULSE is the prediction component; future work connects to intervention delivery (cf. MindScape).
4. **Limitations and ethics**: 50-user pilot, retrospective design, cost, potential for LLM hallucination in clinical contexts, equity considerations.

---

## 6. What to Emphasize vs. De-emphasize

### Emphasize

- **The 2x2 factorial result** (agentic vs. structured x sensing vs. multimodal). This is the methodological crown jewel. It is rare in this field to have such a clean ablation.
- **Agent investigation traces**: Show concrete examples of what the agent does differently from the structured pipeline. This is what IMWUT reviewers will find most novel and compelling.
- **INT_availability finding**: Sensing outperforming diary for behavioral availability is a genuinely new insight with direct JITAI implications.
- **The diary paradox**: This motivates the entire research direction and resonates deeply with anyone who has deployed EMA studies.
- **Statistical rigor**: Wilcoxon signed-rank tests, bootstrap CIs, per-user distributions. IMWUT reviewers expect this.

### De-emphasize

- **Absolute accuracy numbers in isolation**: BA=0.66 sounds modest without context. Always present comparatively (vs. ML baselines, vs. CALLM, vs. structured).
- **The filtered variants (Auto-Sense+, Auto-Multi+)**: The marginal improvement suggests the agent already handles noise. Mention briefly but do not make it a main finding. It is actually an interesting negative result (filtering does not help much when the agent can reason about data quality).
- **Cost/infrastructure details**: Mention Claude Max subscription and MCP tooling, but do not dwell on it. The contribution is the paradigm, not the implementation detail.
- **The ML baseline comparison**: Important to include but do not oversell. The ML baselines are on 399 users with 5-fold CV (different setup). Be transparent about this asymmetry; present it as a reference point, not a head-to-head.

---

## 7. Target Reader

### Primary Audience
**IMWUT researchers working on mobile sensing for health** — the core ubicomp community that has been building sensing pipelines for mental health for the past decade. They care about:
- Does this actually improve prediction? (Yes, substantially over ML baselines and structured LLM approaches.)
- Is this reproducible and rigorous? (2x2 design, N=50 users, ~3900 entries/version, statistical tests.)
- What is the practical path from here to deployment?

### Secondary Audience
**HCI/health researchers working on JITAIs and receptivity.** They care about:
- Can we predict receptivity from passive data? (Yes, particularly availability.)
- How does this fit into intervention delivery systems?

### Tertiary Audience
**AI/NLP researchers interested in agentic LLM applications.** They care about:
- Does agentic tool use actually help vs. just giving the LLM all the data? (Yes, this is a clean empirical demonstration.)
- What tool design enables effective agent investigation of sensing data?

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "50 users is too small"
**Preemption**: This is a pilot (state it explicitly). However, 50 users x ~80 EMA entries each = ~3,900 prediction instances per version. Report representativeness analysis showing the 50 users are statistically representative of the full 418-person dataset on 3 of 4 key target distributions (p>0.05), with only NA_State showing a small effect (r=0.19). Acknowledge that these are high-compliance users (by design — you need sufficient data per user for the agent to learn). Frame the 50-user study as establishing the paradigm with a clear path to full-scale evaluation.

### Objection 2: "This is just prompt engineering"
**Preemption**: The 2x2 design directly addresses this. Structured vs. agentic versions use the *same LLM, same data, same targets*. The only difference is whether the agent has tools and autonomy. The consistent advantage of agentic over structured (~6 BA points on average) demonstrates that the tool-use paradigm itself — not just the prompting — contributes meaningfully. Include agent trace analysis showing qualitatively different investigation strategies.

### Objection 3: "The ML baselines are not comparable (different N, different setup)"
**Preemption**: Acknowledge this explicitly. The ML baselines use 5-fold CV on all 399 users; the LLM versions run on 50 users. This is a known limitation. The primary comparison is within the LLM versions (2x2 design), not LLM-vs-ML. The ML baselines serve as a reference point for the general difficulty of the prediction task. Note that ML baseline performance (BA~0.51-0.52) is consistent with the broader literature on sensing-based affect prediction.

### Objection 4: "Retrospective replay is not the same as real-time deployment"
**Preemption**: Describe the information boundary constraints in detail (the agent never sees future data). Acknowledge that real-time deployment introduces additional challenges (missing data, latency, user reactivity). Frame the retrospective design as standard methodology in this field (cf. CALLM, Feng et al.) and as a necessary first step before a prospective study.

### Objection 5: "LLM predictions are not reproducible / non-deterministic"
**Preemption**: Report temperature settings (if applicable). Note that the *pattern* of results (agentic >> structured) is consistent across targets and users, suggesting the findings are robust to LLM stochasticity. Discuss how session memory creates longitudinal consistency within a user. Consider running a subset with multiple seeds if time permits.

### Objection 6: "Cost and scalability"
**Preemption**: Report average number of tool calls per prediction, approximate tokens consumed, and wall-clock time. Acknowledge that the current system is expensive relative to ML baselines. Discuss the cost-accuracy tradeoff and potential for distillation or caching strategies. Note that LLM costs continue to decrease rapidly.

### Objection 7: "Clinical significance — is BA=0.66 actually useful?"
**Preemption**: Contextualize against the state of the art. For many of these targets, prior work (including traditional ML) achieves near-chance performance. BA=0.66 averaged across 16 diverse binary targets, with standout performance on key clinical targets (ER_desire BA=0.75, PA_State BA=0.73, NA_State BA=0.72). These are in the range where clinical utility becomes plausible as a screening/triage tool, not as a standalone diagnostic.

### Objection 8: "Potential for LLM hallucination in clinical reasoning"
**Preemption**: The system makes structured predictions (JSON output with defined fields), not free-text clinical recommendations. The predictions are validated against ground truth EMA responses. Discuss the importance of grounding via RAG (peer cases provide empirical anchors) and the information boundary (the agent can only reason about data it observes, not fabricate data). Recommend human-in-the-loop for any deployment scenario.

---

## Summary

The strongest version of this paper tells a story about **paradigm shift**: from sensing-as-feature-extraction to sensing-as-investigation. The 2x2 factorial design is the methodological backbone. The clinical population (cancer survivors) and the receptivity framing (Desire AND Availability) ground the work in real clinical need. The paper should be positioned as a bridge between the detection-focused past of ubicomp sensing research and the actionable, intervention-ready future — with PULSE demonstrating that LLM agents can be the reasoning layer that makes this transition possible.

---

*Prepared by: Tanzeem Choudhury (simulated advisory perspective)*
*Date: 2026-03-21*
