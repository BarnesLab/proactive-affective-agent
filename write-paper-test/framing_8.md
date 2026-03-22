# PULSE Paper Framing — Advisory Notes

**Advisor**: Laura Barnes (University of Virginia)
**Expertise**: Mobile sensing for health, affective computing, social anxiety detection via passive sensing, ML for chronic diseases
**Target venue**: IMWUT (Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)

---

## 1. Title (Ranked)

1. **PULSE: Proactive Affective Agents for Cancer Survivors via Autonomous Sensing Investigation**
   - Best option. "Proactive" signals the key advance over reactive diary approaches. "Autonomous Sensing Investigation" captures the agentic novelty. "Cancer Survivors" anchors the clinical domain.

2. **From Diaries to Sensors: LLM Agents That Autonomously Investigate Behavioral Data to Predict Emotional States in Cancer Survivors**
   - More descriptive; spells out the diary-to-sensing paradigm shift. Risk: slightly long for IMWUT norms.

3. **Agentic LLMs for Passive Sensing: Predicting Affect and Intervention Receptivity Without Self-Report**
   - Emphasizes the "without self-report" angle, which is the clinical motivation. Broadest framing; less tied to the cancer domain.

---

## 2. Core Contribution Framing

**The story in 2-3 sentences:**

Current LLM-based approaches to mental health prediction rely on active user input (diaries, social media posts), but the people who need support most are often the ones least likely to provide it. PULSE reframes the problem: instead of waiting for self-report, it deploys LLM agents that autonomously investigate passive smartphone sensing data — deciding which behavioral streams to examine, how far back to look, and which peer cases to consult — to predict emotional states and intervention receptivity in cancer survivors. A controlled 2x2 factorial evaluation (structured vs. autonomous x sensing-only vs. multimodal) on 50 cancer survivors demonstrates that agentic investigation of sensing data (BA=0.660) substantially outperforms both structured LLM pipelines (BA=0.603) and traditional ML baselines (BA=0.518), while approaching the performance ceiling set by diary-based methods — without requiring any active user engagement.

---

## 3. Positioning

### The gap this paper fills

There is a rapidly growing body of work applying LLMs to mental health data, but it falls into two silos:

- **Text-based LLM approaches** (social media, diaries, clinical notes) that require active user disclosure
- **Sensor-based ML approaches** (passive sensing features fed to classifiers) that lose the contextual reasoning LLMs provide

No prior work gives an LLM agent autonomous tool-based access to raw behavioral sensing streams and lets it decide its own investigation strategy. PULSE bridges these two silos: it brings the reasoning power of LLMs to passive sensing data, through an agentic architecture where the model actively queries and cross-references behavioral data rather than passively consuming pre-formatted feature vectors.

### Key comparison papers (all verified via web search)

**Direct predecessors (our own lineage):**

- **CALLM** — Wang, Z., Daniel, K.E., Barnes, L.E., & Chow, P.I. (2025). "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models." arXiv:2503.10707. CHI 2026 (under review/accepted). *Same dataset (BUCS, N=407 cancer survivors), same prediction targets. CALLM is the reactive diary-based predecessor that PULSE aims to surpass proactively.*

**LLMs for mental health prediction:**

- **Mental-LLM** — Xu, X., Yao, B., et al. (2024). "Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 8(1). *Key IMWUT precedent for LLM-based mental health prediction, but uses online text (Reddit/social media), not sensing. Shows instruction-finetuned LLMs can outperform much larger zero-shot models.*

- **LENS** — (2025). "LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models." arXiv:2512.23025. *Trains a patch-level encoder to project raw sensor signals into LLM representation space. Different approach: model fine-tuning vs. our tool-based agentic investigation. Complementary.*

- **GLOSS** — (2025). "GLOSS: Group of LLMs for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*. Presented at UbiComp/ISWC 2025. *Multi-agent LLM system for passive sensing sensemaking using code generation. Closest concurrent work in spirit, but focuses on open-ended interpretation rather than clinical prediction with ground truth evaluation.*

**LLMs + smartphone sensing for affect:**

- **Zhang, T., et al.** (2024). "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features." *Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '24)*. *First to use LLMs for affect prediction from smartphone sensing (10 students, zero/few-shot). Our work goes far beyond: agentic tool use, 50 users, longitudinal memory, clinical population, 2x2 factorial design.*

- **Beukenhorst, A., et al.** (2026). "A Comparative Study of Traditional Machine Learning, Deep Learning, and Large Language Models for Mental Health Forecasting using Smartphone Sensing Data." arXiv:2601.03603. *Head-to-head comparison of ML/DL/LLM on smartphone sensing for mental health forecasting. Finds DL (Transformer) best overall, LLMs strong in contextual reasoning but weak in temporal modeling. Our agentic approach addresses the temporal weakness through tool-based longitudinal investigation.*

**Passive sensing foundations:**

- **Wang, R., et al.** (2014). "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones." *Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)*. *Foundational work establishing smartphone sensing as a valid approach for mental health assessment.*

- **Saeb, S., et al.** (2015). "Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study." *Journal of Medical Internet Research*, 17(7), e175. *Established GPS and phone usage features as correlates of depression severity.*

- **Meegahapola, L., et al.** (2022). "Generalization and Personalization of Mobile Sensing-Based Mood Inference Models: An Analysis of College Students in Eight Countries." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 6(4). IMWUT Distinguished Paper Award. *Highlights generalization challenges and the value of personalization in mobile sensing mood models — directly relevant to our per-user agent memory design.*

**Receptivity and JITAI:**

- **Nahum-Shani, I., et al.** (2018). "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support." *Annals of Behavioral Medicine*, 52(6), 446-462. *Foundational framework defining JITAIs, including the importance of tailoring variables like receptivity. PULSE operationalizes the receptivity component (desire AND availability).*

- **Kunzler, F., Mishra, V., et al.** (2019). "Exploring the State-of-Receptivity for mHealth Interventions." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 3(4), Article 140. *Defines and measures receptivity for mHealth. We adopt and extend their conceptualization (desire + availability).*

- **Mishra, V., et al.** (2021). "Detecting Receptivity for mHealth Interventions in the Natural Environment." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 5(2), Article 74. *Predicts receptivity from contextual sensing features. Uses traditional ML; we show LLM agents can do this without feature engineering.*

**Cancer survivorship mHealth context:**

- **Mattingly, T.J., et al.** (2025). "Identifying Time-Variant Predictors of Interest in Completing Brief Digital Mental Health Interventions Among Adult Survivors of Cancer: Ecological Momentary Assessment Study." *JMIR mHealth and uHealth*, 13, e69244. *N=407 cancer survivors, 5-week EMA — highly likely drawn from the same BUCS cohort. Establishes that cancer survivors' intervention preferences vary with momentary affect and pain, motivating real-time prediction.*

**Agentic AI foundations:**

- **Yao, S., et al.** (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *International Conference on Learning Representations (ICLR)*. *Foundational work on interleaving reasoning and tool use in LLMs. Our autonomous agents follow this paradigm, adapted for sensing data investigation.*

---

## 4. Key Claims

### What this paper CAN claim (supported by evidence):

1. **Agentic reasoning substantially outperforms structured pipelines on the same data.** Auto-Multi (BA=0.660) vs. Struct-Multi (BA=0.603): +9.5% relative improvement. Auto-Sense (BA=0.589) vs. Struct-Sense (BA=0.516): +14.1%. This is the cleanest finding — same model, same data, different reasoning architecture.

2. **LLM agents outperform traditional ML baselines on passive sensing.** Auto-Multi+ (BA=0.661) vs. best ML baseline RF (BA=0.518): +27.6%. Even sensing-only Auto-Sense (BA=0.589) substantially exceeds ML.

3. **Multimodal input (diary + sensing) outperforms either modality alone.** Auto-Multi (BA=0.660) vs. Auto-Sense (BA=0.589) vs. CALLM diary-only (BA=0.611).

4. **Sensing-only prediction is clinically viable.** Auto-Sense achieves BA=0.589 across 16 targets and excels on specific constructs (INT_availability BA=0.706) — without any user burden.

5. **Different constructs have different optimal modalities.** Intervention availability is best predicted by sensing (behavioral signal); emotion regulation desire benefits most from multimodal input. This has clear implications for JITAI design.

6. **The "diary paradox."** Diary text is the most informative signal, but it is absent precisely when users need help most. Sensing-only agents offer a viable fallback.

### What this paper should NOT overclaim:

- **Do not claim real-time deployment.** This is a retrospective replay study. The agents never ran in real time with actual users. Be transparent: "retrospective simulation on longitudinal data."
- **Do not claim generalizability beyond cancer survivors.** The BUCS population (post-treatment cancer survivors) has specific characteristics. The approach likely transfers, but the numbers don't.
- **Do not overclaim about the ML baselines.** The ML baselines used 5-fold CV on 399 users while LLM versions used 50 users. Acknowledge this asymmetry explicitly and note these are placeholder comparisons (as the user indicated they will update).
- **Do not claim the filtering adds meaningful value.** Auto-Multi+ (0.661) vs. Auto-Multi (0.660) is negligible. Frame this as a finding (agents are robust to noise), not a limitation.
- **Do not claim the 50-user subset is perfectly representative.** NA_State showed a small but significant difference (p=0.028); EMA count was much higher by design. Acknowledge and discuss.

---

## 5. Narrative Arc

### Problem (1-2 pages)
Cancer survivors face ongoing emotional challenges post-treatment. Just-in-time adaptive interventions (JITAIs) promise to deliver support at the right moment, but they require knowing two things: (1) does this person need support? and (2) are they available to receive it? Current LLM approaches to mental health prediction rely on active self-report (diary, social media), creating a fundamental gap: the people who most need intervention are least likely to self-report. Passive smartphone sensing captures behavioral signals continuously, but traditional ML on sensing features has shown limited predictive power for emotional states.

### Key insight / Approach (2-3 pages)
What if we gave an LLM agent direct access to a person's behavioral data streams and let it investigate — like a clinician reviewing a patient chart? PULSE deploys per-user LLM agents that autonomously query 8 sensing modalities through purpose-built tools (MCP), maintain longitudinal memory of each individual's patterns, and consult a cross-user case library for calibration. Unlike structured pipelines that apply the same fixed analysis to every user and every moment, autonomous agents adapt their investigation strategy based on what they find.

### Evaluation design (2-3 pages)
A 2x2 factorial design cleanly isolates two dimensions: (1) structured vs. autonomous reasoning, and (2) sensing-only vs. multimodal input. Seven system versions (including CALLM baseline and filtered variants) are evaluated on 50 cancer survivors from the BUCS study (~3,900 EMA entries per version) across 16 binary prediction targets. ML baselines (RF, XGBoost, Logistic Regression) provide a non-LLM reference point.

### Results (4-5 pages)
Present the 2x2 results first — this is the "headline" finding. Then dive into per-target analysis (the INT_availability vs. ER_desire divergence is compelling). Show the ML baseline comparison. Include per-user variability analysis. Discuss the diary paradox.

### Discussion (3-4 pages)
(1) Why agentic reasoning helps: agents adapt their strategy per user/context, catch signals that fixed pipelines miss. (2) Clinical implications: sensing-only prediction enables truly proactive intervention. (3) The right tool for the right construct: behavioral availability is best detected from behavior; emotional desire needs richer context. (4) Limitations: retrospective design, 50-user subset, cost/latency of LLM inference, potential for LLM reasoning errors.

### Impact statement
PULSE demonstrates that LLM agents can serve as proactive clinical reasoning partners, autonomously investigating behavioral data to identify intervention opportunities — a capability that could transform how JITAIs are designed and delivered.

---

## 6. What to Emphasize vs. De-Emphasize

### Emphasize:

- **The 2x2 factorial design.** This is the paper's methodological strength. IMWUT reviewers will appreciate the controlled comparison. It cleanly answers: "Does agentic reasoning help?" (yes) and "Does multimodal input help?" (yes).
- **The agentic investigation process.** Include concrete examples of agent reasoning traces — what tools it called, what data it examined, how it arrived at a prediction. This is what makes the paper come alive and distinguishes it from "we ran GPT on some features."
- **Per-target analysis, especially the INT_availability vs. ER_desire divergence.** This is a clinically meaningful finding: behavioral constructs (availability) are best predicted from behavior (sensing), while psychological constructs (desire) benefit from self-report context. This has direct JITAI design implications.
- **The diary paradox.** This is a compelling narrative hook and the core motivation for proactive sensing.
- **Cross-user RAG for calibration.** The idea that an agent can consult "patients like this one" is intuitive and clinically grounded.

### De-emphasize:

- **Filtering variants (Auto-Sense+ and Auto-Multi+).** Results are negligible over unfiltered versions. Mention briefly as evidence of agent robustness to noise; do not present as a separate contribution.
- **Absolute BA numbers in isolation.** BA of 0.66 is not spectacular in a vacuum. Always present in context: relative to baselines, relative to the difficulty of predicting momentary affect from passive sensing, relative to the clinical use case (screening, not diagnosis).
- **ML baseline comparison details.** These are on a different user set (399 vs. 50) and serve as a reference point, not a fair head-to-head. Acknowledge, present, move on.
- **Raw cost/token counts.** The Claude Max subscription approach is pragmatic but not generalizable. Mention it in the method; do not dwell on it.
- **The MCP protocol specifics.** This is an implementation detail. Describe it enough for reproducibility; do not position MCP itself as a contribution.

---

## 7. Target Reader

### Primary audience:
**IMWUT researchers working at the intersection of mobile sensing, health, and AI** — people who build sensing systems, design mHealth interventions, and evaluate predictive models for behavioral health. They care about:
- Does this sensing approach actually work for a real clinical population?
- Is the evaluation rigorous and controlled?
- What are the practical implications for system design?
- Is this deployable, or is it a proof of concept?

### Secondary audience:
**Affective computing and clinical HCI researchers** — people studying emotion recognition, JITAI design, and digital mental health interventions. They care about:
- The receptivity framework (desire + availability)
- The diary paradox and its implications
- Whether LLMs can replace traditional feature engineering for sensing data

### Tertiary audience:
**LLM/agentic AI researchers** interested in grounding agent reasoning in real-world data streams beyond benchmarks and web tasks. This paper provides a compelling domain-specific application of agentic tool use.

### What readers do NOT care about:
- LLM prompt engineering details (keep in appendix)
- The specific Claude model version (mention once; it will be outdated soon)
- Infrastructure details (MCP server setup, CLI commands)

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "N=50 is too small."
**Preemption:** Acknowledge directly. 50 users were selected for high EMA compliance to ensure evaluation quality (~3,900 entries per version). Show the representativeness analysis (base rates match on 3/4 targets, p>0.05). Argue that the 2x2 factorial design, where each user serves as their own control across conditions, provides statistical power for within-subject comparisons even at N=50. Frame as a pilot that motivates the full 418-user evaluation (which can be presented as ongoing work).

### Objection 2: "Retrospective replay is not the same as real-time deployment."
**Preemption:** Be upfront in the method section. The information boundary is strictly enforced (agents only see past data). Acknowledge that real-time deployment introduces latency constraints, infrastructure challenges, and user interaction dynamics that are not captured. Frame the retrospective design as a necessary first step that establishes whether the agentic approach has predictive validity before committing to a costly real-time study.

### Objection 3: "The ML baselines are not on the same data."
**Preemption:** This is a real weakness. State it clearly: ML baselines used 5-fold CV on 399 users; LLM agents ran on 50. The comparison is directional, not definitive. If possible, re-run ML baselines on the same 50 users before submission (the user indicated these will be updated). If not, present with appropriate caveats.

### Objection 4: "How do you know the LLM isn't just hallucinating plausible-sounding predictions?"
**Preemption:** The 2x2 design is the answer. If the LLM were merely generating plausible outputs, structured and autonomous versions would perform similarly. The consistent advantage of autonomous over structured (across both sensing-only and multimodal conditions) suggests the agent is genuinely leveraging its investigation freedom. Additionally, present agent reasoning traces showing specific data queries and evidence integration.

### Objection 5: "BA of 0.66 is modest. Is this clinically useful?"
**Preemption:** Context is everything. (1) This is momentary affect from passive sensing only — inherently noisy. (2) Traditional ML on the same sensing data achieves BA=0.52. (3) For JITAI applications, even modest improvements in identifying receptive moments can meaningfully reduce user burden and improve intervention effectiveness. (4) Per-target performance reaches BA=0.75 for emotion regulation desire and BA=0.72 for intervention availability — the two most actionable targets.

### Objection 6: "This only works because of the specific LLM (Claude Sonnet). Would it generalize to other models?"
**Preemption:** Acknowledge. The core contribution is the agentic investigation architecture and tool design, not the specific LLM. The 2x2 design isolates the effect of agentic reasoning style, which is model-agnostic in principle. Future work should test with other frontier LLMs. Note that the MCP tool interface is model-agnostic by design.

### Objection 7: "Cost and latency make this impractical for real-time JITAI."
**Preemption:** Acknowledge the current cost/latency profile. Frame PULSE as a research prototype that establishes what is achievable, not a production system. Note that LLM inference costs are declining rapidly. The agentic architecture could be distilled into a lighter-weight model once the investigation patterns are understood. Also, JITAI timing windows are typically 30-60 minutes, not seconds — some latency is tolerable.

### Objection 8: "The per-user memory could introduce data leakage."
**Preemption:** Memory only records prediction outcomes and self-reflections based on receptivity signals (previous EMA response patterns, not ground truth for the target being predicted). The information boundary is strictly enforced: memory is updated only after ground truth for past windows is revealed, never for the current prediction target. Detail this in the method.

---

## Summary of Recommended Strategy

The paper's strongest selling point is the **2x2 factorial design isolating the effect of agentic reasoning** — this is methodologically clean and rare in the LLM-for-health literature, where most papers test a single system. Lead with this.

The **narrative hook** is the diary paradox: the most informative data (self-report) is missing when it matters most. PULSE is the answer: proactive prediction from passive sensing, powered by agents that reason like clinicians.

The **clinical grounding** in cancer survivorship and JITAI design gives this paper a clear application story that IMWUT values. This is not "LLMs for the sake of LLMs" — it is a system designed to solve a real clinical problem.

Position relative to **GLOSS** (concurrent IMWUT work) and **Mental-LLM** (prior IMWUT work) to show awareness of the venue's trajectory. PULSE is distinguished by: (a) clinical population rather than general wellness, (b) prediction with ground-truth evaluation rather than open-ended sensemaking, (c) the controlled factorial design, and (d) the specific focus on intervention receptivity as a compound construct.
