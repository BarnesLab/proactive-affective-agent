# Framing Proposal: PULSE Paper for IMWUT

**Advisor**: Varun Mishra (Northeastern University, UbiWell Lab)
**Perspective**: Receptivity detection, mHealth intervention timing, LLM + passive sensing

---

## 1. Title Options (Ranked)

1. **PULSE: Agentic LLM Investigation of Passive Sensing Data for Proactive Affective State Prediction in Cancer Survivors**
   - Best option. "Agentic LLM Investigation" captures the novelty (autonomous tool-use, not prompt-stuffing). "Passive Sensing Data" grounds it in ubicomp. "Proactive" distinguishes from CALLM's reactive diary approach. "Cancer Survivors" signals clinical population.

2. **From Diaries to Sensors: How Agentic LLMs Unlock Proactive Emotional State Prediction from Passive Mobile Sensing**
   - Narrative-driven, emphasizes the reactive-to-proactive shift. Slightly less precise about the clinical context.

3. **Autonomous LLM Agents for Affective Computing: Tool-Augmented Reasoning over Multimodal Smartphone Sensing**
   - Emphasizes the technical contribution (tool-augmented reasoning). Better for an AI/systems audience than a clinical/ubicomp audience.

**Recommendation**: Title 1 is strongest for IMWUT. It names the system, the method, and the application domain. Reviewers will immediately understand what is new (agentic investigation), what is being predicted (affective states), and from what data (passive sensing).

---

## 2. Core Contribution Framing

THE story of this paper, in 2-3 sentences:

> Existing LLM-based affect prediction systems require users to actively report (e.g., writing diary entries), which creates a fundamental gap: the people who most need support are least likely to provide input. PULSE demonstrates that LLM agents equipped with autonomous sensing-investigation tools can predict emotional states and intervention receptivity from passive smartphone data alone, without requiring any user input. Through a 2x2 factorial evaluation on 50 cancer survivors, we show that agentic reasoning over sensing data (balanced accuracy 0.66) significantly outperforms both structured LLM pipelines (0.60) and traditional ML baselines (0.52), and that the agent's ability to autonomously decide what data to investigate is the key differentiator.

The single-sentence elevator pitch: **Letting LLMs autonomously investigate behavioral sensing data, rather than feeding them pre-formatted features, transforms passive sensing from a weak signal into a clinically viable predictor of emotional states.**

---

## 3. Positioning

### The Gap This Fills

There is a clear gap at the intersection of three rapidly converging threads:

**Thread 1: Passive sensing for mental health** has a long history in ubicomp but is stuck at modest predictive performance. Traditional ML on smartphone features typically achieves balanced accuracies of 0.50-0.65 for affect prediction, with severe generalization problems.
- Wang et al., "Tracking Depression Dynamics in College Students Using Mobile Phone and Wearable Sensing," IMWUT 2018 (Rui Wang, Weichen Wang, Alex daSilva, et al.)
- Xu et al., "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling," IMWUT 2022 (Xuhai Xu, Xin Liu, Han Zhang, et al.) -- Distinguished Paper Award at UbiComp 2023
- Saeb et al., "Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior," JMIR 2015 (Sohrab Saeb, Mi Zhang, Christopher Karr, et al.)
- Adler et al., "Machine Learning for Passive Mental Health Symptom Prediction: Generalization Across Different Longitudinal Mobile Sensing Studies," PLOS ONE 2022 (Daniel Adler, et al.)

**Thread 2: LLMs for health prediction** is nascent but growing, primarily focused on text-based or pre-formatted feature inputs -- not autonomous investigation.
- Xu et al., "Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data," IMWUT 2024 (Xuhai Xu, et al.)
- Kim et al., "Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data," CHIL 2024 (Yubin Kim, Xuhai Xu, Daniel McDuff, et al.)
- Feng et al., "A Comparative Study of Traditional Machine Learning, Deep Learning, and Large Language Models for Mental Health Forecasting using Smartphone Sensing Data," IMWUT 2026 (Kaidong Feng et al.)
- Bae et al., "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features," UbiComp Companion 2024 (authors from University of Melbourne)

**Thread 3: LLM agents with tool use** is transforming AI but has barely been applied to sensing/health domains.
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023 (Shunyu Yao, Jeffrey Zhao, Dian Yu, et al.)
- Choube et al., "GLOSS: Group of LLMs for Open-ended Sensemaking of Passive Sensing Data for Health and Wellbeing," IMWUT 2025 (Akshat Choube, Ha Le, Jiachen Li, Kaixin Ji, Vedant Das Swain, Varun Mishra)

**Thread 4: Receptivity and intervention timing in mHealth.**
- Kunzler, Mishra et al., "Exploring the State-of-Receptivity for mHealth Interventions," IMWUT 2019 (Florian Kunzler, Varun Mishra, Jan-Niklas Kramer, David Kotz, et al.)
- Mishra et al., "Detecting Receptivity for mHealth Interventions in the Natural Environment," IMWUT 2021 (Varun Mishra, Florian Kunzler, Jan-Niklas Kramer, et al.)
- Nahum-Shani et al., "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles," Annals of Behavioral Medicine 2018 (Inbal Nahum-Shani, Shawna N. Smith, et al.)

**The predecessor work:**
- Wang et al., "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models," arXiv 2025 / CHI 2026 submission (this is the direct predecessor; PULSE extends it from reactive diary-based to proactive sensing-based)

### Key Comparisons

| System | What it does differently from PULSE |
|--------|-------------------------------------|
| **CALLM** | Requires diary text (reactive); no sensing; no tool-use |
| **Health-LLM** | Pre-formatted features fed to LLM; no tool-use; no temporal reasoning |
| **Mental-LLM** | Text-only (social media posts); no sensing data; no agentic investigation |
| **GLOSS** | Code-generation approach to sensemaking; not predictive; post-hoc analysis |
| **Feng et al. (2026)** | Compares ML/DL/LLM on sensing but with static prompts, not agentic tool-use |
| **Bae et al. (2024)** | Zero/few-shot LLM on sensor features; no autonomous investigation |
| **Traditional ML** | Feature engineering + classification; no contextual reasoning |

### The Unique Position

PULSE is the **first system to deploy LLM agents with autonomous tool-use for prospective affect prediction from passive sensing data.** Unlike prior work that feeds pre-computed features to LLMs (Health-LLM, Mental-LLM, Bae et al.), PULSE agents decide what data to examine, how far back to look, which cross-user comparisons to make, and how to weight conflicting signals -- all through tool calls during inference. This is closer to how a clinician works: forming hypotheses and investigating evidence, rather than receiving a fixed feature vector.

---

## 4. Key Claims

### Claims the paper CAN make (supported by evidence):

1. **Agentic investigation significantly outperforms structured pipelines on the same data.** Auto-Multi (BA 0.660) vs. Struct-Multi (BA 0.603) is a clean within-design comparison. Same LLM, same data, same prompts -- the only difference is whether the agent decides its own investigation strategy. This is the paper's strongest claim.

2. **Passive sensing alone, when investigated by an agentic LLM, achieves clinically meaningful prediction.** Auto-Sense (BA 0.589) significantly exceeds ML baselines (BA ~0.52) and approaches diary-based CALLM (BA 0.611). This is important because sensing requires zero user effort.

3. **Multimodal integration (diary + sensing) outperforms either modality alone when processed by an agentic LLM.** Auto-Multi (0.660) > Auto-Sense (0.589) > CALLM-diary-only (0.611). The agent can triangulate signals.

4. **Intervention availability is best predicted by sensing, not diaries.** Auto-Sense (BA 0.706) significantly outperforms CALLM (BA 0.542) for INT_avail. This makes theoretical sense: availability is a behavioral construct (are you busy? at home? phone idle?), not an emotional one.

5. **The system works for a clinical cancer-survivor population**, not just college students (which dominate the sensing literature). N=50 users with ~3,900 entries per version is substantial for an LLM-based system.

6. **The LLM agent's investigation strategy matters more than data filtering.** Auto-Multi+ (0.661) ~ Auto-Multi (0.660) -- pre-filtering adds negligible value when the agent can already handle noise.

### Claims the paper should NOT make:

1. **Do NOT claim real-time deployment readiness.** This is retrospective simulation on historical data. The paper should be transparent about this.

2. **Do NOT claim superiority to all ML baselines in a fair comparison.** The ML baselines were trained on 399 users with 5-fold CV; the LLM was tested on 50 users. While the LLM numbers are clearly higher, acknowledge this asymmetry explicitly and frame the ML comparison as "reference point" rather than "head-to-head."

3. **Do NOT claim generalizability beyond this population.** N=50 cancer survivors with high EMA compliance is a specific group. The 50-user subset was selected for high compliance, which could affect base rates (NA_State showed a significant difference).

4. **Do NOT overclaim about the causal mechanism of why agentic works.** You can show that it works better, but the "why" requires analysis of tool-use traces (which tools were called, how often, in what order). If you have this data, include it. If not, acknowledge it as future work.

5. **Do NOT position this as replacing clinical judgment.** Frame as "decision support" or "screening tool."

---

## 5. Narrative Arc

### Problem (1-2 pages)
Cancer survivors need proactive emotional support, but current systems are reactive -- they require users to write diary entries. The paradox: those who need help most are least likely to self-report. Passive smartphone sensing could fill this gap, but traditional ML on sensing features yields weak predictions (BA ~0.52). Meanwhile, LLMs show promise for health prediction but are used passively: features go in, predictions come out.

**Key tension**: Can LLMs do more than pattern-match on pre-formatted features? Can they actively investigate behavioral data the way a clinician investigates a patient's history?

### Approach (4-5 pages)
Introduce PULSE: an LLM agent architecture where each prediction is an autonomous investigation. The agent receives a timestamp and user profile, then decides what to investigate using 6 sensing query tools (daily summary, behavioral timeline, targeted hourly query, baseline comparison, receptivity history, similar days). The agent builds up evidence iteratively before making a prediction.

The 2x2 design: {Structured, Autonomous} x {Sensing-only, Multimodal}. This isolates the contribution of agentic reasoning (structured vs. autonomous) from the contribution of data modality (sensing vs. multimodal). Include CALLM and ML baselines for external reference points.

Describe the MCP tool architecture, cross-user RAG, session memory, and the retrospective simulation protocol (critical: emphasize the information boundary -- agent never sees future data).

### Results (5-6 pages)
Present the 2x2 results first (the factorial design is the cleanest story). Then expand to:
- Per-target analysis (especially the INT_avail finding about sensing > diary for behavioral constructs)
- ML baseline comparison (with appropriate caveats about the different evaluation setup)
- CALLM comparison (diary + RAG vs. sensing + tool-use vs. both)
- Effect of filtering (marginal benefit)
- Statistical tests (Wilcoxon signed-rank, bootstrap CIs)
- Representativeness analysis (50 vs. 418 users)
- (If available) Tool-use analysis: what tools do agents call? How does investigation strategy vary across users and targets?

### Impact (2-3 pages)
This work establishes a new paradigm: LLM agents as active investigators of behavioral data, not passive consumers of feature vectors. Implications for:
- JITAI systems: proactive receptivity detection without user burden
- Clinical populations: cancer survivors are underserved by current mHealth tools
- The passive sensing field: agentic LLMs may revive sensing modalities previously considered too noisy

---

## 6. What to Emphasize vs. De-emphasize

### EMPHASIZE:

1. **The 2x2 factorial design.** This is methodologically strong and uncommon in LLM-for-health papers. It cleanly isolates the agentic reasoning contribution. Lead with this.

2. **Agentic vs. Structured comparison.** This is the paper's strongest and most novel result. Auto-Multi >> Struct-Multi on the same data proves that *how* the LLM interacts with data matters, not just *what* data it sees.

3. **The tool architecture.** IMWUT reviewers care about systems. The 6 MCP tools, the investigation protocol, the cross-user RAG -- these are the technical contributions. Describe them in enough detail that someone could reimplement.

4. **INT_avail as a behavioral construct.** The finding that sensing outperforms diary for availability prediction is theoretically interesting and practically important for JITAI design. This directly connects to the receptivity literature.

5. **The diary paradox.** Diary text is the most informative modality but is absent when most needed. This motivates the entire sensing approach and is a compelling narrative hook.

6. **Per-user analysis and variance.** IMWUT reviewers will want to see that performance is not driven by a few easy users. Show per-user BA distributions.

### DE-EMPHASIZE:

1. **ML baseline comparison.** Present it, but do not make it a primary claim. The evaluation setups differ (399 users 5-fold CV vs. 50 users). Frame as "reference point indicating that traditional ML on sensing features yields near-chance performance on this task."

2. **Filtering variants (Auto-Sense+ and Auto-Multi+).** The marginal improvement is not interesting as a standalone finding. Mention it briefly to show that the agent is robust to noise, then move on.

3. **Absolute BA numbers in isolation.** BA of 0.66 is not impressive in a vacuum. The story is the *relative* improvement over structured (0.60) and ML (0.52), and the fact that this comes from passive sensing with zero user burden.

4. **Cost/infrastructure details.** The Claude Max subscription approach is pragmatically clever but reviewers may see it as a limitation (not reproducible, vendor-dependent). Mention it once, then focus on the methodology.

5. **Session memory / self-reflection.** Unless you can show that performance improves over time within a user's study period (learning curve analysis), do not make strong claims about the memory system. Present it as part of the architecture but focus evaluation on the tool-use mechanism.

---

## 7. Target Reader

### Primary audience: IMWUT/UbiComp researchers working on:
- **Mobile sensing for mental health** (the largest cohort at IMWUT). They care about: Does this actually work on real sensing data? Is the evaluation rigorous? How does it compare to standard ML pipelines?
- **mHealth / JITAI design**. They care about: Can this predict receptivity in real time? What is the latency? Does it work without user input?
- **LLM + sensing systems** (rapidly growing). They care about: What is the technical architecture? Is this reproducible? What is the computational cost?

### Secondary audience:
- **Clinical informatics / digital health researchers** interested in cancer survivorship support
- **AI/ML researchers** exploring agentic tool-use in domain-specific applications

### What the primary reader cares about:
1. **Rigor of the evaluation.** N=50, statistical tests, comparison to baselines, discussion of limitations.
2. **Practical viability.** Can this approach scale? What are the cost, latency, and deployment constraints?
3. **Reproducibility.** Are the tools, prompts, and evaluation protocol described in enough detail?
4. **Clinical relevance.** Is balanced accuracy of 0.66 actually useful for intervention delivery?

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "N=50 is small."
**Preemption**: Each user contributes ~78 EMA entries on average (3,900 total per version). The 50 users were verified to be representative of the full 418-participant BUCS cohort on key outcome distributions (p > 0.05 for PA, ER_desire, INT_avail; small effect for NA). Frame this as a large-scale pilot with plans for full-cohort evaluation. Note that 50 users x 7 versions x ~78 entries = ~27,300 total predictions, which is substantial.

### Objection 2: "The ML baselines are not fairly compared (different N, different evaluation)."
**Preemption**: Acknowledge this explicitly. State that ML baselines on 399 users with 5-fold CV serve as a *reference point for the difficulty of the prediction task*, not as a direct head-to-head comparison. The critical comparison is within the LLM versions (factorial design), where the evaluation setup is identical. Commit to running ML baselines on the same 50 users as a revision deliverable, or include them.

### Objection 3: "This is retrospective, not a real deployment."
**Preemption**: The retrospective simulation protocol strictly enforces temporal boundaries (agent never sees future data). Describe this in detail. Acknowledge that real-time deployment introduces latency and data-quality challenges. Frame the contribution as establishing the *predictive validity* of agentic sensing investigation, with deployment as explicit future work.

### Objection 4: "What is the LLM actually doing? Is this just a black box?"
**Preemption**: This is where tool-use traces become critical. Include an analysis of which tools the agent calls, how often, and how investigation strategy varies by target and user. Provide example investigation traces showing the agent's reasoning. If possible, show that certain tool-use patterns correlate with prediction accuracy. The structured vs. autonomous comparison also partially addresses this: the structured version makes the LLM's process transparent but performs worse, suggesting that the autonomous investigation strategy itself is valuable.

### Objection 5: "Balanced accuracy of 0.66 is still modest. Is this clinically useful?"
**Preemption**: Contextualize against the literature. ML on passive sensing typically achieves BA 0.50-0.65 for similar tasks. The key insight is not that 0.66 is "good enough" in absolute terms, but that (a) it comes from passive data requiring zero user effort, (b) certain targets like INT_avail reach 0.72, and (c) the 2x2 analysis reveals a clear mechanism for improvement (agentic reasoning). Discuss that in a JITAI context, even modest predictive signal enables better-than-random intervention timing, which Kunzler et al. (IMWUT 2019) showed can improve receptivity by up to 77% in F1.

### Objection 6: "The results depend on a specific LLM (Claude Sonnet). Would this generalize to other models?"
**Preemption**: Acknowledge this as a limitation. The contribution is the *architecture* (agentic tool-use for sensing investigation), not the specific LLM. The 2x2 design controls for the LLM itself -- all versions use the same model, so the agentic vs. structured comparison is valid regardless of which LLM is used. Discuss plans for multi-model evaluation.

### Objection 7: "How does this relate to GLOSS? Both use LLMs with sensing data."
**Preemption**: GLOSS (IMWUT 2025) is the closest related work. Key differences: (1) GLOSS is a *sensemaking* system for post-hoc analysis; PULSE is a *prediction* system for prospective use. (2) GLOSS uses code-generation to process data; PULSE uses purpose-built query tools. (3) GLOSS targets open-ended questions; PULSE targets specific clinical prediction tasks with ground truth evaluation. (4) PULSE introduces the agentic vs. structured factorial comparison, which GLOSS does not address. Frame PULSE as complementary: GLOSS showed LLMs can make sense of sensing data; PULSE shows LLM agents can make actionable predictions from it.

### Objection 8: "High-compliance users were selected. Performance may not hold for typical users."
**Preemption**: Acknowledge this directly. The 50-user subset was selected for sufficient data density, which is methodologically necessary for a first evaluation. Show the representativeness statistics. Discuss that in a deployed system, missing data handling (which the agent already does via tool responses returning "no data") would become more important. The finding that Auto-Multi+ ~ Auto-Multi (filtering does not help) suggests the agent already handles sparse/noisy data reasonably well.

---

## Summary of Recommended Framing

The paper should be framed as a **systems + evaluation** paper for IMWUT, with the following priority of contributions:

1. **Primary**: The agentic sensing investigation paradigm (LLM agents with autonomous tool-use for behavioral data)
2. **Secondary**: The 2x2 factorial evaluation isolating agentic reasoning benefit
3. **Tertiary**: Clinical application to cancer survivor emotional state prediction and intervention timing

The narrative should flow: *Problem (diary paradox) -> Insight (let LLMs investigate, not just consume) -> System (PULSE + MCP tools) -> Evaluation (2x2 on 50 cancer survivors) -> Findings (agentic >> structured, sensing viable, multimodal best) -> Implications (new paradigm for sensing + LLM integration).*

Do NOT bury the lead: the key finding is that agentic reasoning is the differentiator, not the specific data modalities or the LLM itself. This is what makes the paper a contribution beyond "we threw sensing data at an LLM."
