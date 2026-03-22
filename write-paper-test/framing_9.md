# Framing Proposal #9 — Bonnie Spring (Northwestern University)

**Perspective**: mHealth behavior change interventions, JITAI optimization, multiple risk behavior change in cancer survivors, mobile technology for health promotion

---

## 1. Title Options (Ranked)

1. **PULSE: Proactive Affective Agents for Predicting Intervention Receptivity in Cancer Survivors via Autonomous Sensing Investigation**
   - Strongest option. Leads with the clinical application ("intervention receptivity in cancer survivors"), foregrounds the proactive/autonomous angle, and flags the sensing investigation as the mechanism. IMWUT reviewers will immediately see both the systems contribution and the health application.

2. **From Reactive Diaries to Proactive Sensing: LLM Agents That Autonomously Investigate Behavioral Data to Predict Emotional States and Intervention Opportunities**
   - Narrative-driven. Sets up the reactive-to-proactive transition that IS the story. Slightly long but captures the trajectory well.

3. **Autonomous LLM Agents for Passive Sensing-Based Affect Prediction: A 2x2 Factorial Evaluation with Cancer Survivors**
   - Clean and methodological. Highlights the factorial design, which is unusual and rigorous for IMWUT. Undersells the clinical motivation.

**My recommendation**: Title 1. It signals clinical relevance (cancer survivors, intervention receptivity), technical novelty (autonomous sensing investigation), and positions the work where it needs to be — at the intersection of AI agents and mHealth, not purely in either camp.

---

## 2. Core Contribution Framing

**THE story of this paper is this**: Current mHealth systems for predicting when cancer survivors need support rely on active self-report (diaries, EMA), but people are least likely to report when they need help most. PULSE demonstrates that LLM agents equipped with purpose-built sensing investigation tools can autonomously examine passive behavioral data — deciding what to look at, how far back to investigate, and which peer comparisons to draw — to predict emotional states and intervention receptivity with accuracy substantially exceeding both traditional ML on the same sensing data and structured LLM pipelines. The 2x2 factorial evaluation cleanly isolates that it is the autonomous investigation strategy, not just the data modality, that drives performance.

This framing matters because it connects the technical contribution (agentic tool use for sensing) directly to a concrete clinical need (knowing when to intervene without requiring the patient to tell you they need help).

---

## 3. Positioning

### The Gap This Fills

There is a well-documented gap between (a) the theoretical promise of JITAIs that deliver support at precisely the right moment, and (b) the practical inability to know WHEN that moment is without asking the person — which defeats the purpose of "just-in-time." This paper fills that gap by demonstrating that autonomous LLM agents can infer intervention-relevant states from passive sensing alone, without any self-report input.

### Key Comparison Papers (Verified)

**Foundational JITAI framework**:
- Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., & Murphy, S. A. (2018). "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support." *Annals of Behavioral Medicine*, 52(6), 446-462. — This is the definitive framework paper. PULSE addresses the "decision point" and "tailoring variable" components by providing a way to assess state without active input.

**Receptivity prediction (closest IMWUT precedent)**:
- Kunzler, F., Mishra, V., Kramer, J.-N., Kotz, D., Fleisch, E., & Kowatsch, T. (2019). "Exploring the State-of-Receptivity for mHealth Interventions." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 3(4), Article 140. — This is the most directly comparable IMWUT paper. They used ML on contextual features to predict receptivity to a physical activity chatbot (N=189). PULSE extends this in three ways: (1) richer behavioral sensing, (2) LLM-based reasoning vs. feature-engineered ML, (3) clinical population (cancer survivors, not healthy adults).

**Direct predecessor (same dataset)**:
- Wang, Z., Daniel, K. E., Barnes, L. E., & Chow, P. I. (2025). "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models." *arXiv preprint 2503.10707* (CHI 2026). — PULSE's direct predecessor. CALLM showed LLMs can predict emotional states from diary text (BA up to 73%). PULSE asks: can we do this WITHOUT requiring the user to write?

**LLMs for affect prediction from sensing**:
- Zhang, T., Teng, S., Jia, H., & D'Alfonso, S. (2024). "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features." *Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '24)*, 709-716. — The most directly comparable LLM-for-sensing work. Their study used 10 students with zero-shot Gemini prompting on weekly sensor summaries. PULSE goes vastly beyond: 50 cancer survivors, agentic tool use (not just prompting), longitudinal memory, cross-user RAG, and a controlled factorial design.

**Health-LLM benchmark**:
- Kim, Y., Xu, X., McDuff, D., Breazeal, C., & Park, H. W. (2024). "Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data." *Proceedings of the Fifth Conference on Health, Inference, and Learning (CHIL)*. — Evaluated 12 LLMs on 10 health prediction tasks from wearable data. Showed context enhancement helps. PULSE extends by giving the LLM agency over what context to retrieve, rather than pre-formatting it.

**Proactive context-aware agents**:
- ContextAgent (NeurIPS 2025). "ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions." — The most relevant agent architecture comparison. ContextAgent uses egocentric wearable sensors (video/audio) for proactive assistance. PULSE addresses a complementary modality space (smartphone behavioral sensing) for a specific clinical application.

**Digital phenotyping foundations**:
- Saeb, S., Zhang, M., Karr, C. J., Schueller, S. M., Corden, M. E., Kording, K. P., & Mohr, D. C. (2015). "Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study." *Journal of Medical Internet Research*, 17(7), e175. — Established that GPS and phone usage features correlate with depression (N=28). PULSE builds on this decade of work by moving from correlation to prediction via LLM reasoning.

**Cross-dataset generalization benchmark**:
- Xu, X., et al. (2022). "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling." *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)*, 6(4). — Showed ML models for mental health prediction generalize poorly across populations. PULSE's approach (LLM reasoning over raw behavioral narratives + per-user memory) offers a potential path around the generalization problem, though this needs explicit testing.

**Cancer survivors and multiple behavior change**:
- Spring, B., Stump, T., Penedo, F., Pfammatter, A. F., & Robinson, J. K. (2019). "Toward a Health-Promoting System for Cancer Survivors: Patient and Provider Multiple Behavior Change." *Preventive Medicine Reports*. — Argued that cancer survivors face a "silver tsunami" of cardiometabolic comorbidities and need proactive health promotion systems. PULSE provides the detection layer that such systems require.

### How to Position

Frame PULSE as advancing the JITAI pipeline specifically at the "state assessment" stage — the bottleneck that has limited real-world JITAI deployment. The contribution is NOT a new intervention. It is a new way to determine WHEN to intervene, using autonomous LLM reasoning over passive sensing data, validated in a clinically relevant population.

---

## 4. Key Claims (What to Claim vs. Not Overclaim)

### Supported Claims

1. **Autonomous agent investigation substantially outperforms structured LLM pipelines on the same data**: Auto-Multi (BA 0.660) vs. Struct-Multi (0.603); Auto-Sense (0.589) vs. Struct-Sense (0.516). This is the cleanest claim — same model, same data, different reasoning strategy. The factorial design isolates the effect.

2. **LLM agents outperform traditional ML baselines for affect prediction from passive sensing**: Auto-Multi+ (0.661) vs. best ML (RF 0.518). This is a strong claim but note the caveat about ML baseline evaluation differences (5-fold CV on 399 users vs. 50-user LLM evaluation).

3. **Multimodal input (diary + sensing) outperforms either alone**: Auto-Multi (0.660) >> Auto-Sense (0.589) >> best ML (0.518). And CALLM (diary only, 0.611) < Auto-Multi (0.660).

4. **Passive sensing alone can predict intervention-relevant states**: Auto-Sense achieves BA 0.589 across 16 targets and notably 0.706 on intervention availability — higher than CALLM's 0.542. This is the most clinically important claim: you can predict availability without any self-report.

5. **INT_availability is best predicted by behavioral sensing, not diary text**: Auto-Sense (0.706) >> CALLM (0.542). This makes theoretical sense — availability is a behavioral state (am I busy? am I in a context where I can engage?) best captured by behavioral data.

6. **The "diary paradox" is real and quantifiable**: Diary input is the single most informative modality, yet it is systematically absent when users are most distressed or disengaged — exactly when intervention is most needed. Sensing provides a fallback that works precisely when diaries fail.

### What NOT to Overclaim

1. **Do NOT claim this is a deployed intervention system.** This is a retrospective simulation on historical data. The agents never actually delivered interventions. Frame as "evaluation of predictive capability" not "evaluation of an intervention."

2. **Do NOT claim real-time feasibility.** The current system runs on Claude via CLI with substantial compute per prediction. Do not claim this could run on-device or in real-time without acknowledging the latency and cost gap.

3. **Do NOT overclaim generalizability beyond this population.** The 50-user sample is a convenience sample of high-compliance users from a specific cancer survivor study. The representativeness analysis shows the sample is similar on most metrics but differs on NA_State and platform distribution.

4. **Do NOT claim the ML baselines are definitive.** They were run on 399 users with 5-fold CV, a different evaluation setup than the 50-user LLM evaluation. Acknowledge this asymmetry transparently.

5. **Do NOT claim agentic reasoning explains WHY performance improves.** You can show THAT it improves, not WHY. The autonomous agent is a black box. Include analysis of agent investigation traces as qualitative evidence, but do not claim mechanistic understanding.

---

## 5. Narrative Arc

### Problem (1-2 pages)
Cancer survivors face elevated risks of emotional distress, yet the mHealth tools meant to support them depend on active self-report. The fundamental paradox: people who most need support are least likely to report that they need it. JITAIs promise to deliver the right support at the right time, but they require accurate, passive assessment of when someone is in a state to benefit from intervention. Traditional ML approaches to this problem have shown modest accuracy (BA ~0.52) and poor generalization.

### Insight (0.5 page)
What if, instead of pre-engineering features and feeding them to a classifier, we let an LLM agent autonomously investigate a user's behavioral data — choosing which modalities to examine, how far back to look, and which peer cases to compare against? This mirrors how a clinical psychologist would reason: examining patterns, comparing to prior sessions, drawing on experience with similar patients.

### Approach (3-4 pages)
PULSE: a system of per-user LLM agents that predict emotional states and intervention receptivity. Each agent has access to 8 MCP tools for querying sensing data, a persistent memory of past interactions with its user, and a cross-user retrieval system for calibration. A 2x2 factorial design (structured vs. autonomous x sensing-only vs. multimodal) cleanly isolates the contributions of agentic reasoning and data modality.

### Evaluation (5-6 pages)
Retrospective evaluation on 50 cancer survivors from the BUCS study (~3,900 prediction entries per version). Seven system versions + ML baselines. Primary metrics: balanced accuracy and F1 across 16 binary prediction targets.

### Results (4-5 pages)
The key results tell a clear story:
1. Autonomous >> Structured (regardless of modality) — agentic reasoning matters
2. Multimodal >> Sensing-only — diary text helps when available
3. LLM agents >> ML baselines — reasoning over data beats feature engineering
4. Sensing alone can predict availability better than diary alone — passive monitoring works for behavioral states
5. The diary paradox: diary is most informative but systematically missing when most needed

### Implications (2-3 pages)
What this means for JITAI design: passive sensing + LLM reasoning can provide the "state assessment" component that JITAIs need but have lacked. Discuss deployment considerations (latency, cost, privacy), limitations of retrospective evaluation, and next steps toward real-time pilot.

---

## 6. What to Emphasize vs. De-emphasize

### Emphasize

- **The 2x2 factorial design**: This is methodologically unusual for IMWUT and is the paper's secret weapon. It lets you make clean causal claims about the value of agentic reasoning and multimodal input separately. Lead with this in the evaluation.

- **The clinical framing**: IMWUT reviewers appreciate papers that solve real problems for real populations. Cancer survivors are sympathetic, the clinical need is genuine, and the "diary paradox" is immediately intuitive. Do not bury the clinical motivation in favor of pure systems description.

- **INT_availability results**: The finding that sensing outperforms diary for availability prediction (Auto-Sense 0.706 vs. CALLM 0.542) is the most clinically actionable result. It means you can determine if someone is available for an intervention without interrupting them to ask.

- **Per-target analysis for the 4 focus constructs**: The disaggregated results (ER_desire, INT_avail, PA_State, NA_State) tell a richer story than the aggregate. Auto-Multi+ achieves 0.751 BA on emotion regulation desire — that is a strong signal for a passive system.

- **Agent investigation traces**: Include qualitative examples of how the autonomous agent decided what to investigate. This makes the "agentic" concept concrete and gives reviewers something to inspect.

### De-emphasize

- **The filtering variants (Auto-Sense+, Auto-Multi+)**: The fact that filtering is marginal (0.661 vs. 0.660) is worth noting briefly but should not be a major focus. It is a negative result that does not advance the narrative.

- **Absolute performance numbers in isolation**: BA of 0.66 may not sound impressive out of context. Always present relative to the baselines and relative to the difficulty of the task (predicting momentary affect from passive sensing is genuinely hard). Frame around effect sizes and paired comparisons.

- **Implementation details of the MCP tool protocol**: Describe what the tools DO (query daily summaries, find similar days, compare to peers), not the plumbing of how MCP works. The contribution is the agentic investigation strategy, not the tool protocol.

- **Cost optimization / Claude Max subscription**: This is an engineering convenience, not a contribution. Mention briefly in the method section but do not feature it.

- **The 16-target breadth**: Having 16 binary targets is comprehensive but risks diluting the message. Structure the results around the 4 focus constructs (ER_desire, INT_avail, PA_State, NA_State) with the full 16 in supplementary material.

---

## 7. Target Reader

### Primary Audience
**IMWUT researchers working on mobile health sensing and context-aware systems**, particularly those building predictive models from smartphone data for health applications. These readers care about:
- Whether LLM-based approaches genuinely outperform traditional ML on sensing data
- How to design agent systems that interact with sensor streams
- Methodological rigor (factorial design, proper baselines, statistical tests)
- Clinical relevance and real-world deployability

### Secondary Audience
**Health behavior researchers and JITAI designers** who need practical solutions for the "state assessment" bottleneck. My community — people designing the next generation of adaptive interventions — will read this paper to understand whether LLM agents can replace or supplement the EMA-based decision rules we currently use. They care about:
- Can this actually be deployed in clinical trials?
- How does it handle missing data and variable compliance?
- What is the clinical significance of the accuracy improvements?
- Does it work for the populations that need it most?

### Tertiary Audience
**AI/ML researchers** interested in agentic tool use for real-world tasks beyond coding and web browsing. The factorial evaluation of agentic vs. structured reasoning on a substantive prediction task is methodologically interesting independent of the health application.

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "This is a retrospective simulation, not a real deployment"
**Severity**: High. This will come up.
**Preemption**: Acknowledge forthrightly in the limitations. Frame the contribution as validating the predictive capability that would undergird a JITAI, not as evaluating an intervention. Cite HeartSteps (Klasnja et al., 2019) and other MRT/JITAI papers that similarly validated components before deployment. Provide a concrete "path to deployment" section discussing what would need to change for real-time use.

### Objection 2: "N=50 is small and non-representative"
**Severity**: High.
**Preemption**: Present the representativeness analysis (base rates match on 3/4 targets, p > 0.05). Acknowledge the selection of high-compliance users and frame it as a deliberate design choice for a pilot — you need sufficient data density per user for the agent to reason about patterns. Discuss in limitations how results might change with lower-compliance users. Note that the 50 users yield ~3,900 prediction entries per version, which is substantial for temporal data.

### Objection 3: "The ML baselines are unfairly weak"
**Severity**: Medium-High.
**Preemption**: This is the most legitimate technical concern. The ML baselines use 5-fold CV on 399 users with daily aggregate features; the LLM evaluation uses 50 users with richer input. Acknowledge the asymmetry. Argue that (a) the daily aggregate features ARE what traditional ML would use in this setting, (b) the LLM's advantage is precisely that it can reason about richer representations without feature engineering, and (c) present the ML results as a "reference point" rather than a head-to-head comparison. If possible before submission, re-run ML baselines on the same 50 users for a fairer comparison.

### Objection 4: "LLM inference is too expensive/slow for real-time JITAI deployment"
**Severity**: Medium.
**Preemption**: Discuss compute requirements honestly. Note that (a) the prediction window in a JITAI is minutes, not milliseconds — you have time for a 10-30 second LLM call, (b) costs are dropping exponentially (cite current API pricing trends), (c) smaller models or distillation could maintain much of the performance at lower cost. Present a cost analysis per prediction.

### Objection 5: "How do we know the agent is doing anything meaningful with the tools, vs. just pattern matching on the prompt?"
**Severity**: Medium.
**Preemption**: This is where the 2x2 factorial is your strongest defense. The same model with the same data performs significantly worse when forced through a structured pipeline (Struct-Multi 0.603 vs. Auto-Multi 0.660). Include analysis of agent tool usage patterns — which tools are called most, how investigation strategies differ across users, and examples where the agent's investigation uncovered non-obvious patterns.

### Objection 6: "No analysis of WHEN the system works well vs. poorly"
**Severity**: Medium.
**Preemption**: Include per-user performance distributions and analyze what predicts good vs. poor performance (data density, user characteristics, target difficulty). IMWUT reviewers expect this kind of disaggregated analysis.

### Objection 7: "The 'diary paradox' claim needs stronger evidence"
**Severity**: Medium.
**Preemption**: This is a known issue in the EMA literature. Cite Shiffman et al. (2008, "Ecological Momentary Assessment") and work on MNAR in mobile health (JMIR 2021, "Data Missing Not at Random in Mobile Health Research"). Show compliance rates by affect level in the BUCS data if available — even a correlation between low compliance and high distress supports the claim.

### Objection 8: "Why not fine-tune or use domain-specific models?"
**Severity**: Low-Medium.
**Preemption**: Acknowledge this as future work. The current contribution is showing that off-the-shelf LLM reasoning with good tools and context is already competitive. Fine-tuning is complementary, not contradictory.

---

## Summary Assessment

This paper has a strong and timely story. The reactive-to-proactive transition is immediately compelling to anyone who has worked on mHealth interventions and hit the wall of "the people who don't respond to EMA are the ones who need us most." The 2x2 factorial design is methodologically elegant and unusual for IMWUT. The cancer survivor population gives clinical weight.

The main risks are (1) the retrospective-only evaluation, (2) the small-N concern, and (3) the ML baseline fairness question. All three can be managed with transparent acknowledgment and appropriate framing. Do not oversell this as a deployed system. Sell it as the most rigorous evaluation to date of whether LLM agents can provide the passive state assessment that JITAIs have always needed but never had.

From my perspective as someone who has spent decades trying to figure out when and how to deliver interventions to people who need them, THIS is the kind of work that could fundamentally change how we build adaptive interventions. The fact that a passive system can predict intervention availability better than a diary-based system (0.706 vs. 0.542) is, frankly, remarkable. That finding alone justifies the paper.
