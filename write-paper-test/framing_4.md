# PULSE Paper Framing — Advisor 4: Andrew Campbell (Dartmouth)

*Perspective: Creator of StudentLife and MindScape. Albert Bradley 1915 Third Century Professor in CS, Dartmouth College. Expertise in mobile/ubiquitous sensing for health, LLM+sensing integration, longitudinal behavioral modeling.*

---

## 1. Title Options (Ranked)

1. **PULSE: Agentic LLMs That Investigate Smartphone Sensing Data to Predict Emotional States and Intervention Opportunities in Cancer Survivors**
   - Strongest: names the system, names the method ("agentic LLMs that investigate"), names the data modality, names the clinical population. IMWUT reviewers want to know immediately what the system does, what data it uses, and who it serves.

2. **From Passive Sensing to Proactive Support: LLM Agents That Autonomously Interpret Behavioral Data for Affective State Prediction**
   - More general framing. Good if you want to emphasize the paradigm shift (passive data → proactive inference) rather than the cancer domain. Risks being too broad for IMWUT reviewers who want specificity.

3. **Agentic Behavioral Sensing: Autonomous LLM Investigation of Smartphone Data for Emotional State Prediction**
   - Cleanest technically but loses the clinical grounding. I would not recommend leading with this for IMWUT --- the clinical population is part of what makes this work matter.

**My recommendation: Option 1.** IMWUT values systems that solve real problems for real people. The cancer survivor population and the clinical constructs (emotion regulation desire, intervention availability) are not just evaluation targets --- they are the reason this system exists.

---

## 2. Core Contribution Framing

**The story of this paper in three sentences:**

We introduce the paradigm of *agentic sensing investigation* --- instead of engineering features from smartphone data and feeding them to a classifier, we equip an LLM agent with tools to autonomously query, compare, and reason over raw behavioral sensing streams, deciding for itself what evidence matters for each prediction. Through a 2x2 factorial evaluation (structured vs. autonomous reasoning x sensing-only vs. multimodal) on 50 cancer survivors from the BUCS study, we demonstrate that autonomous agentic investigation of sensing data (BA=0.660) substantially outperforms both structured LLM pipelines (0.603) and traditional ML baselines (0.518), while revealing that sensing alone can predict intervention availability better than diary text. This work establishes that the next frontier in mobile health sensing is not better features or bigger models, but giving AI agents the autonomy to investigate behavioral data the way a clinician would --- adaptively, contextually, and with access to the right tools.

---

## 3. Positioning Relative to Existing Work

### The Landscape This Paper Enters

This work sits at the intersection of three active research threads that have, until now, been largely separate:

**Thread 1: Passive sensing for mental health (the classic ubicomp thread)**

The foundational work here is our StudentLife system (Campbell et al., "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones," UbiComp 2014, UbiComp 10-Year Impact Award 2024), which established that passively sensed smartphone data --- sleep, movement, social interaction --- correlates with mental health outcomes. Since then, hundreds of studies have followed this paradigm: extract features from sensor data, train ML classifiers, predict some mental health outcome. Saeb et al. ("Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study," JMIR 2015) showed GPS and phone usage features correlate with depression severity. The GLOBEM benchmark (Xu et al., "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling," IMWUT 2022) revealed a hard truth: traditional ML on sensing features barely generalizes across datasets, with domain generalization algorithms performing barely above majority-class guessing.

**The gap PULSE fills in Thread 1:** Traditional ML on sensing features has hit a ceiling. GLOBEM showed generalization fails. PULSE's agentic approach bypasses the feature-engineering bottleneck entirely --- the agent reasons directly over behavioral data rather than operating on fixed feature vectors.

**Thread 2: LLMs for health sensing and digital phenotyping (the emerging thread)**

This is very new. Zhang et al. ("Leveraging LLMs to Predict Affective States via Smartphone Sensor Features," UbiComp Companion 2024) were the first to use LLMs (Gemini 1.5 Pro) for affective state prediction from smartphone sensor data, using zero-shot and few-shot prompting on ~10 students. Kim et al. (Health-LLM: "Large Language Models for Health Prediction via Wearable Sensor Data," CHIL/PMLR 2024) evaluated 12 LLMs on wearable sensor data across 10 health prediction tasks. Wang et al. ("Efficient and Personalized Mobile Health Event Prediction via Small Language Models," MobiCom 2024) explored small language models for on-device health prediction. Xu et al. (LENS: "LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models," arXiv 2512.23025, 2025) constructed a pipeline for translating sensor time series into clinically grounded mental health narratives, yielding 100K+ sensor-text pairs from 258 participants.

**The gap PULSE fills in Thread 2:** All existing LLM+sensing work uses LLMs as *passive reasoners* --- you format the data, stuff it into a prompt, and ask for a prediction. PULSE is fundamentally different: the agent has *tools* and *autonomy*. It decides what data to examine, what comparisons to make, what historical context to retrieve. This is the difference between giving someone an answer sheet and giving them a laboratory.

**Thread 3: Proactive intervention and receptivity prediction (the clinical thread)**

Nahum-Shani et al. ("Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support," Annals of Behavioral Medicine, 2018) laid the theoretical foundations for JITAIs. Our group's work on detecting receptivity (Mishra, Kunzler, Kramer, Fleisch, Kowatsch, & Kotz, "Detecting Receptivity for mHealth Interventions in the Natural Environment," IMWUT 2021, Distinguished Paper Award) showed that ML models on sensor features could improve receptivity-based intervention timing by up to 40%. The predecessor to PULSE is CALLM (Wang et al., "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models," arXiv 2503.10707, CHI 2026), which used LLM+RAG on diary text to predict emotional states in cancer survivors.

**The gap PULSE fills in Thread 3:** CALLM requires the user to write diary entries --- but the diary paradox means entries are missing precisely when the user needs help most. PULSE's sensing-only variants (Auto-Sense BA=0.589) show you can predict intervention availability (BA=0.706) *without any active user input*, and the full multimodal system (Auto-Multi BA=0.660) exceeds CALLM (0.611) even when diary text is available.

### Key Comparison Papers (all verified)

| Paper | Venue | Year | Relationship to PULSE |
|-------|-------|------|----------------------|
| Campbell et al., StudentLife | UbiComp | 2014 | Foundational passive sensing paradigm |
| Saeb et al., Mobile phone sensor correlates | JMIR | 2015 | Early digital phenotyping for depression |
| Nahum-Shani et al., JITAIs | Annals Behav Med | 2018 | Theoretical framework for intervention timing |
| Mishra et al., Detecting Receptivity | IMWUT | 2021 | Receptivity prediction via ML on sensors |
| Xu et al., GLOBEM | IMWUT | 2022 | Reveals generalization failure of traditional ML |
| Zhang et al., LLMs for Affective States | UbiComp Companion | 2024 | First LLM+sensing for affect (zero-shot, N=10) |
| Kim et al., Health-LLM | CHIL/PMLR | 2024 | LLM benchmarking on wearable health data |
| Wang et al., Small Language Models for mHealth | MobiCom | 2024 | Efficient on-device LLM health prediction |
| Nepal et al., MindScape | IMWUT | 2024 | LLM+sensing for journaling intervention |
| Xu et al., LENS | arXiv | 2025 | LLM narrative synthesis from multimodal sensing |
| Wang et al., CALLM | CHI | 2026 | Direct predecessor; diary-only LLM for same population |

---

## 4. Key Claims (What You Can and Cannot Claim)

### Claims Supported by Evidence

1. **Agentic investigation > structured pipelines.** The 2x2 factorial cleanly isolates this: Auto-Multi (0.660) vs. Struct-Multi (0.603) on the same data. Auto-Sense (0.589) vs. Struct-Sense (0.516). The agent deciding its own investigation strategy produces substantially better predictions than a fixed pipeline, even with the same underlying LLM and data access. This is the headline result.

2. **Multimodal > sensing-only, but sensing alone is viable.** Auto-Multi (0.660) > Auto-Sense (0.589), confirming diary text adds signal. But Auto-Sense alone beats CALLM's diary-only approach (0.611 vs. 0.589 on mean BA across all targets, and critically 0.706 vs. 0.542 on INT_availability). Sensing-only predictions are clinically actionable.

3. **LLM agents > traditional ML on sensing features.** Auto-Multi+ (0.661) and Auto-Sense (0.589) both exceed the best traditional ML baseline (RF at 0.518). This matters because it suggests LLM reasoning over raw behavioral data captures signal that feature engineering misses.

4. **The diary paradox is real and sensing addresses it.** The paper can claim that passive sensing enables prediction precisely when active reporting fails. The INT_availability finding (Auto-Sense 0.706 >> CALLM 0.542) is the killer result here --- whether someone is *available* for an intervention is fundamentally a behavioral/contextual signal, not a self-report signal.

5. **Cross-user RAG provides empirical grounding.** The agent does not just reason in a vacuum; it retrieves similar cases from the population. This is a legitimate contribution to the "LLM hallucination" concern in health applications.

### What NOT to Overclaim

1. **Do not claim this replaces clinical assessment.** BA of 0.66 is promising but not clinical-grade. Frame as "decision support" and "pre-screening for intervention timing," never as diagnosis or treatment.

2. **Do not overclaim generalizability.** N=50 is a pilot within a specific population (cancer survivors, high-compliance, recruited from BUCS). The 50 users were selected for high EMA compliance (82 vs. 34 responses), and there is a slight NA_State distributional shift (p=0.028). Be transparent about this.

3. **Do not claim the ML baselines are definitive.** They use 5-fold CV on 399 users, not the same 50. Acknowledge this mismatch. The ML baselines serve as reference points for "what traditional approaches achieve on this data," not as a strict apples-to-apples comparison.

4. **Do not claim real-time deployment readiness.** This is a retrospective replay simulation. The system has not been deployed live. The cost and latency of running agentic LLM calls per-user per-EMA-window have not been characterized for real-time use.

5. **Do not overstate the filtering contribution.** Auto-Multi+ (0.661) is essentially indistinguishable from Auto-Multi (0.660). The filtering variants add complexity without clear benefit. Report honestly; don't try to make this a contribution.

---

## 5. Narrative Arc

### Problem (1.5 pages)
Open with the fundamental tension in mobile health: we can sense behavior passively, but making sense of noisy, incomplete, multimodal behavioral data to predict psychological states remains the bottleneck. Traditional ML on engineered features has hit a ceiling (cite GLOBEM). Active self-report (diary, EMA) captures rich signal but suffers from the diary paradox: entries are missing when the user most needs support. Cancer survivors face acute emotional challenges where timely intervention matters. The JITAI framework promises right-support-at-the-right-time, but predicting *when* someone is both in need and available remains unsolved.

### Insight (0.5 pages)
The key insight: clinicians do not process patient data through fixed feature pipelines. They *investigate* --- they decide what to look at, how far back to go, what comparisons to make, based on what they see. Modern LLM agents with tool-use capabilities can mimic this investigative process. Rather than formatting sensing data into feature vectors, we give the agent tools to query behavioral streams and let it decide its own investigation strategy.

### System (4-5 pages)
Present the PULSE architecture: MCP-based sensing query tools (8 tools), per-user session memory with self-reflections, cross-user RAG for population calibration, and the 2x2 design (structured vs. autonomous x sensing vs. multimodal). Walk through a concrete example of the agent investigating one user's data --- show the tool calls, the reasoning, the prediction. This is where the paper lives or dies for IMWUT; the system description must be concrete and reproducible.

### Evaluation (5-6 pages)
Present the 2x2 factorial results. Lead with the headline: agentic investigation outperforms structured pipelines across all comparisons. Then unpack: multimodal vs. sensing-only, LLM vs. ML baselines, per-target analysis (highlight INT_availability and ER_desire), per-user variability. Include the representativeness analysis (50 vs. 418) upfront --- do not bury it.

### Discussion (2-3 pages)
The diary paradox and what it means for JITAI systems. The implications of agentic sensing investigation as a new paradigm for ubicomp. Limitations (N=50, retrospective, cost, no real-time deployment). Future work: real-time deployment, intervention delivery, scaling to full cohort.

---

## 6. What to Emphasize vs. De-emphasize

### Emphasize

- **The 2x2 factorial design.** This is the methodological backbone. It cleanly isolates the effect of autonomy (structured vs. agentic) from the effect of data modality (sensing vs. multimodal). Very few LLM+sensing papers have this kind of controlled comparison. Reviewers will respect this.

- **The INT_availability result (Auto-Sense 0.706 >> CALLM 0.542).** This is the single most compelling finding. It shows that *whether someone is available for intervention* is better predicted by behavioral sensing than by what they write in a diary. This directly validates the passive sensing paradigm for JITAI timing.

- **Concrete examples of agentic investigation.** Show what tool calls the agent makes, how it reasons about conflicting signals, how it uses cross-user comparisons. IMWUT reviewers want to see *how the system works*, not just that it works. Include at least 2-3 detailed case studies of the agent's reasoning process.

- **The representativeness analysis.** Being upfront about the 50 vs. 418 comparison, including the NA_State distributional shift, builds credibility. Reviewers will check this; putting it in the main text (not just supplementary) signals confidence.

- **Per-target analysis.** The four focus targets (ER_desire, INT_availability, PA_State, NA_State) tell different stories. ER_desire and PA_State show the largest agentic gains. INT_availability is uniquely sensing-predictable. This granularity matters.

### De-emphasize

- **The filtering variants (Auto-Sense+, Auto-Multi+).** The gains are negligible (0.660 vs. 0.661). Mention these in a paragraph; do not give them a full subsection. The story is cleaner without them as a headline.

- **Absolute accuracy numbers in isolation.** BA of 0.66 sounds modest to someone unfamiliar with EMA prediction. Always contextualize: (a) relative to baselines, (b) relative to the difficulty of the task (predicting psychological states from passively sensed behavior), (c) relative to prior work on similar populations.

- **The cost/infrastructure details.** The Claude Max subscription setup is interesting but tangential. Mention it in the implementation section; don't make it a selling point. Reviewers may be skeptical of reproducibility tied to a specific commercial subscription tier.

- **The ML baseline comparison.** Useful as context but since it is 5-fold CV on 399 users (not the same 50), it is not a strict comparison. Present it honestly as "reference performance of traditional ML on this dataset" and move on.

---

## 7. Target Reader

**Primary audience: IMWUT/UbiComp researchers working on mobile sensing for health and well-being.**

These readers care about:
- Whether passive sensing can actually predict clinically meaningful psychological constructs (many are skeptical after years of modest results)
- System architecture and reproducibility (they want to build on your work)
- Evaluation rigor --- controlled comparisons, not just "our system does well"
- Clinical relevance --- does this matter for real patients?
- The paradigm shift from feature engineering to LLM reasoning over sensing data

**Secondary audience: Affective computing and health informatics researchers** who are exploring LLMs for clinical prediction but come from NLP/ML backgrounds rather than ubicomp. They care about prompt design, RAG architecture, and how LLMs handle noisy real-world data.

**Tertiary audience: Clinical researchers in cancer survivorship and JITAI design** who want to understand how AI can improve intervention timing. They care less about the technical architecture and more about: does this work, is it feasible, and can it be deployed?

---

## 8. Potential Reviewer Objections and Preemptive Strategies

### Objection 1: "N=50 is too small for IMWUT"
**Preempt:** Be explicit that this is a pilot within a larger study (N=418). Frame the 50 users as a rigorous feasibility evaluation, not as the final word. Show the representativeness analysis (base rates match on 3/4 targets, EMA count higher by design). Emphasize that 50 users x ~78 EMA entries each = ~3,900 prediction instances per version, which provides substantial statistical power for within-subject comparisons. The BUCS dataset itself (418 cancer survivors, 5-week study, 8 sensing modalities) is a large-scale clinical dataset; the 50-user pilot is appropriately scoped for the agentic LLM evaluation, which is computationally expensive.

### Objection 2: "This is just prompt engineering, not a system contribution"
**Preempt:** The contribution is not prompt engineering --- it is the architecture of agentic investigation over sensing data. The 8 MCP tools, the cross-user RAG, the session memory, the 2x2 factorial design: these constitute a system. The key finding (agentic > structured) demonstrates that the *investigative autonomy* matters, not just the prompt text. Show concrete tool-call traces to make the system tangible.

### Objection 3: "The ML baselines are not on the same 50 users"
**Preempt:** Acknowledge this explicitly and early. Note that the ML baselines were computed on the full 399-user set with 5-fold CV, making them a generous reference (more training data). If anything, the comparison is *favorable* to ML. Offer to release code and data for exact replication on the 50-user subset.

### Objection 4: "Retrospective replay is not real deployment"
**Preempt:** Agree, and discuss this as a limitation. Emphasize the strict information boundary (agent only sees data before the prediction window). Note that retrospective evaluation is standard in the mobile sensing literature (cite StudentLife, GLOBEM, and virtually all EMA prediction papers). Outline concrete plans for real-time deployment as future work.

### Objection 5: "How reproducible is this? You're using Claude via a commercial subscription"
**Preempt:** Discuss reproducibility explicitly. The MCP tool interface is model-agnostic --- the same tools could wrap any LLM with function-calling capability. The structured variants provide a direct comparison point for non-agentic LLM use. Release the tool definitions, prompts, and evaluation pipeline as open source. Note that the agentic paradigm is the contribution, not the specific model backend.

### Objection 6: "BA of 0.66 is not clinically actionable"
**Preempt:** Contextualize against prior work. CALLM achieved 0.611 on the same constructs. Zhang et al.'s LLM+sensing work on a simpler population showed comparable or lower accuracy. The GLOBEM benchmark showed traditional ML barely exceeds 0.51 on cross-dataset depression detection. More importantly, in a JITAI context, the agent does not need to be perfect --- it needs to be *better than random timing* of intervention delivery, which this clearly achieves. The 0.706 BA on INT_availability means the system correctly identifies availability >70% of the time, which is operationally useful.

### Objection 7: "What about LLM stochasticity? Are these results stable?"
**Preempt:** Report confidence intervals from bootstrap. Discuss whether multiple runs produce consistent results. The factorial design itself helps --- the relative ordering (agentic > structured) should be robust even if absolute numbers vary across runs. If you have multiple-run data, include it.

### Objection 8: "MindScape already does LLM + sensing"
**Preempt:** MindScape (Nepal, Pillai, Campbell et al., IMWUT 2024) uses LLM+sensing for *generating journaling prompts* --- it is an intervention delivery system, not a prediction system. PULSE uses LLM agents for *predicting emotional states and intervention opportunities* --- fundamentally different goals. MindScape does not give the LLM autonomy to investigate data; it uses sensing data as context for prompt generation. PULSE's agentic investigation paradigm is architecturally distinct.

---

## Summary: The Paper in One Paragraph

PULSE demonstrates that giving LLM agents tools to autonomously investigate smartphone sensing data --- rather than pre-engineering features or stuffing formatted data into prompts --- produces substantially better predictions of emotional states and intervention opportunities in cancer survivors. The 2x2 factorial design cleanly isolates the benefit of agentic autonomy (structured vs. autonomous) and data modality (sensing vs. multimodal), showing that autonomous investigation of multimodal data achieves BA=0.660, outperforming structured pipelines (0.603), the diary-only CALLM baseline (0.611), and traditional ML (0.518). The most striking finding is that sensing-only agents predict intervention availability (BA=0.706) far better than diary-based approaches (0.542), validating passive sensing as a viable path to proactive just-in-time adaptive interventions --- especially when users cannot or do not self-report.

---

*Prepared by Andrew Campbell, Dartmouth College. Ready for panel discussion.*
