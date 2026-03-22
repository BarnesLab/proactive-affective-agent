# Paper Framing Advisory — Tim Althoff Perspective

**Advisor**: Tim Althoff (Stanford Biomedical Data Science / UW Allen School)
**Expertise**: Computational health, large-scale behavioral data, empathic AI, statistical rigor
**Target venue**: IMWUT (Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)

---

## 1. Title Options (Ranked)

1. **PULSE: Agentic LLMs for Proactive Emotional State Prediction from Passive Smartphone Sensing in Cancer Survivors**
   - Captures the three key novelties: agentic architecture, proactive sensing, clinical population
   - "PULSE" is memorable and thematically resonant (sensing the "pulse" of emotional state)

2. **From Reactive Diaries to Proactive Sensing: LLM Agents That Autonomously Investigate Behavioral Data to Predict Emotional States**
   - Emphasizes the reactive-to-proactive narrative arc, which is the core conceptual shift
   - More descriptive of the contribution but longer

3. **Autonomous LLM Agents for Multimodal Emotional State Prediction: A Factorial Evaluation with Cancer Survivors' Smartphone Sensing Data**
   - Most technically precise; emphasizes the rigorous factorial evaluation design
   - Better if reviewers are expected to value methodological rigor over narrative

**Recommendation**: Option 1 for the main submission. It is concise, has a system name for citability, and signals the clinical domain immediately.

---

## 2. Core Contribution Framing

**THE story of this paper (2-3 sentences):**

Existing approaches to predicting emotional states in mobile health rely on users actively reporting (e.g., diary entries), creating a fundamental gap: the people who most need intervention are least likely to self-report. PULSE addresses this by deploying autonomous LLM agents that proactively investigate passive smartphone sensing data — deciding which behavioral signals to examine, how far back to look, and which cross-user comparisons to make — achieving balanced accuracy of 0.66 across 16 emotional and clinical targets in 50 cancer survivors, significantly outperforming both structured LLM pipelines (0.60, p < 10^-11) and traditional ML baselines (0.52). Through a 2x2 factorial design isolating agentic reasoning from data modality, we demonstrate that autonomous tool-using investigation is the primary driver of prediction quality, not simply having more data.

---

## 3. Positioning Relative to Existing Work

### The gap this paper fills

There is a clear trajectory in the literature: (a) passive sensing established behavioral correlates of mental health, (b) LLMs were introduced for text-based affective prediction, and (c) LLM agents emerged as autonomous reasoners with tool use. **No prior work combines all three: LLM agents that autonomously investigate passive sensing data for affective state prediction.** PULSE sits precisely at this intersection.

### Key comparison papers (all web-search verified)

**Direct predecessor (the "reactive" baseline this paper extends):**

- **CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models.**
  Wang et al. arXiv:2503.10707, 2025 (CHI 2026 submission).
  Uses LLMs + RAG over diary text to predict emotional states in cancer survivors (N=407, BUCS dataset). Reports BA up to 0.73 for individual constructs. *This is the direct baseline — PULSE uses the same dataset and extends from reactive diary to proactive sensing.*

**LLMs + smartphone sensing (the emerging space):**

- **Zhang, T., Teng, S., Jia, H., & D'Alfonso, S. "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features."**
  UbiComp 2024 Companion, Melbourne, Australia. ACM, 2024.
  First work to use LLMs for affective state prediction from smartphone sensors. Uses Gemini 1.5 Pro with zero/few-shot prompting on ~150 university students. *PULSE advances beyond this with autonomous agentic investigation (not just prompting), longitudinal memory, and a clinical population.*

- **Xu, W., Pillai, A., Nepal, S., Collins, A.C., Mackin, D.M., Heinz, M.V., Griffin, T.Z., Jacobson, N.C., & Campbell, A. "LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models."**
  arXiv:2512.23025, December 2025.
  Trains a patch-level encoder to project sensor time-series into LLM representation space for narrative generation. N=258. *LENS requires model fine-tuning; PULSE uses off-the-shelf LLMs with tool-based data access, making it more deployable.*

- **Feng, K., Sun, Z., Lee, R.K.-W., Jiang, X., Theng, Y.-L., & Ding, Y. "A Comparative Study of Traditional Machine Learning, Deep Learning, and Large Language Models for Mental Health Forecasting using Smartphone Sensing Data."**
  arXiv:2601.03603, January 2026.
  Benchmarks ML vs. DL vs. LLM on the CES dataset. Finds Transformers best (Macro-F1=0.58), LLMs strong in contextual reasoning but weak in temporal modeling. *PULSE's agentic architecture with memory and tool use addresses exactly the temporal modeling weakness they identify.*

**Foundational passive sensing work:**

- **Wang, R., Chen, F., Chen, Z., Li, T., Harari, G., Tignor, S., Zhou, X., Ben-Zeev, D., & Campbell, A.T. "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones."**
  UbiComp 2014. ACM.
  Seminal study establishing smartphone sensing for mental health assessment. N=48 Dartmouth students, 10-week term. *PULSE builds on this tradition but replaces statistical correlation with LLM-based reasoning.*

- **Saeb, S., Zhang, M., Karr, C.J., Schueller, S.M., Corden, M.E., Kording, K.P., & Mohr, D.C. "Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study."**
  JMIR, 17(7):e175, July 2015.
  Early demonstration that GPS and phone usage features correlate with PHQ-9 depression scores. N=28. *Establishes the behavioral signal that PULSE's agents learn to interpret.*

**JITAI and receptivity (the application context):**

- **Nahum-Shani, I., Smith, S.N., Spring, B.J., Collins, L.M., Witkiewitz, K., Tewari, A., & Murphy, S.A. "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support."**
  Annals of Behavioral Medicine, 52(6):446-462, 2018.
  Defines the JITAI framework. *PULSE provides the prediction engine that a JITAI system needs — predicting both desire and availability for intervention.*

- **Mishra, V., Kunzler, F., Kramer, J.-N., Kotz, D., Fleisch, E., & Kowatsch, T. "Detecting Receptivity for mHealth Interventions in the Natural Environment."**
  IMWUT, 5(2), 2021. (Distinguished Paper Award at UbiComp 2022.)
  Deploys ML models to predict receptivity for physical activity interventions. N=189, 6-week study. *PULSE's conceptualization of receptivity as desire AND availability is more nuanced, and uses LLM agents rather than traditional ML.*

- **Kunzler, F., Mishra, V., Kramer, J.-N., Kotz, D., Fleisch, E., & Kowatsch, T. "Exploring the State-of-Receptivity for mHealth Interventions."**
  IMWUT, 3(4):Article 140, 2019.
  Foundational exploration of contextual factors affecting user receptivity. *PULSE adds the dimension of emotional regulation desire to the receptivity construct.*

**LLM agents and tool use (the technical paradigm):**

- **Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. "ReAct: Synergizing Reasoning and Acting in Language Models."**
  ICLR 2023 (Oral).
  Introduces the reason-then-act paradigm for LLM agents. *PULSE's autonomous versions implement this paradigm for sensing data investigation.*

- **Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. "Voyager: An Open-Ended Embodied Agent with Large Language Models."**
  NeurIPS 2023.
  Demonstrates LLM agents with skill libraries and autonomous exploration in Minecraft. *PULSE applies analogous ideas (tool libraries, autonomous investigation, memory) to health sensing.*

---

## 4. Key Claims — What Can and Cannot Be Claimed

### Supported claims (with statistical evidence):

1. **Autonomous agentic reasoning significantly outperforms structured pipelines for the same data modality.** Auto-Multi vs. Struct-Multi: mean BA 0.646 vs. 0.604, Wilcoxon p < 10^-11, effect size r = 0.95. Auto-Sense vs. Struct-Sense: 0.548 vs. 0.509, p < 10^-10, r = 0.91. This is the paper's strongest and cleanest claim.

2. **Multimodal input (sensing + diary) significantly outperforms sensing alone when combined with agentic reasoning.** Auto-Multi (0.646) vs. Auto-Sense (0.548), a substantial gap. This establishes the value of diary text as complementary evidence.

3. **LLM agents significantly outperform traditional ML on the same sensing features.** Auto-Multi+ (0.661 mean BA) vs. best ML baseline RF (0.518). Even sensing-only agents (Auto-Sense, 0.589) beat all ML baselines.

4. **Passive sensing alone reaches clinically useful prediction levels for specific constructs.** Auto-Sense achieves BA=0.706 for intervention availability, outperforming diary-based CALLM (0.542). This supports the proactive argument: sensing detects behavioral availability that self-report misses.

5. **The pilot sample is representative of the larger BUCS population on the key clinical targets.** PA_State, ER_desire, INT_availability all show p > 0.05 for base rate comparisons (50 vs. 349 remaining users).

### Claims to make carefully (with caveats):

6. **Auto-Multi outperforms CALLM (diary-only).** Mean BA 0.646 vs. 0.611, Wilcoxon p < 10^-11, r = 0.93. This is statistically significant, but the practical margin (~3.5 percentage points in mean BA) is modest. Emphasize that Auto-Multi uses sensing+diary while CALLM uses diary alone — the comparison is about adding sensing + agentic reasoning, not replacing diary.

7. **INT_availability improvement over CALLM is not significant at p < 0.05.** Auto-Multi vs. CALLM on INT_availability: p = 0.107. This weakens the "proactive sensing beats reactive diary" narrative for this specific construct. Report honestly.

### What NOT to overclaim:

- **Do NOT claim real-time or deployed system performance.** This is retrospective replay on historical data. The system has never been tested in a live intervention setting.
- **Do NOT claim generalizability beyond cancer survivors with high EMA compliance.** The 50-user pilot was selected for high compliance (82 vs. 34 EMA entries). NA_State base rate differs significantly (p=0.028). Platform distribution also differs (p=0.027).
- **Do NOT claim the filtering variants (Auto-Sense+, Auto-Multi+) provide meaningful improvement.** Auto-Multi+ (0.661) vs. Auto-Multi (0.660) is negligible. This is actually a positive finding worth spinning: the agent already handles noise well.
- **Do NOT claim causal relationships** between sensing features and emotional states. All findings are correlational/predictive.
- **Do NOT extrapolate to other clinical populations** without qualification. Cancer survivors have distinct emotional trajectories.

---

## 5. Narrative Arc

### Problem (1-2 pages)
Mobile health interventions for cancer survivors depend on knowing when someone needs help and is available to engage. Current approaches require active self-report (diaries, EMA), but compliance is poorest when distress is highest — the "diary paradox." Passive smartphone sensing offers a path to proactive prediction, but traditional ML on sensor features performs poorly on small, heterogeneous clinical populations (BA ~0.52). Recent work shows LLMs can interpret text-based emotional reports, but how should an LLM reason about raw behavioral sensing data?

### Approach (4-5 pages)
We introduce PULSE, a system in which LLM agents autonomously investigate behavioral sensing data through purpose-built tools. Frame the 2x2 factorial design as the core methodological innovation: by crossing reasoning style (structured vs. autonomous) with data modality (sensing-only vs. multimodal), we can isolate exactly what drives prediction quality. Present the MCP tool architecture, cross-user RAG, and longitudinal session memory as enabling components. Emphasize that the agent *decides* what data to examine — it is not fed a fixed feature vector.

### Results (5-6 pages)
Lead with the factorial analysis — the 2x2 table is the centerpiece. Show that agentic reasoning is the dominant factor (large effect sizes). Then show multimodal > sensing-only. Then compare against CALLM and ML baselines. Present per-target analyses for the four focus constructs, highlighting the INT_availability finding (sensing beats diary). Include per-user variability analysis. Present the representativeness analysis transparently.

### Discussion (3-4 pages)
What does it mean that an LLM agent investigating behavioral data outperforms both human-designed feature engineering (ML) and structured LLM pipelines? Discuss the implications for JITAI design. Address the diary paradox directly. Discuss cost (free via Claude Max) and deployment feasibility. Be honest about limitations: retrospective replay, high-compliance subsample, N=50.

### Impact (1 page)
This work demonstrates a new paradigm for mobile health prediction: LLM agents as autonomous investigators of behavioral data. The approach could generalize beyond cancer to any population with passive sensing data.

---

## 6. What to Emphasize vs. De-Emphasize

### EMPHASIZE:

- **The 2x2 factorial design.** This is methodologically rare and powerful. Most papers in this space compare "our system" vs. baselines. You can isolate the effect of agentic reasoning independently of data modality. Reviewers will appreciate this.
- **Effect sizes, not just p-values.** The Wilcoxon effect sizes are enormous (r > 0.9 for several key comparisons). Report these prominently. They matter more than p-values for N=50.
- **Per-user variability.** Show distributions, not just means. IMWUT reviewers expect this. The per-user BA distributions tell a richer story than aggregate numbers.
- **The INT_availability finding** (sensing-only outperforms diary). This is the most clinically surprising and important result. It directly supports the proactive intervention argument.
- **The agent's investigation process.** Include qualitative examples of how the agent decides what data to examine. This is what makes the system contribution tangible and differentiates from simple prompting.
- **Representativeness analysis.** Presenting it transparently (including the significant differences) builds credibility. Most papers in this space don't do this.
- **The "diary paradox" framing.** This is the conceptual hook that makes the proactive approach necessary, not just novel.

### DE-EMPHASIZE:

- **Filtering variants (Auto-Sense+, Auto-Multi+).** The marginal improvement is negligible. Mention them briefly as robustness checks but do not present them as a contribution.
- **Absolute BA numbers in isolation.** A mean BA of 0.66 is meaningful but not spectacular. Always present in context of (a) the difficulty of the task (16 heterogeneous binary targets), (b) comparisons to baselines, and (c) per-target performance where some constructs reach 0.72-0.75.
- **ML baselines as a primary comparison.** The ML baselines are on 399 users with 5-fold CV, not the same 50 users. This is an apples-to-oranges comparison. Use them as a reference point, not a headline claim. Be explicit about this methodological difference.
- **The CALLM comparison for constructs where the advantage is not significant** (INT_availability specifically). Report it but do not build the narrative around it.
- **Cost.** Mentioning "free via Claude Max subscription" is fine in a practical note, but do not position cost as a contribution. Reviewers will correctly note that this depends on a commercial subscription that could change.

---

## 7. Target Reader

### Primary audience: IMWUT/UbiComp researchers working on:
- **Mobile sensing for health** — they care about: does passive sensing actually predict anything clinically meaningful? How much better than traditional ML? What new architectures are possible with LLMs?
- **Just-in-time adaptive interventions** — they care about: can we predict receptivity? Can we do it without burdening the user? How does this feed into intervention timing?
- **LLM-based sensing interpretation** — an emerging community that cares about: is agentic tool use better than prompting? How should LLMs interface with sensor data?

### Secondary audience:
- **Clinical researchers in cancer survivorship** — they care about: is this deployable? Does it respect the patient's autonomy and privacy? What's the clinical utility?
- **AI/NLP researchers exploring agent architectures** — they care about: does the agentic paradigm transfer to a real-world domain? What can we learn about tool-use effectiveness?

### What the primary reader cares about most:
1. Is this system actually useful for detecting intervention opportunities? (Clinical validity)
2. Is the evaluation rigorous and the sample representative? (Statistical rigor)
3. Can I adapt this approach to my sensing dataset? (Generalizability of the method)
4. What specifically makes the agent better than simpler approaches? (Ablation, the 2x2 design)

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "N=50 is too small. How do we know this generalizes?"
**Preemption:** Present the representativeness analysis upfront (Section 4/5). Show that base rates for 3/4 focus targets are not significantly different from the full population. Acknowledge the significant differences (NA_State, EMA count, platform) honestly. Frame N=50 as a rigorous pilot with 3,900+ predictions per version — the per-prediction sample size is substantial. Note that the IMWUT-awarded receptivity work by Mishra et al. used comparable sample sizes (N=189 total, but with far fewer predictions per user).

### Objection 2: "This is retrospective replay, not a real deployment."
**Preemption:** Acknowledge this explicitly in limitations. Emphasize the strict information boundary (agent never sees future data). Argue that retrospective replay is standard practice for developing JITAI prediction components (cite Nahum-Shani et al., 2018). Position this as necessary groundwork before a prospective study. Include a concrete plan for prospective evaluation in the discussion.

### Objection 3: "The ML baselines are not comparable (399 users, 5-fold CV vs. 50 users)."
**Preemption:** This is a real weakness. Acknowledge it explicitly. Present the ML baselines as reference points, not head-to-head comparisons. If possible before submission, rerun ML baselines on the same 50 users. Alternatively, argue that the ML baselines have a structural advantage (more training data) and still perform worse, making the comparison conservative.

### Objection 4: "Dependency on a commercial LLM (Claude) — reproducibility concerns."
**Preemption:** Report exact model versions. Discuss that the architecture is model-agnostic — the MCP tool interface could work with any LLM that supports tool use. Note that the structured vs. autonomous comparison provides insight into what capabilities matter, which is transferable. Acknowledge that exact numbers may vary with different models, but the architectural contribution (agentic tool use for sensing data) is the lasting one.

### Objection 5: "The improvement over CALLM is modest (~3.5 BA points). Is it worth the complexity?"
**Preemption:** Frame the comparison correctly: CALLM uses diary text, which PULSE also uses in its multimodal versions. The key comparison is not PULSE vs. CALLM but *what PULSE adds*: proactive capability (sensing-only still reaches 0.59) and significantly better performance for specific constructs (PA_State: 0.70 vs. 0.54; INT_availability: 0.56 vs. 0.55). The conceptual contribution is the shift from reactive to proactive, not just an accuracy improvement.

### Objection 6: "Why not fine-tune an LLM or train a specialized model instead of using expensive API calls?"
**Preemption:** Cite LENS (Xu et al., 2025) as the fine-tuning approach and compare: fine-tuning requires large paired sensor-text datasets and retraining for each new sensing modality or population. PULSE's tool-use approach is zero-shot adaptable — add a new sensor stream by adding a new tool. This is a fundamental architectural trade-off worth studying. Both approaches have merit; they serve different deployment scenarios.

### Objection 7: "High-compliance users are not the ones who need proactive intervention most."
**Preemption:** This is the strongest objection. Acknowledge it directly. The high-compliance selection was necessary to have sufficient ground truth for evaluation. Argue that (a) high compliance with EMA does not mean these users don't experience emotional distress — the within-user variability in emotional states is substantial, (b) the sensing-based prediction pipeline does not require EMA compliance at deployment time, only for training/evaluation, and (c) a prospective study with the full population is the natural next step.

### Objection 8: "How do you know the agent is actually 'investigating' in a meaningful way and not just using the system prompt?"
**Preemption:** Include a qualitative analysis section showing representative agent investigation trajectories. Show that the agent makes different tool calls for different users and contexts. Compare the investigation strategies of the autonomous agent vs. the structured pipeline to demonstrate emergent investigation behavior. This is where the system contribution becomes tangible.

---

## Summary of Strategic Recommendations

1. **Lead with the diary paradox and the proactive vision** — this is the hook that makes the paper necessary.
2. **Center the 2x2 factorial design** — this is the methodological strength that distinguishes the paper from a systems demo.
3. **Be rigorously honest about limitations** — N=50, retrospective, high-compliance subset, ML baseline comparability. Transparency builds credibility.
4. **Show the agent's work** — qualitative investigation trajectories are essential for an IMWUT paper. Reviewers need to see what "agentic" means concretely.
5. **Frame as a paradigm, not just a system** — the lasting contribution is the idea that LLM agents should investigate sensing data, not just receive it as input.
6. **Per-user analysis is non-negotiable** — IMWUT reviewers expect it. Show distributions, individual trajectories, and user-level variation.
