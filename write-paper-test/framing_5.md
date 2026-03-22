# Framing Perspective 5: Shrikanth Narayanan (USC)

**Role**: University Professor, NAE member, Director of SAIL. Signal processing fundamentals, behavioral signal processing, multimodal behavior analysis.

**Stance**: Grounded in principled signal processing theory. Skeptical of approaches without theoretical justification. Values systematic decomposition, interpretable behavioral representations, and rigorous evaluation methodology.

---

## 1. Title (Ranked)

1. **"From Passive Sensing to Proactive Inference: Agentic LLM Reasoning over Multimodal Behavioral Signals for Predicting Emotional States and Intervention Receptivity in Cancer Survivors"**
   - Rationale: Captures the full signal processing chain (sensing -> inference -> prediction), emphasizes the behavioral signal interpretation angle, and names the clinical application domain explicitly. Long but appropriate for IMWUT norms.

2. **"PULSE: Autonomous LLM Agents for Behavioral Signal Interpretation and Affective State Prediction from Smartphone Sensing"**
   - Rationale: Concise, names the system, foregrounds the signal interpretation contribution. Slightly less specific about the clinical population.

3. **"Agentic Behavioral Signal Processing: LLM-Driven Investigation of Smartphone Sensing Data for Emotional State and Receptivity Prediction"**
   - Rationale: Explicitly bridges agentic AI and behavioral signal processing traditions. May be too conceptual for IMWUT reviewers who expect systems contributions.

My recommendation is **Title 1** for the full paper, using "PULSE" as the system name throughout the text. The title should telegraph that this is not merely "feed sensor data to GPT" — it is a principled pipeline from passive behavioral signals through autonomous agent reasoning to clinically meaningful predictions.

---

## 2. Core Contribution Framing

The story of this paper, in my view, is:

**Passive smartphone sensing data are rich in behavioral information but poor as direct predictive features — the gap between raw behavioral signals and affective state inference requires contextual interpretation that traditional ML pipelines cannot provide. This paper demonstrates that LLM agents, equipped with tool-based access to multimodal behavioral data streams and empowered to autonomously decide what evidence to investigate, can bridge this interpretation gap — achieving substantially higher prediction accuracy for emotional states and intervention receptivity in cancer survivors than either structured LLM pipelines or conventional ML approaches operating on the same behavioral signals.**

The key intellectual move is reframing the LLM not as a classifier but as a *behavioral signal interpreter* — an entity that performs the kind of contextual, multi-scale reasoning about human behavior that a clinician does when reading a patient's behavioral patterns. The agent's tool use (querying different time windows, comparing against peer cases, examining cross-modal consistency) mirrors the investigative process of behavioral signal analysis.

---

## 3. Positioning Relative to Existing Work

### The gap this paper fills

There is a clear gap at the intersection of three lines of work: (a) passive mobile sensing for mental health, (b) LLM-based health prediction, and (c) agentic AI with tool use. Existing work in (a) treats sensor features as inputs to fixed statistical pipelines. Work in (b) converts sensor data to text and prompts LLMs in a single pass. No prior work lets LLMs *autonomously investigate* behavioral data streams through tool use, deciding what to examine and how to reason about it.

### Key comparison papers (all verified)

1. **Wang et al., "StudentLife: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones," UbiComp 2014.**
   Authors: Rui Wang, Fanglin Chen, Zhenyu Chen, Tianxing Li, Gabriella Harari, Stefanie Tignor, Xia Zhou, Dror Ben-Zeev, Andrew T. Campbell.
   Venue: Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14), Seattle, WA.
   *Relevance*: Foundational mobile sensing + mental health work. Established the paradigm of correlating passively sensed smartphone features with psychological outcomes. PULSE extends this by replacing statistical correlation with autonomous agent-driven behavioral interpretation.

2. **Xu et al., "GLOBEM: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling," IMWUT 2022 (Distinguished Paper Award, UbiComp 2023).**
   Authors: Xuhai Xu, Xin Liu, Han Zhang, Weichen Wang, Subigya Nepal, Yasaman Sefidgar, Woosuk Seo, Kevin S. Kuehn, Jeremy F. Huckins, Margaret E. Morris, Paula S. Nurius, Eve A. Riskin, Shwetak Patel, Tim Althoff, Andrew Campbell, Anind K. Dey, Jennifer Mankoff.
   Venue: Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 6, 4, Article 190 (December 2022).
   *Relevance*: Benchmark showing that traditional ML generalizes poorly across populations and years for behavior modeling. Consolidates 19 algorithms; best methods still struggle with individual differences. PULSE's per-user agent memory and cross-user RAG directly address the personalization-generalization tension GLOBEM exposes.

3. **Xu et al., "Leveraging LLMs to Predict Affective States via Smartphone Sensor Features," UbiComp Companion 2024.**
   Authors: Matteo Hilty, Fausto Frassinelli, Elena Di Lascio, Shkurta Gashi, and others (University of Melbourne collaboration).
   Venue: Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing.
   *Relevance*: The most direct precursor — first to use LLMs for affective state prediction from smartphone sensing. However, it was small-scale (10 students), used single-pass prompting (zero-shot and few-shot on pre-formatted text), and treated the LLM as a static classifier. PULSE's factorial design, agentic tool use, cross-user RAG, and longitudinal memory represent a substantial advance in both methodology and scale.

4. **Kim et al., "Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data," CHIL 2024 (PMLR 248).**
   Authors: Yubin Kim, Xuhai Xu, Daniel McDuff, Cynthia Breazeal, Hae Won Park.
   Venue: Proceedings of the Fifth Conference on Health, Inference, and Learning, PMLR 248:522-539, 2024.
   *Relevance*: Comprehensive evaluation of LLMs for health prediction from wearable sensor data across 10 tasks. Demonstrates that contextual prompting improves performance up to 23.8%. PULSE extends this by moving from static prompting to autonomous agentic investigation — the LLM decides what context to gather rather than being given a fixed prompt.

5. **Wang et al., "CALLM: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models," arXiv 2503.10707 (under review, CHI 2026).**
   Authors: (same research group, same BUCS dataset).
   Venue: arXiv preprint, reported as under review for CHI 2026.
   *Relevance*: The direct predecessor and baseline. Uses diary text + RAG for reactive emotional state prediction. PULSE extends CALLM from reactive (requires diary) to proactive (works from passive sensing alone), and from structured single-call LLM to autonomous agentic investigation.

6. **Mishra et al., "Detecting Receptivity for mHealth Interventions in the Natural Environment," IMWUT 2021.**
   Authors: Varun Mishra, Florian Künzler, Jan-Niklas Kramer, Elgar Fleisch, Tobias Kowatsch, David Kotz.
   Venue: Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 5, 2, Article 1-24 (June 2021).
   *Relevance*: Key work on predicting intervention receptivity using ML on contextual features. Achieved up to 40% improvement over control. PULSE predicts both receptivity components (desire + availability) simultaneously with richer behavioral interpretation.

7. **Künzler et al., "Exploring the State-of-Receptivity for mHealth Interventions," IMWUT 2019.**
   Authors: Florian Künzler, Varun Mishra, Jan-Niklas Kramer, David Kotz, Elgar Fleisch, Tobias Kowatsch.
   Venue: Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 3, 4, Article 140 (December 2019).
   *Relevance*: Defined receptivity operationally and identified contextual predictors. Built ML models achieving up to 77% F1 improvement over baseline. PULSE builds on this conceptual framework by decomposing receptivity into desire and availability.

8. **Nahum-Shani et al., "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support," Annals of Behavioral Medicine, 2018.**
   Authors: Inbal Nahum-Shani, Shawna N. Smith, Bonnie J. Spring, Linda M. Collins, Katie Witkiewitz, Ambuj Tewari, Susan A. Murphy.
   Venue: Annals of Behavioral Medicine, 52(6), 446-462, 2018.
   *Relevance*: Foundational JITAI framework. Defines the components that PULSE implements: decision points (EMA windows), tailoring variables (sensing + diary), intervention options, and decision rules. PULSE operationalizes the "decision rule" as an autonomous LLM agent.

9. **Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023 (Notable Top 5%).**
   Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao.
   Venue: 11th International Conference on Learning Representations (ICLR 2023), Kigali, Rwanda.
   *Relevance*: Foundational framework for reasoning + acting in LLMs. PULSE's autonomous agent variants (Auto-Sense, Auto-Multi) implement a domain-specific version of ReAct where the "actions" are behavioral data queries and the "reasoning" is clinical-behavioral interpretation.

10. **Li et al., "Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-in-the-Loop LLM," IMWUT 2025.**
    Authors: Jiachen Li, Akshat Choube, and others (Northeastern University).
    Venue: Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. (2025).
    *Relevance*: Recent IMWUT work combining LLMs with multimodal sensing data for sensemaking. However, focuses on human-in-the-loop expert visualization rather than autonomous prediction. PULSE removes the human from the loop entirely — the agent performs the sensemaking autonomously.

11. **Narayanan et al., "Behavioral Signal Processing: Deriving Human Behavioral Informatics From Speech and Language," Proceedings of the IEEE, 2013.**
    Authors: Shrikanth Narayanan and Panayiotis G. Georgiou.
    Venue: Proceedings of the IEEE, 101(5), 1203-1233, 2013.
    *Relevance*: Establishes the theoretical framework of behavioral signal processing — deriving human behavioral informatics from multimodal signals. PULSE can be positioned as extending BSP principles to LLM-based behavioral interpretation, where the LLM performs the "deriving" step that has traditionally been done through hand-crafted feature engineering.

---

## 4. Key Claims (Supported vs. At Risk of Overclaiming)

### Claims the evidence supports:

1. **Agentic autonomy matters**: The 2x2 factorial design cleanly isolates that autonomous agent reasoning (Auto-Multi BA=0.660) substantially outperforms structured pipelines (Struct-Multi BA=0.603) on identical data. This is the cleanest and most defensible claim.

2. **Multimodality matters**: Adding diary text to sensing data significantly improves performance (Auto-Multi 0.660 vs. Auto-Sense 0.589). This holds across the factorial design.

3. **LLM agents outperform traditional ML**: Auto-Multi+ (0.661) substantially exceeds the best ML baseline (RF 0.518) on the same prediction tasks. The margin is large enough to be meaningful.

4. **Sensing data alone has value**: Auto-Sense (0.589) substantially outperforms random (0.50) and is competitive, demonstrating that passive sensing alone — without diary text — provides actionable signal for emotional state prediction. This is clinically important because diary compliance drops when it matters most.

5. **Behavioral signals inform availability more than text does**: Auto-Sense achieves 0.706 BA on INT_avail vs. CALLM's 0.542. Sensing data captures availability-relevant behavioral patterns (activity, location, screen use) that diary text misses.

6. **The "diary paradox"**: This is an important finding — the most informative modality (diary text) is missing precisely when intervention is most needed. The sensing-only agent provides a viable fallback.

### What NOT to overclaim:

1. **Do not claim real-time deployment readiness**: This is a retrospective simulation on historical data. The chronological ordering is principled, but real-world deployment introduces latency, missing data patterns, and user interaction effects not captured here.

2. **Do not claim the LLM "understands" behavioral signals**: The LLM performs contextual pattern matching and reasoning-by-analogy. Avoid language suggesting genuine comprehension of the underlying psychophysiology. I would frame it as "contextual behavioral signal interpretation" rather than "understanding."

3. **Do not overclaim on the ML baseline comparison**: The ML baselines use 5-fold CV on 399 users while LLM results are on 50 users. This asymmetry must be flagged. The comparison is still informative (the ML models have *more* data and still perform worse), but the evaluation conditions differ.

4. **Do not claim generalizability beyond cancer survivors**: The BUCS population has specific characteristics (diagnosis-related stress, treatment effects, high motivation). The method may or may not transfer to other populations.

5. **Do not claim cost-free scalability**: The current system runs on a Claude Max subscription. At scale, LLM inference costs and latency would be non-trivial. Acknowledge this as a deployment consideration.

6. **Do not overclaim on the "filtering is marginal" finding**: Auto-Multi+ (0.661) vs. Auto-Multi (0.660) is not a clean null result — it suggests the agent is already robust to noise, but the small sample (50 users) limits statistical power to detect small effects.

---

## 5. Narrative Arc

### Problem (Section 1-2): The Interpretation Gap in Behavioral Sensing

Open with the clinical need: cancer survivors experience fluctuating emotional states and receptivity to intervention, yet the window for timely support is narrow. Mobile sensing can capture behavioral signals continuously and passively, but a fundamental gap exists between raw behavioral features and clinically meaningful affective state predictions.

**The signal processing framing**: Traditional approaches extract fixed feature sets from sensor streams and feed them to classifiers. This discards the contextual, temporal, and cross-modal reasoning that clinicians use when interpreting behavioral patterns. The result is poor prediction performance (BA around 0.51-0.52 in our ML baselines and in prior GLOBEM benchmarks).

**The diary paradox**: Text-based approaches (CALLM) achieve better performance but require active user input — which drops precisely when intervention is most needed. We need a system that works proactively from passive signals alone.

### Approach (Section 3-4): Agentic Behavioral Signal Interpretation

Frame the LLM agent as a principled solution to the interpretation gap. The agent:
- Receives behavioral data streams through queryable tools (not pre-formatted text)
- Autonomously decides what evidence to investigate (which modalities, what time windows, what peer comparisons)
- Integrates multimodal behavioral signals with longitudinal memory and population context
- Produces interpretable reasoning traces alongside predictions

**The 2x2 factorial design** is the methodological backbone: {Structured, Autonomous} x {Sensing-only, Multimodal}. This cleanly isolates the contribution of (a) autonomous reasoning and (b) data modality.

Present the MCP tool architecture as domain-specific behavioral signal access — 8 tools covering different aspects of behavioral investigation (daily summary, temporal patterns, peer comparison, etc.).

### Results (Section 5-6): Evidence for Agentic Behavioral Interpretation

Lead with the factorial design results:
1. Autonomous >> Structured (the interpretation gap is real and agents close it)
2. Multimodal >> Sensing-only (text + sensing > either alone)
3. Agents >> ML baselines (contextual reasoning > feature engineering)
4. Sensing-only is viable (the diary paradox has a solution)

Then deep-dive into the clinically interesting targets:
- ER_desire and PA_State show the largest agent advantage
- INT_avail is uniquely well-served by sensing data
- NA_State shows consistent improvement across modalities

### Discussion (Section 7): Implications and Limitations

1. **What the agent does differently**: Analyze the reasoning traces to show how the agent investigates behavioral patterns — this is the "behavioral signal interpretation" story. Show examples of the agent noticing cross-modal inconsistencies, temporal patterns, or unusual deviations from baseline.

2. **Clinical implications**: The combination of desire prediction + availability prediction enables true receptivity detection. Discuss how this operationalizes the JITAI framework.

3. **Limitations**: Retrospective design, single population, LLM cost/latency, reproducibility concerns (stochastic LLM outputs), and the 50 vs. 399 user asymmetry in ML comparison.

4. **Future work**: Real-time deployment study, intervention delivery based on predictions, extending to other clinical populations, investigating what the agent actually attends to (interpretability analysis of reasoning traces).

---

## 6. What to Emphasize vs. De-Emphasize

### Emphasize:

- **The 2x2 factorial design**: This is methodologically rigorous and rare in LLM-based systems papers. It cleanly decomposes contributions. Reviewers will appreciate the experimental design discipline.

- **Agentic autonomy as the key differentiator**: The jump from Structured to Autonomous within the same data modality is the paper's strongest and most novel finding. Auto-Multi (0.660) vs. Struct-Multi (0.603) is a 9.5% absolute improvement from architecture alone.

- **INT_avail from sensing**: Auto-Sense (0.706) dramatically outperforming CALLM (0.542) on intervention availability is a striking result with clear clinical implications. Behavioral signals (activity patterns, screen use, location) directly indicate availability in ways diary text does not.

- **The clinical targets (ER_desire, INT_avail, PA_State, NA_State)**: Focus the results narrative on these four clinically meaningful constructs rather than spreading attention across all 16 binary targets.

- **Per-user analysis and distribution**: Show the distribution of per-user balanced accuracies, not just means. Demonstrate that the improvement is consistent across users, not driven by outliers.

- **Reasoning trace examples**: Selectively include 2-3 examples showing *how* the agent investigates behavioral data. This provides interpretability and distinguishes PULSE from "prompt-and-predict" approaches.

### De-emphasize:

- **The filtered variants (Auto-Sense+, Auto-Multi+)**: The marginal improvement from filtering is not a strong story. Mention it but do not make it a main finding. Frame it positively: "the agent is already robust to noisy data."

- **Absolute performance numbers in isolation**: BA of 0.66 is not spectacular on its own. The story is the *relative improvement* from the factorial design and the *comparison to baselines*.

- **The ML baseline comparison**: Important to include but do not over-index on it. The evaluation conditions differ (50 vs. 399 users, different CV procedures). Present it as "even with more data, fixed feature pipelines cannot match the agent's contextual reasoning."

- **Technical details of MCP/Claude CLI**: The specific LLM infrastructure is implementation detail. Focus on the abstract agent architecture (tool access, autonomous investigation, memory) rather than Anthropic-specific tooling.

- **Cost analysis**: Mention the Claude Max subscription as a proof of feasibility, but do not dwell on it. Reviewers may see reliance on a commercial API as a weakness.

---

## 7. Target Reader

### Primary audience: IMWUT/UbiComp researchers working on mobile sensing for health

These readers care about:
- Whether the system works with realistic, noisy, real-world sensing data
- Methodological rigor in evaluation (within-subject comparisons, statistical tests, effect sizes)
- Novelty of the system architecture and its implications for the field
- Reproducibility and deployability
- Clinical relevance and the path to real-world impact

They will be familiar with the StudentLife, GLOBEM, and receptivity prediction literature. They will be less familiar with agentic AI and MCP — this needs clear, self-contained explanation.

### Secondary audience: Affective computing and digital health researchers

These readers will evaluate whether the affective state prediction is principled (not just prompt engineering) and whether the results have clinical validity.

### What the reader should take away:

"LLM agents that autonomously investigate behavioral sensing data — deciding what to examine and how to reason about it — substantially outperform both structured LLM pipelines and traditional ML approaches for predicting emotional states and intervention receptivity. The agentic investigation paradigm represents a new approach to behavioral signal interpretation that could reshape how mobile sensing systems make inferences about human behavior."

---

## 8. Potential Reviewer Objections and Preemptions

### Objection 1: "This is just prompt engineering, not a scientific contribution."
**Preemption**: The 2x2 factorial design provides controlled scientific evidence. The contribution is not the prompt — it is the demonstration that autonomous agent reasoning over behavioral data (with tool use) systematically outperforms structured reasoning on identical data. The factorial design isolates this effect with statistical rigor. Furthermore, the tool-based architecture is a systematic framework, not ad hoc prompt tuning.

### Objection 2: "The evaluation is retrospective — how do we know this works in real time?"
**Preemption**: Acknowledge this limitation explicitly. Emphasize the strict chronological information boundary (the agent never sees future data). Frame the retrospective design as a necessary first step — similar to how GLOBEM and other behavioral modeling benchmarks are retrospective. Commit to future deployment studies.

### Objection 3: "50 users is too small. How generalizable is this?"
**Preemption**: Report the representativeness analysis (50 vs. 418 users). Show that base rates of key targets do not differ significantly (except NA_State, which has small effect size). Acknowledge that the 50 users were selected for high EMA compliance, and discuss what this means for generalizability. Emphasize that 50 users x ~78 entries/user = ~3,900 prediction instances per version — the per-prediction sample size is substantial.

### Objection 4: "The ML baselines are evaluated on 399 users while LLM is on 50. This is not a fair comparison."
**Preemption**: This is a real weakness. Address it head-on: the ML baselines have *more* training data (399 users in 5-fold CV), which should *advantage* them. The fact that they still underperform the LLM agent (which sees only 50 users' data) strengthens the argument. Commit to running ML baselines on the same 50 users for a direct comparison in the final version.

### Objection 5: "LLM outputs are stochastic and not reproducible."
**Preemption**: Report temperature settings and any measures taken for consistency. Acknowledge the inherent stochasticity but argue that the *systematic* advantage of autonomous over structured variants across multiple targets and users suggests the finding is robust. Consider running a reproducibility check (same inputs, multiple runs) on a subset.

### Objection 6: "What does the agent actually attend to? This is a black box."
**Preemption**: Include a qualitative analysis of agent reasoning traces. Show examples of what behavioral signals the agent investigates and how it synthesizes them. This is where PULSE's tool-use architecture is an advantage — every tool call is logged, providing a trace of the agent's investigation process. This is *more* interpretable than a neural network classifier.

### Objection 7: "The clinical significance of BA improvements is unclear."
**Preemption**: Frame results in terms of clinical utility. A system that correctly identifies intervention receptivity 71.6% of the time (Auto-Multi on INT_avail) vs. 54.2% (CALLM) means substantially more successful intervention opportunities. Calculate the number of additional correctly-timed interventions per user per week. Connect to the JITAI framework: even modest improvements in receptivity prediction translate to meaningful intervention delivery improvements.

### Objection 8: "Why not fine-tune a smaller model instead of using an expensive LLM with tool use?"
**Preemption**: The fine-tuning approach (as in Health-LLM) requires labeled sensor-affect pairs for training. PULSE's zero-shot approach works without task-specific training data — the LLM brings general behavioral reasoning capabilities. This is particularly important for clinical populations where labeled data is scarce. Additionally, the autonomous investigation strategy cannot be replicated by fine-tuning — it requires the ability to make sequential decisions about what data to examine.

### Objection 9 (from my perspective as a signal processing researcher): "Where is the theoretical grounding? Why should LLM-based behavioral interpretation work?"
**Preemption**: This is the deepest critique. I would advise grounding the approach in the behavioral signal processing framework: behavioral signals are information-bearing but require contextual interpretation. Traditional approaches use hand-crafted features that lose context. The LLM agent performs contextual interpretation by jointly considering temporal dynamics, cross-modal consistency, individual baselines, and population norms — all of which are established principles in behavioral signal analysis, now operationalized through natural language reasoning rather than fixed signal processing pipelines. The paper should be explicit about this theoretical connection.

---

## Summary Assessment

This is a strong systems paper with a clean experimental design. The 2x2 factorial is its methodological backbone — protect it at all costs. The clinical population (cancer survivors) and the receptivity decomposition (desire x availability) give it applied significance that pure ML papers lack.

The biggest risk is being perceived as "just threw data at GPT." Mitigate this by emphasizing the principled experimental design, the theoretical grounding in behavioral signal interpretation, and the systematic analysis of when and why the agent succeeds. The reasoning traces provide an interpretability advantage that should be exploited.

From a behavioral signal processing standpoint, the most interesting finding is not the aggregate numbers but the differential performance across targets — particularly that sensing data excels at availability prediction while diary text excels at desire/affect prediction. This cross-modal complementarity is principled and expected from a signal processing perspective, and the paper should develop this insight.
