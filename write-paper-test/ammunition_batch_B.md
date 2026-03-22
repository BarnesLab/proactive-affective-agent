# Ammunition Round — Batch B (Experts 6-10)

## Expert 6: Tim Althoff (UW → Stanford)
**Domain**: Computational health, large-scale behavioral data, statistical rigor

---

### Section 1: Introduction

**Key arguments from Althoff's perspective:**
- Large-scale behavioral data from smartphones and wearables can reveal health insights invisible at small scale — but only with rigorous statistical methodology
- The gap between sensing data collection and actionable health inference is the central challenge; PULSE addresses this by replacing fixed feature pipelines with agentic investigation
- Population-level behavioral data studies consistently show modest effect sizes; the 2x2 factorial design provides the statistical rigor needed to isolate the agentic reasoning contribution

**Evidence to cite:**
- Agentic vs. structured effect: Auto-Multi (0.660) vs. Struct-Multi (0.603), p < 10^-10, r > 0.9
- ML baselines (RF 0.518, XGBoost 0.514, Logistic 0.515) — ceiling of traditional feature engineering
- 50 users, ~3,900 entries per version — sufficient power for within-subject comparisons

**Related work to reference:**
- Althoff et al., Nature 2017 — large-scale physical activity data revealing population-level patterns
- Hicks, Althoff et al., npj Digital Medicine 2019 — best practices for wearable data analysis
- Xu, Althoff et al., IMWUT 2022 — GLOBEM benchmark showing generalization challenges

**Phrasing suggestions:**
- "...move beyond fixed feature extraction to autonomous behavioral data investigation..."
- "...the 2x2 factorial design cleanly isolates the contribution of agentic reasoning from data modality..."
- "...traditional ML on engineered features from sensing data has plateaued near chance (BA ~0.52)..."

**Pitfalls to avoid:**
- Do NOT claim the ML baselines are a fair head-to-head comparison (399 users vs. 50 users)
- Do NOT overstate effect sizes without proper confidence intervals and bootstrap CIs
- Do NOT conflate statistical significance with clinical meaningfulness

**Reviewer preemption:**
- Preempt: "Why not more ML baselines?" — Frame ML as reference points for the sensing feature ceiling, not direct comparisons. Emphasize the factorial design as the primary evaluation mechanism.
- Preempt: "N=50 is small" — Present representativeness analysis showing p > 0.05 for 3/4 key targets; emphasize within-subject design with ~3,900 entries per version; cite Althoff's work on best practices for analyzing large-scale data.

---

### Section 2: Related Work

**Key arguments:**
- Position PULSE within the arc from passive sensing (StudentLife, Saeb) → benchmarks (GLOBEM) → LLM-based inference (Health-LLM, Mental-LLM) → agentic investigation (PULSE)
- GLOBEM showed that cross-dataset generalization is fundamentally hard with fixed features; PULSE's agentic approach may sidestep this by letting the LLM adaptively select which signals matter
- The statistical rigor gap: most sensing papers report accuracy without proper per-user distributions, effect sizes, or bootstrap CIs

**Evidence to cite:**
- GLOBEM: 19 methods tested, domain generalization algorithms "barely any advantage over naive baseline"
- PULSE per-user BA distributions available for all version pairs — exemplary reporting

**Related work to reference:**
- Wang et al. (StudentLife, UbiComp 2014) — foundational passive sensing
- Saeb et al. (JMIR 2015) — GPS correlates of depression
- Xu et al. (GLOBEM, IMWUT 2022) — generalization benchmark
- Althoff et al. (Nature 2017) — population-scale behavioral data
- Hicks, Althoff et al. (npj Digital Medicine 2019) — best practices for wearable analysis

**Phrasing suggestions:**
- "...prior benchmarks reveal a generalization ceiling for fixed-feature approaches..."
- "...PULSE shifts the locus of intelligence from feature engineering to autonomous investigation..."

**Pitfalls to avoid:**
- Do NOT dismiss ML baselines as useless — they represent decades of important work
- Do NOT claim PULSE "solves" the generalization problem; it is a new paradigm, not yet tested cross-dataset

**Reviewer preemption:**
- Preempt: "How does this relate to GLOBEM?" — GLOBEM tests fixed-feature generalization; PULSE tests whether agentic reasoning can overcome the fixed-feature ceiling within a single dataset. Cross-dataset testing is future work.

---

### Section 3: System Design

**Key arguments:**
- The 2x2 factorial design is the system's methodological backbone — cleanly isolates agentic reasoning (rows) from data modality (columns)
- 7 versions including CALLM baseline enable multiple comparison axes
- 5-fold across-subject CV prevents information leakage; temporal boundary enforcement ensures no future data contamination

**Evidence to cite:**
- 5-fold CV on 418 users → 50 test users evaluated
- 8 sensing modalities with platform-specific coverage (iOS vs Android)
- ~3,900 entries per version — sufficient statistical power

**Phrasing suggestions:**
- "...the factorial design enables clean causal attribution — the agentic architecture, not merely richer data, drives the performance gain..."
- "...cross-user RAG provides empirical calibration, grounding predictions in population-level evidence..."

**Pitfalls to avoid:**
- Do NOT undersell the statistical design — it is a major contribution
- Do NOT forget to justify why 50 users is sufficient (within-subject, high-compliance selection)

**Reviewer preemption:**
- Preempt: "Why only 50 users?" — Selected for high-compliance (82 vs 34 EMA mean); representativeness analysis shows no significant difference on 3/4 key targets; within-subject design with thousands of entries per version. Present representativeness in main text, not supplementary.

---

### Section 4: Evaluation Methodology

**Key arguments:**
- Balanced accuracy as primary metric is correct for imbalanced health data
- Per-user BA distributions (not just means) reveal individual variation
- Wilcoxon signed-rank tests are appropriate for paired, non-normal distributions
- Bootstrap CIs provide distribution-free uncertainty quantification

**Evidence to cite:**
- Statistical tests: Wilcoxon signed-rank with bootstrap CIs in statistical_tests.json
- Per-user BA distributions available for all version pairs
- Representativeness: PA_State, ER_desire, INT_avail base rates p > 0.05; NA_State p = 0.028 (small effect r = 0.19)

**Phrasing suggestions:**
- "...we report effect sizes (r) alongside p-values, following best practices for behavioral data analysis..."
- "...per-user balanced accuracy distributions reveal that the agentic advantage is consistent, not driven by outliers..."

**Pitfalls to avoid:**
- Do NOT rely solely on p-values — emphasize effect sizes and confidence intervals
- Do NOT ignore the ML baseline evaluation asymmetry (399 vs 50 users) — flag it explicitly

**Reviewer preemption:**
- Preempt: "Unfair ML comparison" — Acknowledge explicitly; frame ML as reference points for the sensing feature ceiling, not head-to-head competitors. The primary comparison is within the factorial design.

---

### Section 5: Results

**Key arguments:**
- Lead with the 2x2 factorial result: agentic >> structured across both modality conditions
- Effect sizes (r > 0.9) are remarkably large for behavioral data — contextualize against typical r values in the field
- Per-user distributions show the advantage is consistent, not driven by a few outliers

**Evidence to cite:**
- Agentic vs. Structured: Auto-Multi 0.660 vs. Struct-Multi 0.603; Auto-Sense 0.589 vs. Struct-Sense 0.516
- p < 10^-10, r > 0.9 (Wilcoxon signed-rank)
- INT_avail: Auto-Sense 0.706 >> CALLM 0.542 — sensing alone outperforms diary
- Filtering marginal: Auto-Multi+ 0.661 ≈ Auto-Multi 0.660

**Phrasing suggestions:**
- "...the effect size (r > 0.9) is exceptionally large by the standards of behavioral sensing research, where effect sizes of r = 0.3-0.5 are typical..."
- "...the minimal marginal gain from filtering (0.661 vs. 0.660) suggests that agentic agents are inherently robust to noisy input..."

**Pitfalls to avoid:**
- Do NOT overclaim filtering adds value — the difference is negligible
- Do NOT present aggregate BA without per-user distributions

**Reviewer preemption:**
- Preempt: "Are these effect sizes inflated?" — Report per-user distributions showing consistent advantage, not outlier-driven. Provide bootstrap CIs.

---

### Section 6: Discussion

**Key arguments:**
- Connect to BSP framework (Narayanan 2013): PULSE performs contextual behavioral signal interpretation analogous to a clinician's chart review
- The INT_avail finding (sensing >> diary) has methodological implications: behavioral constructs should be measured with behavioral data
- Cross-user RAG as calibration (not knowledge retrieval) is a novel methodological contribution worth emphasizing

**Evidence to cite:**
- INT_avail: Auto-Sense 0.706 vs. CALLM 0.542 — the behavioral construct finding
- Continuous calibration: negativity bias for NA, mean regression for PA
- Binary robust despite continuous miscalibration

**Phrasing suggestions:**
- "...the robust binary classification despite poor continuous calibration mirrors findings in large-scale behavioral data analysis where relative ordering outperforms absolute prediction..."
- "...population-level calibration via cross-user RAG serves a function analogous to empirical Bayes estimation..."

**Pitfalls to avoid:**
- Do NOT claim deployment readiness without real-time testing
- Do NOT generalize beyond cancer survivors without caveats

**Reviewer preemption:**
- Preempt: "Model dependency on Claude" — Acknowledge; propose cross-model comparison as future work. Emphasize that the paradigm contribution (agentic investigation) is model-agnostic.

---

### BibTeX References (Althoff-relevant)

```bibtex
@article{Althoff2017Nature,
  author = {Althoff, Tim and Sosi\v{c}, Rok and Hicks, Jennifer L. and King, Abby C. and Delp, Scott L. and Leskovec, Jure},
  title = {Large-scale physical activity data reveal worldwide activity inequality},
  journal = {Nature},
  volume = {547},
  number = {7663},
  pages = {336--339},
  year = {2017},
  doi = {10.1038/nature23018}
}

@article{Hicks2019BestPractices,
  author = {Hicks, Jennifer L. and Althoff, Tim and Sosic, Rok and Kuhar, Peter and Bostjancic, Bojan and King, Abby C. and Leskovec, Jure and Delp, Scott L.},
  title = {Best practices for analyzing large-scale health data from wearables and smartphone apps},
  journal = {npj Digital Medicine},
  volume = {2},
  number = {45},
  year = {2019},
  doi = {10.1038/s41746-019-0121-1}
}

@article{Xu2022GLOBEM,
  title = {{GLOBEM}: {Cross}-{Dataset} {Generalization} of {Longitudinal} {Human} {Behavior} {Modeling}},
  author = {Xu, Xuhai and Liu, Xin and Zhang, Han and Wang, Weichen and Nepal, Subigya and Sefidgar, Yasaman and Seo, Woosuk and Kuehn, Kevin S. and Huckins, Jeremy F. and Morris, Margaret E. and Nurius, Paula S. and Riskin, Eve A. and Patel, Shwetak and Althoff, Tim and Campbell, Andrew and Dey, Anind K. and Mankoff, Jennifer},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {6},
  number = {4},
  pages = {190},
  year = {2022},
  doi = {10.1145/3569485}
}

@inproceedings{Althoff2017Sleep,
  author = {Althoff, Tim and Horvitz, Eric and White, Ryen W. and Zeitzer, Jamie},
  title = {Harnessing the Web for Population-Scale Physiological Sensing: A Case Study of Sleep and Performance},
  booktitle = {Proceedings of the 26th International Conference on World Wide Web},
  pages = {113--122},
  year = {2017},
  doi = {10.1145/3038912.3052637}
}

@article{Althoff2016Counseling,
  author = {Althoff, Tim and Clark, Kevin and Leskovec, Jure},
  title = {Large-scale Analysis of Counseling Conversations: An Application of Natural Language Processing to Mental Health},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {4},
  pages = {463--476},
  year = {2016},
  doi = {10.1162/tacl_a_00111}
}

@article{Sharma2023Empathy,
  author = {Sharma, Ashish and Lin, Inna W. and Miner, Adam S. and Atkins, David C. and Althoff, Tim},
  title = {Human--{AI} collaboration enables more empathic conversations in text-based peer-to-peer mental health support},
  journal = {Nature Machine Intelligence},
  volume = {5},
  pages = {46--57},
  year = {2023},
  doi = {10.1038/s42256-022-00593-2}
}
```

---
---

## Expert 7: Varun Mishra (Northeastern)
**Domain**: Receptivity detection, GLOSS multi-agent sensing, UbiWell Lab

---

### Section 1: Introduction

**Key arguments from Mishra's perspective:**
- Receptivity is the missing link in JITAI deployment — knowing WHEN to intervene is as important as knowing WHAT to intervene on
- The diary paradox is fundamentally a receptivity problem: users who most need support are least receptive to active data collection
- PULSE's INT_avail prediction from sensing alone (0.706) directly addresses the receptivity detection challenge with a new paradigm

**Evidence to cite:**
- INT_avail: Auto-Sense 0.706 >> CALLM 0.542 — sensing-only outperforms diary for detecting intervention availability
- INT_avail improvement minimal with diary addition — confirming it is a behavioral (not emotional) construct
- Auto-Multi 0.716 for INT_avail — best overall

**Related work to reference:**
- Mishra et al. (IMWUT 2021) — detecting receptivity in natural environments
- Kunzler, Mishra et al. (IMWUT 2019) — exploring state-of-receptivity for mHealth
- Choube, Mishra et al. (IMWUT 2025) — GLOSS multi-agent sensemaking

**Phrasing suggestions:**
- "...PULSE reframes receptivity detection from a classification problem over contextual features to an agentic investigation of behavioral patterns..."
- "...the finding that INT_availability is best predicted by sensing alone validates the behavioral nature of receptivity..."
- "...while prior receptivity models rely on fixed contextual features, PULSE's agentic approach autonomously identifies which behavioral signals are informative for each individual..."

**Pitfalls to avoid:**
- Do NOT equate INT_avail directly with Mishra's receptivity construct — INT_avail is self-reported availability, not response to intervention
- Do NOT claim PULSE replaces receptivity detection — it predicts a proxy (self-reported availability)
- Do NOT ignore the distinction between static vs. adaptive receptivity models

**Reviewer preemption:**
- Preempt: "How does INT_avail relate to actual receptivity?" — Acknowledge it is a proxy; self-reported availability correlates with but is not identical to actual intervention receptivity. The sensing-only prediction result supports the behavioral nature of both constructs.
- Preempt: "Why not compare with Mishra's receptivity models?" — Different population (cancer survivors vs. general), different construct (self-reported availability vs. intervention response), different sensing modalities. Position as complementary evidence.

---

### Section 2: Related Work

**Key arguments:**
- Position PULSE in the receptivity literature: from contextual feature models (Kunzler/Mishra 2019) → ML-based detection (Mishra 2021) → LLM sensemaking (GLOSS 2025) → agentic investigation (PULSE)
- GLOSS (Choube/Mishra 2025) is the closest multi-agent system but focuses on open-ended sensemaking, not prediction; PULSE focuses on prediction with autonomous investigation
- The evolution from fixed features to multi-agent sensemaking to agentic tool use represents a paradigm shift in how sensing data is interpreted

**Evidence to cite:**
- GLOSS: 87.93% accuracy on sensemaking tasks — shows LLM agents can interpret sensing data
- Mishra 2021: 77% F1 increase over random for receptivity detection — ceiling of fixed-feature approaches
- PULSE: 0.706 BA for INT_avail from sensing alone — new approach to receptivity-adjacent prediction

**Related work to reference:**
- Kunzler, Mishra et al. (IMWUT 2019) — state-of-receptivity exploration
- Mishra et al. (IMWUT 2021) — detecting receptivity with ML models (static + adaptive)
- Choube et al. (IMWUT 2025) — GLOSS multi-agent LLM sensemaking
- Nahum-Shani et al. (2018) — JITAI framework defining receptivity as key component

**Phrasing suggestions:**
- "...GLOSS demonstrates that multi-agent LLM systems can perform complex sensemaking over passive sensing; PULSE extends this by giving agents autonomous investigation tools for prediction..."
- "...the progression from feature-based receptivity detection to agentic behavioral investigation represents a qualitative shift in how we reason about intervention timing..."

**Pitfalls to avoid:**
- Do NOT position PULSE as replacing GLOSS — they address different problems (prediction vs. sensemaking)
- Do NOT overclaim the receptivity connection without acknowledging construct differences

**Reviewer preemption:**
- Preempt: "How does this compare to GLOSS?" — GLOSS is open-ended sensemaking with multi-agent debate; PULSE is prediction-oriented with single-agent tool use. Complementary approaches: GLOSS for understanding, PULSE for decision support.

---

### Section 3: System Design

**Key arguments:**
- The 8 MCP sensing tools mirror how a receptivity researcher would investigate behavioral context — checking activity patterns, sleep, phone usage
- The agentic investigation loop (decide what to examine → query → reason → query again) parallels the adaptive models in Mishra 2021 that continuously learn individual patterns
- Session memory captures user-level behavioral regularities without ground truth leakage — similar to how adaptive receptivity models personalize over time

**Evidence to cite:**
- 8 tools: daily summary, behavioral timeline, targeted hourly query, raw events, baseline comparison, receptivity history, similar days, peer cases
- Agent averages 30-90 seconds per prediction, making multiple tool calls per investigation
- Session memory stores per-user reflections on behavioral patterns

**Phrasing suggestions:**
- "...the agent's investigation strategy parallels adaptive receptivity models that learn individual behavioral patterns over time..."
- "...the receptivity history tool explicitly queries past intervention opportunity signals, grounding predictions in longitudinal behavioral context..."

**Pitfalls to avoid:**
- Do NOT claim the tools were designed specifically for receptivity — they are general-purpose sensing query tools
- Do NOT overstate the session memory as replacing adaptive ML — it is a simpler mechanism

**Reviewer preemption:**
- Preempt: "Why not use a multi-agent architecture like GLOSS?" — Single-agent with tools is simpler, more interpretable, and sufficient for prediction. Multi-agent debate adds complexity without clear prediction benefits. Future work could combine approaches.

---

### Section 4: Evaluation Methodology

**Key arguments:**
- The within-subject design with ~3,900 entries per version is appropriate for detecting behavioral pattern differences
- Balanced accuracy is the right metric for imbalanced receptivity-related constructs
- The evaluation should be contextualized against receptivity detection benchmarks

**Evidence to cite:**
- INT_avail base rate comparison: p > 0.05 (representative sample)
- Per-user BA distributions available for all version pairs

**Phrasing suggestions:**
- "...within-subject evaluation with thousands of entries per version provides sufficient power to detect meaningful differences in behavioral state prediction..."

**Pitfalls to avoid:**
- Do NOT compare directly to receptivity detection accuracy numbers (different constructs, different metrics)

**Reviewer preemption:**
- Preempt: "Retrospective evaluation is unrealistic for receptivity" — Acknowledge; real-time detection is future work. The retrospective design isolates the algorithm's ability to interpret behavioral signals.

---

### Section 5: Results

**Key arguments:**
- INT_avail is the standout finding from a receptivity perspective: sensing alone (0.706) dramatically outperforms diary (CALLM 0.542)
- This validates the behavioral nature of availability — it is about what you are DOING, not what you are FEELING
- The diary paradox is most acute for receptivity: the user who needs intervention cannot self-report availability

**Evidence to cite:**
- INT_avail: Auto-Sense 0.706, Auto-Multi 0.716, CALLM 0.542
- INT_avail improvement from diary: 0.716 vs 0.706 — marginal (only 0.01 gain)
- ER_desire: 0.751 with multimodal — emotional constructs still benefit from diary

**Phrasing suggestions:**
- "...the INT_avail result demonstrates that intervention availability is fundamentally a behavioral construct — best captured by passive sensing of what the user is doing, not by asking them how they feel..."
- "...the negligible improvement from diary addition (0.716 vs 0.706) for INT_avail, contrasted with the substantial improvement for ER_desire (0.751 vs 0.653), reveals a clean dissociation between behavioral and emotional intervention constructs..."

**Pitfalls to avoid:**
- Do NOT claim this proves receptivity is purely behavioral — INT_avail is a proxy
- Do NOT ignore that ER_desire (emotional) still benefits substantially from diary

**Reviewer preemption:**
- Preempt: "INT_avail is not receptivity" — Agreed, it is a proxy. But the construct (self-reported availability for intervention) is closely related to what receptivity research targets. The sensing-only finding has direct implications for receptivity model design.

---

### Section 6: Discussion

**Key arguments:**
- Separate desire from availability in JITAI systems — this is a concrete design recommendation
- The agentic approach handles the temporal dynamics of receptivity naturally — the agent can look back hours, days, or weeks
- The diary paradox motivates sensing-first JITAI design where receptivity is estimated passively

**Evidence to cite:**
- INT_avail vs ER_desire dissociation: sensing excels at behavioral, diary at emotional
- Auto-Sense overall 0.589 — viable for deployment without diary

**Phrasing suggestions:**
- "...JITAI designers should model intervention desire and intervention availability as separate constructs, using different data sources for each..."
- "...the agentic investigation paradigm offers a natural framework for adaptive receptivity detection that can dynamically weight different behavioral signals..."

**Pitfalls to avoid:**
- Do NOT claim PULSE is ready for real-time receptivity detection (30-90s latency)
- Do NOT ignore the cost implications for JITAI deployment

**Reviewer preemption:**
- Preempt: "Latency prohibits real-time JITAI use" — Acknowledge; propose edge LLMs and model distillation as paths to deployment. The insight (sensing alone predicts availability) is valuable regardless of current latency.

---

### BibTeX References (Mishra-relevant)

```bibtex
@article{Mishra2021Receptivity,
  author = {Mishra, Varun and K\"{u}nzler, Florian and Kramer, Jan-Niklas and Fleisch, Elgar and Kowatsch, Tobias and Kotz, David},
  title = {Detecting Receptivity for {mHealth} Interventions in the Natural Environment},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {5},
  number = {2},
  pages = {74},
  year = {2021},
  doi = {10.1145/3463492}
}

@article{Kunzler2019Receptivity,
  author = {K\"{u}nzler, Florian and Mishra, Varun and Kramer, Jan-Niklas and Kotz, David and Fleisch, Elgar and Kowatsch, Tobias},
  title = {Exploring the State-of-Receptivity for {mHealth} Interventions},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {3},
  number = {4},
  pages = {140},
  year = {2019},
  doi = {10.1145/3369805}
}

@article{Choube2025GLOSS,
  author = {Choube, Akshat and Le, Ha and Li, Jiachen and Ji, Kaixin and Das Swain, Vedant and Mishra, Varun},
  title = {{GLOSS}: Group of {LLMs} for Open-ended Sensemaking of Passive Sensing Data for Health and Wellbeing},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {9},
  number = {3},
  pages = {76:1--76:32},
  year = {2025},
  doi = {10.1145/3749474}
}

@article{NahumShani2018JITAI,
  author = {Nahum-Shani, Inbal and Smith, Shawna N. and Spring, Bonnie J. and Collins, Linda M. and Witkiewitz, Katie and Tewari, Ambuj and Murphy, Susan A.},
  title = {Just-in-Time Adaptive Interventions ({JITAIs}) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support},
  journal = {Annals of Behavioral Medicine},
  volume = {52},
  number = {6},
  pages = {446--462},
  year = {2018},
  doi = {10.1007/s12160-016-9830-8}
}
```

---
---

## Expert 8: Laura Barnes (UVA)
**Domain**: Sensing for chronic disease, social anxiety, cancer populations

---

### Section 1: Introduction

**Key arguments from Barnes's perspective:**
- Cancer survivors represent a clinically vulnerable population where passive sensing can be transformative — they face sustained mental health burden post-treatment but fall through gaps in follow-up care
- Social anxiety research has shown that GPS mobility patterns correlate with clinical constructs (r = -0.67 for location entropy and social anxiety); PULSE extends this paradigm from correlation to prediction
- The diary paradox is especially acute in clinical populations where symptom burden itself reduces self-report capacity

**Evidence to cite:**
- PULSE achieves 0.661 mean BA across 16 binary targets for cancer survivors
- ER_desire 0.751 — detecting emotion regulation desire is directly clinically relevant
- NA_State 0.722 — negative affect detection in cancer survivors from sensing+diary
- Sensing-only viable: Auto-Sense 0.589 overall, INT_avail 0.706

**Related work to reference:**
- Boukhechba, Barnes et al. (JMIR Mental Health 2018) — predicting social anxiety from GPS
- Chow, Barnes et al. (JMIR 2017) — mobile sensing for depression, social anxiety, state affect
- Cai, Boukhechba, Barnes et al. (Smart Health 2020) — mobile sensing framework for breast cancer patients

**Phrasing suggestions:**
- "...cancer survivors face a mental health monitoring gap: acute treatment ends but psychological burden persists, and passive sensing can bridge this gap..."
- "...prior work in our group established that GPS-derived mobility features predict social anxiety with 85% accuracy; PULSE extends this paradigm by using LLM agents to autonomously investigate multi-modal behavioral patterns..."
- "...the BUCS dataset of 418 cancer survivors with 5-week longitudinal sensing represents one of the largest clinical sensing datasets in oncology supportive care..."

**Pitfalls to avoid:**
- Do NOT generalize from cancer survivors to all chronic disease populations without caveats
- Do NOT ignore platform differences (iOS vs Android) in sensing coverage
- Do NOT understate the clinical heterogeneity of cancer survivors (different cancer types, stages, treatments)

**Reviewer preemption:**
- Preempt: "Why cancer survivors specifically?" — They represent a clinically underserved population with high mental health burden (depression rates 20-30%), ongoing follow-up care contact points, and willingness to use smartphone monitoring. The BUCS study provides a unique large-scale clinical dataset.
- Preempt: "How does platform difference affect results?" — Acknowledge iOS limitations (no app usage, light data); the sensing tools are designed to work with available modalities per platform.

---

### Section 2: Related Work

**Key arguments:**
- Position PULSE in the clinical sensing literature: from student populations (StudentLife) → general depression (Saeb 2015) → anxiety (Boukhechba/Barnes 2018) → cancer (BUCS/PULSE)
- The gap: most sensing studies use student populations; clinical populations (especially cancer) are underrepresented
- Clinical sensing faces unique challenges: treatment effects on behavior, variable health status, heterogeneous symptoms

**Evidence to cite:**
- Boukhechba, Barnes 2018: 228 undergrads, GPS only, social anxiety prediction accuracy 85%
- Chow, Barnes et al. 2017: mobile sensing testing clinical models across depression, social anxiety, state affect
- BUCS: 418 cancer survivors, 8 modalities, 5 weeks — dramatically larger than most clinical sensing studies

**Related work to reference:**
- Boukhechba et al. (JMIR Mental Health 2018) — GPS and social anxiety
- Chow et al. (JMIR 2017) — mobile sensing and clinical models
- Cai et al. (Smart Health 2020) — breast cancer mobile sensing framework
- Wang et al. (StudentLife, UbiComp 2014) — foundational but student population
- Saeb et al. (JMIR 2015) — GPS depression correlates

**Phrasing suggestions:**
- "...while passive sensing for mental health has been extensively studied in college populations, clinical populations—particularly cancer survivors—remain underexplored..."
- "...the BUCS dataset extends mobile sensing research to a clinically relevant population with known mental health burden..."

**Pitfalls to avoid:**
- Do NOT dismiss student population studies — they built the foundational evidence
- Do NOT overstate the uniqueness of cancer survivors — the diary paradox and sensing approach are general

**Reviewer preemption:**
- Preempt: "Why not use clinical sensing benchmarks?" — None exist for cancer survivors with multi-modal sensing. BUCS is the first of its kind at this scale.

---

### Section 3: System Design

**Key arguments:**
- The BUCS dataset design (3x daily EMA, 8 sensing modalities, 5 weeks) balances clinical feasibility with data richness
- Platform-specific sensing coverage is a real-world clinical constraint — the system handles it gracefully
- The focus targets (ER_desire, INT_avail, PA_State, NA_State) were selected for clinical relevance to cancer survivor mental health

**Evidence to cite:**
- 418 cancer survivors, ~5 weeks, 3x daily EMA
- 8 sensing modalities: motion, GPS, screen, keyboard, app (Android), light (Android), music, sleep
- Platform distribution: slightly more Android in pilot (36% vs 25%)
- Focus targets map to JITAI tailoring variables: desire (ER_desire) and availability (INT_avail)

**Phrasing suggestions:**
- "...the BUCS study design prioritizes ecological validity over controlled data collection, reflecting real-world clinical conditions..."
- "...ER_desire and INT_avail map directly to JITAI tailoring variables (desire to engage and availability for intervention), making predictions immediately actionable..."

**Pitfalls to avoid:**
- Do NOT ignore missing data — address how the system handles incomplete sensing
- Do NOT conflate EMA compliance with clinical engagement
- Do NOT overlook that high-compliance users (selected for pilot) may not be representative of the hardest-to-reach survivors

**Reviewer preemption:**
- Preempt: "High-compliance users may not need passive sensing" — Acknowledged in representativeness analysis (82 vs 34 mean EMA). However, the diary paradox means even high-compliance users miss EMAs during acute distress. Sensing-only prediction provides a safety net.

---

### Section 4: Evaluation Methodology

**Key arguments:**
- Representativeness analysis is critical for clinical populations — report in main text
- The NA_State difference (p = 0.028) between pilot and full cohort should be interpreted carefully
- Cross-subject CV prevents within-subject overfitting but may miss individual-level patterns

**Evidence to cite:**
- Representativeness: 3/4 key targets p > 0.05; NA_State p = 0.028, r = 0.19 (small effect)
- EMA count significantly higher in pilot (82 vs 34) — by design

**Phrasing suggestions:**
- "...representativeness analysis reveals that the pilot subset is demographically comparable to the full cohort on most clinical measures, with a small effect on negative affect base rates..."

**Pitfalls to avoid:**
- Do NOT bury the representativeness analysis in supplementary — it belongs in main text
- Do NOT ignore that NA_State difference could bias results

**Reviewer preemption:**
- Preempt: "The NA_State base rate difference undermines results" — Small effect (r = 0.19), and PULSE still achieves 0.722 BA for NA_State — strong performance despite the slight distributional shift.

---

### Section 5: Results

**Key arguments:**
- The clinical actionability of predictions: ER_desire (0.751) means we can detect 3 out of 4 moments when a cancer survivor wants emotional support
- NA_State (0.722) and PA_State (0.733) — affect state detection at clinically useful levels
- Sensing-only performance (0.589) is already actionable for clinical deployment where diary compliance is uncertain

**Evidence to cite:**
- ER_desire: 0.751 (Auto-Multi+) — high clinical utility
- PA_State: 0.733, NA_State: 0.722 — well above chance (0.5) and ML baselines (~0.52)
- INT_avail: 0.706 from sensing alone — actionable without requiring any self-report
- Sensing-only overall: 0.589 — viable clinical deployment baseline

**Phrasing suggestions:**
- "...ER_desire prediction at 0.751 BA means the system could correctly identify approximately three-quarters of moments when a cancer survivor desires emotional support..."
- "...the clinical implication is a proactive monitoring system that identifies emotional need without adding to patient burden..."

**Pitfalls to avoid:**
- Do NOT translate BA directly to clinical sensitivity/specificity without per-class analysis
- Do NOT claim the system can replace clinical assessment
- Do NOT ignore that false positives (unnecessary interventions) and false negatives (missed needs) have different clinical costs

**Reviewer preemption:**
- Preempt: "What is the clinical impact of errors?" — False positives (unnecessary wellness check-in) have low cost; false negatives (missed distress) have high cost. The system is designed as a screening/triage layer, not a diagnostic tool.

---

### Section 6: Discussion

**Key arguments:**
- Graceful degradation is essential for clinical deployment: multimodal when diary is available, sensing-only as fallback
- The diary paradox is especially pronounced in cancer survivors during acute distress episodes, treatment side effects, or fatigue
- The system could integrate into existing survivorship care pathways as a passive monitoring layer

**Evidence to cite:**
- Sensing-only fallback: 0.589 overall, 0.706 for INT_avail
- Multimodal when available: 0.660 overall, 0.751 for ER_desire

**Phrasing suggestions:**
- "...graceful degradation from multimodal (0.660) to sensing-only (0.589) ensures the system remains clinically useful even during the diary gaps that characterize acute distress episodes..."
- "...passive sensing for cancer survivors fills a gap between scheduled follow-up visits, providing continuous behavioral monitoring without adding to patient burden..."

**Pitfalls to avoid:**
- Do NOT claim integration into clinical workflows without discussing regulatory, ethical, and implementation barriers
- Do NOT ignore that cancer survivors may have different relationships with technology monitoring (surveillance concerns)

**Reviewer preemption:**
- Preempt: "Privacy concerns with passive sensing in cancer survivors" — Discuss IRB approval, informed consent, data de-identification, and the critical distinction between passive sensing for the user's benefit vs. surveillance. Clinical populations may be more accepting when the monitoring serves their care.

---

### BibTeX References (Barnes-relevant)

```bibtex
@article{Boukhechba2018SocialAnxiety,
  author = {Boukhechba, Mehdi and Chow, Philip and Fua, Karl and Teachman, Bethany A. and Barnes, Laura E.},
  title = {Predicting Social Anxiety From Global Positioning System Traces of College Students: Feasibility Study},
  journal = {JMIR Mental Health},
  volume = {5},
  number = {3},
  pages = {e10101},
  year = {2018},
  doi = {10.2196/10101}
}

@article{Chow2017MobileSensing,
  author = {Chow, Philip I. and Fua, Karl and Huang, Yu and Bonelli, Wesley and Xiong, Haoyi and Barnes, Laura E. and Teachman, Bethany A.},
  title = {Using Mobile Sensing to Test Clinical Models of Depression, Social Anxiety, State Affect, and Social Isolation Among College Students},
  journal = {Journal of Medical Internet Research},
  volume = {19},
  number = {3},
  pages = {e62},
  year = {2017},
  doi = {10.2196/jmir.6820}
}

@article{Cai2020BreastCancer,
  author = {Cai, Lihua and Boukhechba, Mehdi and Gerber, Ben S. and Barnes, Laura E. and Showalter, Shayna L. and Cohn, Wendy F. and Chow, Philip I.},
  title = {An integrated framework for using mobile sensing to understand response to mobile interventions among breast cancer patients},
  journal = {Smart Health},
  volume = {15},
  pages = {100086},
  year = {2020},
  doi = {10.1016/j.smhl.2019.100086}
}

@inproceedings{Wang2014StudentLife,
  author = {Wang, Rui and Chen, Fanglin and Chen, Zhenyu and Li, Tianxing and Harari, Gabriella and Tignor, Stefanie and Zhou, Xia and Ben-Zeev, Dror and Campbell, Andrew T.},
  title = {{StudentLife}: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones},
  booktitle = {Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
  pages = {3--14},
  year = {2014},
  doi = {10.1145/2632048.2632054}
}

@article{Saeb2015Depression,
  author = {Saeb, Sohrab and Zhang, Mi and Karr, Christopher J. and Schueller, Stephen M. and Corden, Marya E. and Kording, Konrad P. and Mohr, David C.},
  title = {Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study},
  journal = {Journal of Medical Internet Research},
  volume = {17},
  number = {7},
  pages = {e175},
  year = {2015},
  doi = {10.2196/jmir.4273}
}
```

---
---

## Expert 9: Bonnie Spring (Northwestern)
**Domain**: Cancer survivorship mHealth, JITAI optimization, behavior change

---

### Section 1: Introduction

**Key arguments from Spring's perspective:**
- Cancer survivors face a health-promoting system gap: no entity claims responsibility for ongoing behavioral health after treatment ends
- JITAIs are theoretically ideal for cancer survivorship care but depend on tailoring variables that are hard to estimate — PULSE provides a method to estimate these variables from passive data
- The diary paradox undermines the self-report backbone of current JITAI designs; sensing-based prediction of tailoring variables is essential

**Evidence to cite:**
- PULSE predicts ER_desire (0.751) and INT_avail (0.706) — two key JITAI tailoring variables
- Sensing-only prediction is viable (0.589 overall) — critical for JITAI deployment where diary compliance varies
- 418 cancer survivors, ~5 weeks, 3x daily EMA — clinical-grade longitudinal data

**Related work to reference:**
- Nahum-Shani, Spring et al. (2018) — JITAI framework defining tailoring variables
- Spring et al. (2019) — toward a health-promoting system for cancer survivors
- Spring et al. (JMIR 2018) — Make Better Choices 2, multicomponent mHealth for behavior change

**Phrasing suggestions:**
- "...cancer survivors need proactive mental health support, but the health care system lacks clear responsibility for behavioral health post-treatment..."
- "...JITAI theory identifies vulnerability, receptivity, and tailoring variables as critical — but estimating these in real-time remains the central challenge..."
- "...PULSE directly addresses the JITAI implementation gap by estimating tailoring variables (emotion regulation desire, intervention availability) from passive sensing data..."

**Pitfalls to avoid:**
- Do NOT present PULSE as a complete JITAI — it is the prediction layer, not the intervention
- Do NOT ignore the behavioral health context: cancer survivors face multiple simultaneous risk factors (diet, activity, smoking, distress)
- Do NOT conflate prediction accuracy with intervention effectiveness

**Reviewer preemption:**
- Preempt: "This is just prediction, not intervention" — PULSE provides the tailoring variable estimation layer that JITAI theory identifies as critical but hard to implement. The prediction-to-intervention pipeline is future work.
- Preempt: "How does this integrate with existing survivorship care?" — PULSE could integrate with the health-promoting system framework (Spring et al. 2019) as a passive monitoring layer that triggers proactive outreach from health promotionists.

---

### Section 2: Related Work

**Key arguments:**
- Position PULSE within JITAI literature: the theory is well-developed (Nahum-Shani 2018) but implementation is bottlenecked by tailoring variable estimation
- Prior mHealth behavior change interventions (MBC2) show that technology + coaching is effective — but require active engagement
- The gap: no system estimates JITAI tailoring variables from passive sensing alone

**Evidence to cite:**
- MBC2: sustained improvements in diet and activity through mHealth + coaching
- Nahum-Shani 2018: defines vulnerability, receptivity, tailoring variables as core JITAI components
- PULSE fills the tailoring variable estimation gap

**Related work to reference:**
- Nahum-Shani et al. (2018) — JITAI framework
- Spring et al. (JMIR 2018) — MBC2 mHealth behavior change trial
- Spring et al. (Health Psychology 2019) — health-promoting system for cancer survivors
- Mishra et al. (IMWUT 2021) — receptivity detection (related tailoring variable)

**Phrasing suggestions:**
- "...while mHealth behavior change interventions demonstrate efficacy when engagement is maintained, the diary paradox means the most vulnerable moments are precisely when engagement drops..."
- "...JITAI theory has outpaced implementation: Nahum-Shani et al. (2018) define tailoring variables as critical, but no existing system estimates them from passive sensing in clinical populations..."

**Pitfalls to avoid:**
- Do NOT overstate the JITAI connection — PULSE does not implement the full JITAI loop
- Do NOT ignore that tailoring variable estimation is necessary but not sufficient for effective JITAIs

**Reviewer preemption:**
- Preempt: "How does this connect to actual behavior change?" — PULSE is the sensing-to-prediction layer; connecting to intervention delivery and evaluating behavior change outcomes is future work. The contribution is enabling JITAI tailoring variables to be estimated passively.

---

### Section 3: System Design

**Key arguments:**
- ER_desire and INT_avail map directly to JITAI tailoring variables: desire/vulnerability and receptivity/availability
- The 2x2 factorial design is methodologically aligned with the MOST (Multiphase Optimization Strategy) framework that Spring and Collins advocate for optimizing mHealth interventions
- Cross-user RAG for calibration parallels population-level norms used in behavioral health assessment

**Evidence to cite:**
- Focus targets: ER_desire (emotion regulation desire → vulnerability), INT_avail (intervention availability → receptivity)
- Factorial design: 2x2 + baseline = clean optimization of components
- Cross-user RAG: population-level calibration analogous to normative comparison in clinical assessment

**Phrasing suggestions:**
- "...the 2x2 factorial design echoes the MOST framework's emphasis on component-level optimization before full intervention deployment..."
- "...ER_desire captures the 'vulnerability' tailoring variable in JITAI theory — the moment when a cancer survivor most needs support..."
- "...INT_avail captures the 'receptivity' tailoring variable — when the survivor is available to engage with an intervention..."

**Pitfalls to avoid:**
- Do NOT overclaim alignment with MOST — PULSE optimizes prediction components, not intervention components
- Do NOT ignore that the JITAI framework includes intervention options, decision rules, and distal outcomes that PULSE does not address

**Reviewer preemption:**
- Preempt: "A factorial design is not MOST" — Correct; the factorial design optimizes the prediction layer. A full MOST optimization of the JITAI would include intervention components. Position the factorial as proof-of-concept for component optimization.

---

### Section 4: Evaluation Methodology

**Key arguments:**
- The choice of 4 focus targets (ER_desire, INT_avail, PA_State, NA_State) is grounded in JITAI theory
- The evaluation is retrospective — prospective evaluation in a JITAI framework is needed
- The ML baselines provide a reference for what traditional approaches achieve on these clinical constructs

**Evidence to cite:**
- ML baselines: RF 0.518, XGBoost 0.514, Logistic 0.515 — near chance for clinical constructs
- Focus targets selected for JITAI relevance, not just predictability

**Phrasing suggestions:**
- "...the focus targets were selected for their direct mapping to JITAI tailoring variables, ensuring clinical relevance of the evaluation..."

**Pitfalls to avoid:**
- Do NOT claim the evaluation demonstrates JITAI effectiveness — it demonstrates prediction accuracy
- Do NOT ignore that prospective evaluation may yield different results

**Reviewer preemption:**
- Preempt: "Retrospective evaluation is insufficient for JITAI validation" — Agreed; this is acknowledged as a limitation. The contribution is demonstrating that passive sensing can predict tailoring variables at useful accuracy. Prospective JITAI evaluation is future work.

---

### Section 5: Results

**Key arguments:**
- ER_desire (0.751) is the most clinically exciting result — detecting when cancer survivors want emotional support
- The desire-availability dissociation (ER_desire benefits from diary, INT_avail from sensing) has direct JITAI design implications
- Sensing-only performance provides a floor for JITAI deployment without active engagement

**Evidence to cite:**
- ER_desire: Auto-Multi+ 0.751 vs. CALLM 0.632 — dramatic improvement
- INT_avail: Auto-Sense 0.706 vs. CALLM 0.542 — sensing outperforms diary
- PA_State: 0.733, NA_State: 0.722 — affect detection at clinically useful levels
- Sensing-only overall: 0.589 — viable deployment baseline

**Phrasing suggestions:**
- "...detecting emotion regulation desire at 0.751 BA means the system could identify three out of four moments when proactive outreach from a health promotionist would be welcome..."
- "...the dissociation between desire (diary-enhanced) and availability (sensing-sufficient) suggests JITAI systems should use different data sources for different tailoring variables..."

**Pitfalls to avoid:**
- Do NOT equate prediction accuracy with intervention effectiveness
- Do NOT ignore false positive/negative implications for clinical deployment
- Do NOT overstate the sensing-only result as sufficient for all JITAI tailoring variables

**Reviewer preemption:**
- Preempt: "What about intervention effectiveness?" — PULSE provides the prediction layer; intervention effectiveness depends on the intervention itself. The contribution is enabling passive estimation of JITAI tailoring variables.

---

### Section 6: Discussion

**Key arguments:**
- The desire-availability dissociation has profound JITAI design implications: desire is emotional (diary helps), availability is behavioral (sensing sufficient)
- PULSE could integrate into Spring's health-promoting system framework as the passive monitoring layer
- The graceful degradation from multimodal to sensing-only addresses the real-world challenge of variable patient engagement
- Cost and latency must be addressed before clinical deployment — but the paradigm is demonstrated

**Evidence to cite:**
- Desire-availability dissociation: ER_desire multimodal >> sensing; INT_avail sensing ≈ multimodal
- Graceful degradation: 0.660 → 0.589 overall; 0.716 → 0.706 for INT_avail

**Phrasing suggestions:**
- "...the practical implication for JITAI design is a two-channel system: passive sensing continuously estimates behavioral availability, while brief check-ins (when the user is available) assess emotional desire for support..."
- "...this aligns with Spring et al.'s (2019) vision of a health-promoting system where technology supports proactive outreach without adding patient burden..."

**Pitfalls to avoid:**
- Do NOT claim PULSE replaces clinical assessment or counseling
- Do NOT ignore that cancer survivors face multiple simultaneous risk factors that PULSE does not address (diet, activity, smoking)
- Do NOT overstate deployment readiness

**Reviewer preemption:**
- Preempt: "How does this address multiple risk factors?" — PULSE currently targets emotional states and intervention receptivity. Extending to physical activity, diet, and other behavioral targets is future work. The paradigm (agentic sensing investigation) is generalizable.
- Preempt: "Cost of LLM inference for clinical deployment" — Acknowledged as limitation. Current cost is 30-90s per prediction via Claude. Edge LLMs and distillation could reduce cost by orders of magnitude.

---

### BibTeX References (Spring-relevant)

```bibtex
@article{NahumShani2018JITAI,
  author = {Nahum-Shani, Inbal and Smith, Shawna N. and Spring, Bonnie J. and Collins, Linda M. and Witkiewitz, Katie and Tewari, Ambuj and Murphy, Susan A.},
  title = {Just-in-Time Adaptive Interventions ({JITAIs}) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support},
  journal = {Annals of Behavioral Medicine},
  volume = {52},
  number = {6},
  pages = {446--462},
  year = {2018},
  doi = {10.1007/s12160-016-9830-8}
}

@article{Spring2018MBC2,
  author = {Spring, Bonnie and Pellegrini, Christine and McFadden, H. Gene and Pfammatter, Angela F. and Stump, Tanya K. and Siddique, Juned and King, Abby C. and Hedeker, Donald},
  title = {Multicomponent {mHealth} Intervention for Large, Sustained Change in Multiple Diet and Activity Risk Behaviors: The {Make Better Choices 2} Randomized Controlled Trial},
  journal = {Journal of Medical Internet Research},
  volume = {20},
  number = {6},
  pages = {e10528},
  year = {2018},
  doi = {10.2196/10528}
}

@article{Spring2019CancerSurvivors,
  author = {Spring, Bonnie and Stump, Tammy and Penedo, Frank and Pfammatter, Angela Fidler and Robinson, June K.},
  title = {Toward a health-promoting system for cancer survivors: Patient and provider multiple behavior change},
  journal = {Health Psychology},
  volume = {38},
  number = {9},
  pages = {840--850},
  year = {2019},
  doi = {10.1037/hea0000760}
}

@article{Mishra2021Receptivity,
  author = {Mishra, Varun and K\"{u}nzler, Florian and Kramer, Jan-Niklas and Fleisch, Elgar and Kowatsch, Tobias and Kotz, David},
  title = {Detecting Receptivity for {mHealth} Interventions in the Natural Environment},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {5},
  number = {2},
  pages = {74},
  year = {2021},
  doi = {10.1145/3463492}
}
```

---
---

## Expert 10: Yubin Kim (MIT Media Lab)
**Domain**: Health-LLM, LLMs for wearable health prediction

---

### Section 1: Introduction

**Key arguments from Kim's perspective:**
- Health-LLM showed that LLMs can process wearable sensor data for health prediction — but with fixed prompts and structured input, hitting a ceiling
- PULSE takes the next step: from structured input formatting to autonomous investigation, where the LLM decides what data to examine
- Context enhancement (user context, health knowledge, temporal information) was key in Health-LLM; PULSE's agentic investigation is a more powerful form of context gathering

**Evidence to cite:**
- Health-LLM: context enhancement yields up to 23.8% improvement
- PULSE: agentic approach yields Auto-Multi 0.660 vs. Struct-Multi 0.603 (9.5% relative improvement from architecture alone)
- The improvement from agentic reasoning parallels but exceeds the improvement from context enhancement in Health-LLM

**Related work to reference:**
- Kim et al. (CHIL 2024) — Health-LLM: LLMs for health prediction via wearable sensor data
- Merrill, Althoff et al. (Nature Communications 2026) — PHIA: LLM agent for wearable analysis
- Zhang et al. (UbiComp 2024 Workshop) — LLMs for affective state prediction from smartphone sensors

**Phrasing suggestions:**
- "...Health-LLM demonstrated that context enhancement improves LLM health prediction by up to 23.8%; PULSE's agentic investigation represents a more powerful form of contextual reasoning where the agent autonomously gathers relevant information..."
- "...the progression from Health-LLM (structured prompts) to PHIA (code generation for analysis) to PULSE (autonomous investigation via tools) represents an arc toward increasingly autonomous LLM reasoning over health data..."
- "...PULSE extends Health-LLM's context enhancement paradigm from static prompt construction to dynamic, agent-driven information gathering..."

**Pitfalls to avoid:**
- Do NOT position PULSE as directly competing with Health-LLM — they address different scales and tasks
- Do NOT ignore that Health-LLM used fine-tuning while PULSE uses zero-shot prompting + tools
- Do NOT understate the cost difference: Health-LLM uses efficient fine-tuned models, PULSE uses large frontier models

**Reviewer preemption:**
- Preempt: "Why not fine-tune like Health-LLM?" — Fine-tuning requires labeled training data at scale. PULSE demonstrates that zero-shot agentic reasoning with tools can achieve strong performance without task-specific training. The approaches are complementary: fine-tuned models for efficiency, agentic models for reasoning.
- Preempt: "PHIA already does agentic wearable analysis" — PHIA targets open-ended health questions with code generation; PULSE targets specific clinical prediction (affect, intervention receptivity) with domain-specific sensing tools. Different tasks, complementary paradigms.

---

### Section 2: Related Work

**Key arguments:**
- Position PULSE in the LLM-for-health trajectory: Health-LLM (structured prompts, CHIL 2024) → Mental-LLM (text-based mental health, IMWUT 2024) → Zhang et al. (LLMs + smartphone sensing, UbiComp 2024) → PHIA (agentic wearable analysis, Nature Comms 2026) → PULSE (agentic sensing investigation for clinical prediction)
- The key differentiator: prior work uses fixed prompts or post-hoc analysis; PULSE gives the LLM autonomous investigation tools
- LENS (Xu et al. 2025) bridges sensing and language models through narrative synthesis — PULSE bridges them through tool-mediated investigation

**Evidence to cite:**
- Health-LLM: 12 LLMs evaluated, HealthAlpaca fine-tuned model best on 8/10 tasks
- Mental-LLM: instruction fine-tuning boosts mental health prediction
- PHIA: 84% accuracy on objective questions, 83% favorable on open-ended
- GLOSS: 87.93% accuracy on sensemaking
- Zhang et al.: "promising predictions" using zero-shot/few-shot LLMs on smartphone sensing
- PULSE: 0.661 mean BA with agentic investigation — first to combine tool-use agency with clinical prediction

**Related work to reference:**
- Kim et al. (CHIL 2024) — Health-LLM
- Xu et al. (IMWUT 2024) — Mental-LLM
- Zhang et al. (UbiComp 2024 Workshop) — LLMs for affective states via smartphone sensing
- Merrill et al. (Nature Communications 2026) — PHIA
- Choube et al. (IMWUT 2025) — GLOSS
- Xu et al. (arXiv 2025) — LENS: narrative synthesis from sensing data

**Phrasing suggestions:**
- "...Health-LLM established that LLMs can process wearable data for health prediction; PULSE demonstrates that giving LLMs autonomous investigation tools over sensing data dramatically improves this capability..."
- "...while Mental-LLM and Health-LLM rely on structured input formatting, PULSE's agentic approach allows the model to decide what information is relevant per-prediction..."

**Pitfalls to avoid:**
- Do NOT frame prior LLM health work as inadequate — frame PULSE as the next step
- Do NOT ignore that fine-tuned models (HealthAlpaca) are more efficient than agentic frontier models
- Do NOT overclaim novelty — the individual components (LLMs, sensing, tools) exist; the combination for clinical prediction is new

**Reviewer preemption:**
- Preempt: "What about LENS for sensing+LLM alignment?" — LENS aligns sensing with language models through narrative synthesis (sensor→text training); PULSE uses tool-mediated investigation (agent→sensor query). Complementary approaches. LENS could improve PULSE's structured agents; PULSE could improve LENS's reasoning capability.

---

### Section 3: System Design

**Key arguments:**
- The 8 MCP sensing tools provide a richer interface than Health-LLM's structured prompts or PHIA's code generation
- Health-LLM's context enhancement (user, health knowledge, temporal) maps to PULSE's tool categories: baseline comparison (user context), receptivity history (health knowledge), behavioral timeline (temporal context)
- The agentic investigation loop naturally implements the "synergistic" context enhancement that Health-LLM found most effective

**Evidence to cite:**
- Health-LLM's 3 context types → PULSE's 8 tools (more granular, agent-selected)
- Agentic agents average multiple tool calls per prediction — dynamic context gathering
- Cross-user RAG for calibration — analogous to population-level norms in Health-LLM

**Related work to reference:**
- Kim et al. (CHIL 2024) — context enhancement strategy (user + health + temporal)
- Yao et al. (ICLR 2023) — ReAct framework for reasoning + acting
- Schick et al. (NeurIPS 2023) — Toolformer: LLMs learning to use tools

**Phrasing suggestions:**
- "...Health-LLM demonstrated that combining user context, health knowledge, and temporal information synergistically improves predictions; PULSE's agentic tools operationalize this principle by letting the model dynamically select which contextual information to gather..."
- "...the MCP tool architecture extends the ReAct framework to domain-specific health data investigation..."

**Pitfalls to avoid:**
- Do NOT oversimplify the comparison with Health-LLM — different datasets, different tasks, different LLMs
- Do NOT claim the tools are optimal — they are a first instantiation of the paradigm

**Reviewer preemption:**
- Preempt: "Why not use code generation like PHIA?" — Domain-specific tools provide guardrails: the agent cannot accidentally access future data or generate incorrect analyses. Tools enforce temporal boundaries and data access patterns. Code generation is more flexible but less safe for clinical applications.

---

### Section 4: Evaluation Methodology

**Key arguments:**
- The evaluation covers more health prediction tasks (16 binary targets) than Health-LLM's 10 tasks
- Balanced accuracy on binary targets enables direct comparison with the Health-LLM evaluation paradigm
- The 2x2 factorial design adds methodological rigor absent from prior LLM health prediction studies

**Evidence to cite:**
- 16 binary targets × 7 versions = comprehensive evaluation matrix
- Health-LLM: 10 tasks, 4 datasets, multiple LLMs — breadth
- PULSE: 16 tasks, 1 dataset, 1 LLM, 7 architecture variants — depth on architecture

**Phrasing suggestions:**
- "...while Health-LLM evaluated breadth across tasks and models, PULSE evaluates depth — isolating the architectural contribution of agentic reasoning through controlled factorial comparison..."

**Pitfalls to avoid:**
- Do NOT claim superiority over Health-LLM based on different evaluation setups
- Do NOT ignore that Health-LLM tested multiple LLMs while PULSE tests one (Claude Sonnet)

**Reviewer preemption:**
- Preempt: "Only one LLM tested" — Acknowledged. Cross-model comparison is future work. The factorial design isolates the agentic architecture effect independent of model capability. If agentic >> structured with one model, the principle should transfer.

---

### Section 5: Results

**Key arguments:**
- The agentic advantage (0.660 vs 0.603) is architecturally driven — the same LLM with tools vs without tools
- This parallels Health-LLM's finding that context enhancement yields up to 23.8% improvement — but PULSE's improvement comes from self-directed investigation, not pre-constructed prompts
- The multi-modal advantage (0.660 vs 0.589) demonstrates that more data helps — consistent with Health-LLM's context enhancement findings

**Evidence to cite:**
- Agentic vs. structured: 0.660 vs. 0.603 (multimodal), 0.589 vs. 0.516 (sensing-only)
- Health-LLM context enhancement: up to 23.8%
- PULSE agentic enhancement: up to 14.1% (0.589 vs. 0.516 in sensing-only condition)

**Related work to reference:**
- Kim et al. (CHIL 2024) — context enhancement results
- Merrill et al. (Nature Comms 2026) — PHIA performance (84% accuracy on objective questions)

**Phrasing suggestions:**
- "...the agentic advantage in PULSE parallels Health-LLM's context enhancement findings but achieves it through autonomous investigation rather than static prompt design..."
- "...the 14.1% improvement from agentic architecture in the sensing-only condition demonstrates that the reasoning strategy, not just the data, drives prediction quality..."

**Pitfalls to avoid:**
- Do NOT directly compare PULSE BA numbers with Health-LLM accuracy numbers (different metrics, different tasks)
- Do NOT claim PULSE outperforms PHIA (different evaluation paradigms)

**Reviewer preemption:**
- Preempt: "How does this compare to Health-LLM's accuracy?" — Different tasks, different datasets, different metrics. The comparison is structural: both show that how the LLM processes health data matters more than which data it processes. PULSE shows this for agentic investigation; Health-LLM shows it for context enhancement.

---

### Section 6: Discussion

**Key arguments:**
- PULSE demonstrates that the LLM health prediction paradigm scales from structured prompts to agentic investigation
- The tool-use paradigm is model-agnostic and could be applied with Health-LLM's fine-tuned models for efficiency
- Future work: combine Health-LLM's fine-tuning efficiency with PULSE's agentic investigation capability

**Evidence to cite:**
- PULSE uses zero-shot Claude Sonnet; Health-LLM uses fine-tuned HealthAlpaca
- Both achieve strong results; PULSE with better reasoning, Health-LLM with better efficiency

**Phrasing suggestions:**
- "...the synergy between structured health knowledge (Health-LLM's approach) and autonomous investigation (PULSE's approach) suggests a future system that combines fine-tuned efficiency with agentic reasoning..."
- "...PULSE demonstrates that the health prediction community should consider investigation strategy as a first-class design dimension alongside data modality and model architecture..."

**Pitfalls to avoid:**
- Do NOT claim PULSE makes Health-LLM's approach obsolete — they are complementary
- Do NOT ignore efficiency: Health-LLM's HealthAlpaca runs on consumer hardware; PULSE requires frontier API access
- Do NOT understate the potential of combining approaches

**Reviewer preemption:**
- Preempt: "Why not just fine-tune on your data?" — Fine-tuning requires labeled data at scale and loses the ability to investigate novel patterns. Agentic investigation works zero-shot. For deployment, a distilled agentic model could combine both advantages.
- Preempt: "Model dependency limits reproducibility" — The paradigm (agent + sensing tools) is reproducible with any capable LLM. Cross-model comparison is future work. The tool interface is an open protocol (MCP).

---

### BibTeX References (Kim-relevant)

```bibtex
@inproceedings{Kim2024HealthLLM,
  author = {Kim, Yubin and Xu, Xuhai and McDuff, Daniel and Breazeal, Cynthia and Park, Hae Won},
  title = {Health-{LLM}: Large Language Models for Health Prediction via Wearable Sensor Data},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  pages = {522--539},
  year = {2024},
  volume = {248},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}

@article{Merrill2026PHIA,
  author = {Merrill, Mike A. and Paruchuri, Akshay and Rezaei, Naghmeh and Kovacs, Geza and Perez, Javier and Liu, Yun and Schenck, Erik and Hammerquist, Nova and Sunshine, Jake and Tailor, Shyam and Ayush, Kumar and Su, Hao-Wei and He, Qian and McLean, Cory Y. and Malhotra, Mark and Patel, Shwetak and Zhan, Jiening and Althoff, Tim and McDuff, Daniel and Liu, Xin},
  title = {Transforming wearable data into personal health insights using large language model agents},
  journal = {Nature Communications},
  volume = {17},
  pages = {1143},
  year = {2026},
  doi = {10.1038/s41467-025-67922-y}
}

@article{Xu2024MentalLLM,
  author = {Xu, Xuhai and Yao, Bingsheng and Dong, Yuanzhe and Gabriel, Saadia and Yu, Hong and Hendler, James and Ghassemi, Marzyeh and Dey, Anind K. and Wang, Dakuo},
  title = {Mental-{LLM}: Leveraging Large Language Models for Mental Health Prediction via Online Text Data},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {8},
  number = {1},
  pages = {31},
  year = {2024},
  doi = {10.1145/3643540}
}

@inproceedings{Zhang2024AffectLLM,
  author = {Zhang, Tianyi and Teng, Songyan and Jia, Hong and D'Alfonso, Simon},
  title = {Leveraging {LLMs} to Predict Affective States via Smartphone Sensor Features},
  booktitle = {Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp Adjunct)},
  pages = {709--716},
  year = {2024},
  doi = {10.1145/3675094.3678420}
}

@inproceedings{Yao2023ReAct,
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}

@inproceedings{Schick2023Toolformer,
  author = {Schick, Timo and Dwivedi-Yu, Jane and Dess\`{i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  title = {Toolformer: Language Models Can Teach Themselves to Use Tools},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2023}
}

@article{Choube2025GLOSS,
  author = {Choube, Akshat and Le, Ha and Li, Jiachen and Ji, Kaixin and Das Swain, Vedant and Mishra, Varun},
  title = {{GLOSS}: Group of {LLMs} for Open-ended Sensemaking of Passive Sensing Data for Health and Wellbeing},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {9},
  number = {3},
  pages = {76:1--76:32},
  year = {2025},
  doi = {10.1145/3749474}
}

@article{Xu2025LENS,
  author = {Xu, Wenxuan and Pillai, Arvind and Nepal, Subigya and Collins, Amanda C. and Mackin, Daniel M. and Heinz, Michael V. and Griffin, Tess Z. and Jacobson, Nicholas C. and Campbell, Andrew},
  title = {{LENS}: {LLM}-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models},
  journal = {arXiv preprint arXiv:2512.23025},
  year = {2025}
}

@article{Narayanan2013BSP,
  author = {Narayanan, Shrikanth S. and Georgiou, Panayiotis},
  title = {Behavioral Signal Processing: Deriving Human Behavioral Informatics from Speech and Language},
  journal = {Proceedings of the IEEE},
  volume = {101},
  number = {5},
  pages = {1203--1233},
  year = {2013},
  doi = {10.1109/JPROC.2012.2236291}
}
```

---
---

## Cross-Expert Positioning Table Reference

| Dimension | Health-LLM (Kim) | PHIA (Merrill) | GLOSS (Mishra) | Mental-LLM (Xu) | Zhang et al. | PULSE |
|-----------|------------------|----------------|-----------------|------------------|--------------|-------|
| Input | Wearable features | Wearable data | Passive sensing | Online text | Smartphone sensing | Multi-modal sensing + diary |
| Architecture | Structured prompts / fine-tuning | Code generation agent | Multi-agent debate | Instruction fine-tuning | Zero/few-shot prompts | Single agent + MCP tools |
| Task | 10 health prediction | Open-ended QA | Open-ended sensemaking | Mental health classification | Affect prediction | 16 clinical targets |
| Autonomy | None (fixed prompts) | High (code gen) | High (multi-agent) | None (fixed prompts) | None (fixed prompts) | High (tool investigation) |
| Population | General (4 datasets) | General (wearable users) | General | General (Reddit) | Students | Cancer survivors |
| Evaluation | Accuracy, multiple LLMs | Human evaluation | Accuracy + consistency | BA, multiple LLMs | Correlation | BA, 2x2 factorial |
| Key innovation | Context enhancement | Code gen + web search | Multi-agent triangulation | Instruction fine-tuning | Lifecycle descriptions | Agentic sensing tools |

---

## Shared BibTeX (Referenced Across Multiple Experts)

```bibtex
@inproceedings{Wang2014StudentLife,
  author = {Wang, Rui and Chen, Fanglin and Chen, Zhenyu and Li, Tianxing and Harari, Gabriella and Tignor, Stefanie and Zhou, Xia and Ben-Zeev, Dror and Campbell, Andrew T.},
  title = {{StudentLife}: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students Using Smartphones},
  booktitle = {Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing},
  pages = {3--14},
  year = {2014},
  doi = {10.1145/2632048.2632054}
}

@article{Saeb2015Depression,
  author = {Saeb, Sohrab and Zhang, Mi and Karr, Christopher J. and Schueller, Stephen M. and Corden, Marya E. and Kording, Konrad P. and Mohr, David C.},
  title = {Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study},
  journal = {Journal of Medical Internet Research},
  volume = {17},
  number = {7},
  pages = {e175},
  year = {2015},
  doi = {10.2196/jmir.4273}
}

@article{NahumShani2018JITAI,
  author = {Nahum-Shani, Inbal and Smith, Shawna N. and Spring, Bonnie J. and Collins, Linda M. and Witkiewitz, Katie and Tewari, Ambuj and Murphy, Susan A.},
  title = {Just-in-Time Adaptive Interventions ({JITAIs}) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support},
  journal = {Annals of Behavioral Medicine},
  volume = {52},
  number = {6},
  pages = {446--462},
  year = {2018},
  doi = {10.1007/s12160-016-9830-8}
}

@article{Xu2022GLOBEM,
  title = {{GLOBEM}: {Cross}-{Dataset} {Generalization} of {Longitudinal} {Human} {Behavior} {Modeling}},
  author = {Xu, Xuhai and Liu, Xin and Zhang, Han and Wang, Weichen and Nepal, Subigya and Sefidgar, Yasaman and Seo, Woosuk and Kuehn, Kevin S. and Huckins, Jeremy F. and Morris, Margaret E. and Nurius, Paula S. and Riskin, Eve A. and Patel, Shwetak and Althoff, Tim and Campbell, Andrew and Dey, Anind K. and Mankoff, Jennifer},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume = {6},
  number = {4},
  pages = {190},
  year = {2022},
  doi = {10.1145/3569485}
}

@article{Narayanan2013BSP,
  author = {Narayanan, Shrikanth S. and Georgiou, Panayiotis},
  title = {Behavioral Signal Processing: Deriving Human Behavioral Informatics from Speech and Language},
  journal = {Proceedings of the IEEE},
  volume = {101},
  number = {5},
  pages = {1203--1233},
  year = {2013},
  doi = {10.1109/JPROC.2012.2236291}
}

@inproceedings{Yao2023ReAct,
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}

@inproceedings{Schick2023Toolformer,
  author = {Schick, Timo and Dwivedi-Yu, Jane and Dess\`{i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  title = {Toolformer: Language Models Can Teach Themselves to Use Tools},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2023}
}
```
