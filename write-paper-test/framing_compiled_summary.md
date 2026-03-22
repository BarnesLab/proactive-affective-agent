# PULSE Framing Proposals: Compiled Summary

**Compiled from 10 expert advisory proposals for IMWUT submission**
**Date**: 2026-03-21

**Experts**: (1) Tanzeem Choudhury, (2) Xuhai "Orson" Xu, (3) Inbal Nahum-Shani, (4) Andrew Campbell, (5) Shrikanth Narayanan, (6) Tim Althoff, (7) Varun Mishra, (8) Laura Barnes, (9) Bonnie Spring, (10) Yubin Kim

---

## 1. Title Consensus

### Word/Phrase Frequency Across All Proposed Titles (30 titles total)

| Word/Phrase | Appearances | Experts |
|---|---|---|
| "PULSE" (as system name) | 16 of 30 titles | All 10 experts use it in at least one title |
| "Agentic" / "Autonomous" | 28 of 30 titles | All 10 |
| "Passive Sensing" / "Smartphone Sensing" | 24 of 30 | All 10 |
| "Cancer Survivors" | 18 of 30 | 1, 2, 3, 4, 6, 7, 8, 9, 10 (9/10) |
| "Proactive" | 12 of 30 | 1, 2, 3, 6, 7, 8, 9, 10 (8/10) |
| "Investigation" / "Investigate" | 16 of 30 | 1, 2, 3, 4, 5, 7, 8, 9, 10 (9/10) |
| "Emotional States" / "Affective" / "Affect" | 22 of 30 | All 10 |
| "LLM Agents" / "LLM" | 26 of 30 | All 10 |
| "Intervention Receptivity" / "Intervention Opportunities" | 8 of 30 | 3, 4, 5, 9, 10 (5/10) |
| "Behavioral Data" | 10 of 30 | 1, 2, 4, 6, 7, 8, 9 (7/10) |

### Each Expert's Top-Ranked Title

1. **Choudhury**: "PULSE: Closing the Sensing-to-Prediction Gap with Agentic LLMs for Proactive Affective Computing in Cancer Survivors"
2. **Xu**: "PULSE: Agentic LLM Investigation of Passive Sensing for Proactive Emotional Support in Cancer Survivors"
3. **Nahum-Shani**: "PULSE: Proactive Prediction of Intervention Receptivity via Agentic LLM Investigation of Passive Smartphone Sensing"
4. **Campbell**: "PULSE: Agentic LLMs That Investigate Smartphone Sensing Data to Predict Emotional States and Intervention Opportunities in Cancer Survivors"
5. **Narayanan**: "From Passive Sensing to Proactive Inference: Agentic LLM Reasoning over Multimodal Behavioral Signals for Predicting Emotional States and Intervention Receptivity in Cancer Survivors"
6. **Althoff**: "PULSE: Agentic LLMs for Proactive Emotional State Prediction from Passive Smartphone Sensing in Cancer Survivors"
7. **Mishra**: "PULSE: Agentic LLM Investigation of Passive Sensing Data for Proactive Affective State Prediction in Cancer Survivors"
8. **Barnes**: "PULSE: Proactive Affective Agents for Cancer Survivors via Autonomous Sensing Investigation"
9. **Spring**: "PULSE: Proactive Affective Agents for Predicting Intervention Receptivity in Cancer Survivors via Autonomous Sensing Investigation"
10. **Kim**: "PULSE: LLM Agents as Behavioral Investigators for Proactive Affect Prediction from Passive Smartphone Sensing"

### Recommended Composite Title

Based on convergence across all 10 experts, the strongest composite title would include: **PULSE** + **agentic/autonomous** + **LLM** + **investigation** + **passive sensing** + **emotional/affective states** + **cancer survivors**. Two strong candidates:

**Option A (descriptive, IMWUT-standard length):**
> PULSE: Agentic LLM Investigation of Passive Smartphone Sensing for Proactive Affective State and Intervention Receptivity Prediction in Cancer Survivors

**Option B (tighter, paradigm-forward):**
> PULSE: Agentic LLMs That Investigate Behavioral Sensing Data to Predict Emotional States and Intervention Opportunities in Cancer Survivors

---

## 2. Framing Consensus (7+ Experts Agree)

### Areas of Near-Universal Agreement (9-10/10 experts):

1. **The 2x2 factorial design is THE methodological backbone.** All 10 experts identify the {Structured, Autonomous} x {Sensing-only, Multimodal} factorial as the paper's strongest methodological asset and recommend building the evaluation around it.

2. **"Agentic investigation" is the core contribution, not "LLMs for health."** All 10 agree the paper must be framed as introducing a new *paradigm* (autonomous investigation of sensing data) rather than as "another LLM-for-health paper." The differentiator is *how* the LLM interacts with data, not that an LLM is used.

3. **The "diary paradox" should be a central narrative hook.** All 10 experts identify the diary paradox (most informative data absent when users most need help) as the compelling motivating story that grounds the sensing-first approach.

4. **Agent investigation traces must be shown qualitatively.** All 10 insist on including concrete examples of the agent's investigation process (which tools called, reasoning chains, how it differs from structured pipelines). This is what makes the system contribution tangible.

5. **De-emphasize absolute BA numbers; always present comparatively.** All 10 experts warn that BA=0.66 sounds modest in isolation and must always be contextualized against baselines (ML ~0.52, Structured ~0.60, CALLM ~0.61).

6. **De-emphasize filtering variants (Auto-Sense+, Auto-Multi+).** All 10 agree the marginal improvement (0.661 vs 0.660) should be mentioned briefly as evidence of agent robustness to noise, not as a main finding.

7. **Acknowledge N=50 limitation transparently but frame it correctly.** All 10 recommend being upfront about the 50-user pilot, presenting the representativeness analysis prominently, and noting that 50 users x ~80 entries = ~3,900 predictions per version provides substantial statistical power.

8. **Acknowledge ML baseline asymmetry (399 vs 50 users).** All 10 flag this as a real weakness and recommend framing ML baselines as reference points, not head-to-head comparisons. Several suggest re-running ML on the same 50 users before submission.

9. **INT_availability as a behavioral signal is a highlight finding.** All 10 identify Auto-Sense BA=0.706 on INT_avail (vs CALLM 0.542) as a key result worth emphasizing: sensing captures availability better than diary text.

10. **Do not claim real-time deployment readiness.** All 10 agree this must be clearly framed as retrospective replay, not a deployed system.

### Areas of Strong Agreement (7-8/10 experts):

11. **Lead the results with the agentic vs. structured comparison** (the 2x2 factorial result), not the ML baseline comparison. (All 10 agree on this order.)

12. **Frame the work through the JITAI lens.** 8/10 experts (all except possibly 5-Narayanan who frames via BSP, and 10-Kim who is more LLM-focused) recommend connecting explicitly to the JITAI framework and receptivity constructs.

13. **Per-user analysis and variance distributions are essential.** 9/10 explicitly state this (all except 3-Nahum-Shani, who implies it).

14. **The architecture is model-agnostic in principle.** 8/10 emphasize that the contribution is the agentic investigation paradigm, not the specific LLM (Claude). Recommend discussing this for reproducibility.

---

## 3. Positioning Consensus: Most-Cited Comparison Papers

### Papers cited by 8+ experts:

| Paper | Cited by | Count |
|---|---|---|
| **CALLM** (Wang et al., 2025, arXiv/CHI 2026) — direct predecessor | All 10 | 10/10 |
| **Nahum-Shani et al. (2018), JITAIs** — foundational framework | All 10 | 10/10 |
| **Wang et al. (2014), StudentLife** — foundational sensing | All 10 | 10/10 |
| **Mishra et al. (2021), Detecting Receptivity** — IMWUT, Distinguished Paper | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | 10/10 |
| **Kunzler/Mishra et al. (2019), State-of-Receptivity** | 2, 3, 4, 5, 6, 7, 8, 9, 10 | 9/10 |
| **Saeb et al. (2015), Mobile phone sensor correlates** | 1, 3, 4, 5, 6, 7, 8, 9 | 8/10 |
| **ReAct (Yao et al., 2023)** — LLM agent architecture | 2, 3, 4, 5, 6, 7, 8, 10 | 8/10 |
| **Zhang/Bae et al. (2024), LLMs for Affective States** — UbiComp Companion | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 | 10/10 |

### Papers cited by 5-7 experts:

| Paper | Cited by | Count |
|---|---|---|
| **GLOBEM (Xu et al., 2022)** — cross-dataset generalization | 2, 4, 5, 6, 7, 8, 10 | 7/10 |
| **Health-LLM (Kim et al., 2024)** — CHIL/PMLR | 2, 4, 5, 6, 7, 9, 10 | 7/10 |
| **Nepal et al. (2024), MindScape** — LLM+sensing journaling | 1, 2, 3, 4 | 4/10 |
| **Meegahapola et al. (2022/2023)** — mood inference generalization | 2, 4, 8, 10 | 4/10 |
| **Mental-LLM (Xu et al., 2024)** — IMWUT | 3, 6, 7, 8, 10 | 5/10 |
| **Feng et al. (2026)** — ML/DL/LLM comparison | 1, 6, 7, 8 | 4/10 |
| **GLOSS (Choube et al., 2025)** — IMWUT, multi-LLM sensemaking | 3, 7, 8, 10 | 4/10 |
| **Adler et al. (2024), Beyond Detection** — IMWUT | 1, 7 | 2/10 |
| **LENS (Xu et al., 2025)** — sensor-to-LLM alignment | 4, 6, 8 | 3/10 |

### Consensus positioning strategy:
All 10 experts frame PULSE at the intersection of three threads: (1) passive mobile sensing for mental health, (2) LLMs for health prediction, and (3) JITAI/receptivity. The unique contribution sits at this intersection: an LLM agent that autonomously investigates behavioral sensing data to predict clinical affect and intervention receptivity.

---

## 4. Claim Consensus

### Claims supported by all 10 experts:

| Claim | Support |
|---|---|
| **Agentic >> Structured on same data** (Auto-Multi 0.660 vs Struct-Multi 0.603) — cleanest, strongest claim | 10/10 |
| **Multimodal >> Sensing-only** (0.660 vs 0.589) | 10/10 |
| **LLM agents >> ML baselines** (0.661 vs 0.518) — with caveat about different evaluation setups | 10/10 |
| **Sensing-only is viable** (Auto-Sense 0.589 >> ML ~0.52) | 10/10 |
| **INT_availability best predicted by sensing** (0.706 vs CALLM 0.542) | 10/10 |
| **The diary paradox is real** — sensing provides fallback when diary absent | 10/10 |

### "Do Not Overclaim" consensus (all 10 agree):

| Do NOT Claim | Support |
|---|---|
| Do not claim real-time deployment readiness | 10/10 |
| Do not claim generalizability beyond cancer survivors | 10/10 |
| Do not overclaim ML baseline comparison (different N, different setup) | 10/10 |
| Do not claim filtering adds meaningful value | 10/10 |
| Do not claim cost-efficiency or scalability | 9/10 |

---

## 5. Disagreements

### Disagreement 1: Title — Include "Intervention Receptivity" or "Emotional States"?

- **Intervention receptivity in title**: Nahum-Shani (#3), Spring (#9) — argue the JITAI/receptivity framing is the core scientific contribution
- **Emotional states / affective in title**: Choudhury (#1), Xu (#2), Narayanan (#5), Althoff (#6), Mishra (#7), Kim (#10) — argue emotional state prediction is more general and accessible
- **Both in title**: Campbell (#4), Barnes (#8)
- **Resolution**: Most experts (6/10) favor "emotional/affective states" as the primary framing, with receptivity as a secondary story. Including both (as Campbell does) may be optimal.

### Disagreement 2: Narrative Entry Point — Clinical Need vs. Technical Paradigm

- **Lead with clinical need (JITAI/cancer survivors)**: Nahum-Shani (#3), Spring (#9), Barnes (#8), Campbell (#4) — "Lead with the JITAI tailoring variable problem, not with we used LLMs"
- **Lead with paradigm shift (passive-to-proactive, feature-engineering-to-investigation)**: Choudhury (#1), Xu (#2), Narayanan (#5), Kim (#10) — "The paradigm of agentic sensing investigation is the lasting contribution"
- **Balanced**: Althoff (#6), Mishra (#7) — "Lead with the diary paradox (clinical), then the paradigm shift"
- **Resolution**: Mild disagreement. The diary paradox provides a natural bridge — it is a clinical problem that motivates the technical paradigm. Most experts converge on "problem first (diary paradox + clinical need), then insight (agentic investigation)."

### Disagreement 3: How Prominently to Feature the ML Baseline Comparison

- **Minimize it, present as reference only**: Xu (#2), Nahum-Shani (#3), Althoff (#6), Mishra (#7), Barnes (#8), Kim (#10) — "The ML baselines are on different user sets; the primary comparison is the 2x2 factorial"
- **Include but be transparent**: Choudhury (#1), Campbell (#4), Narayanan (#5), Spring (#9) — "Important to include but always acknowledge the asymmetry; still informative because ML had MORE data"
- **Resolution**: Universal agreement to include ML baselines but not as the headline comparison. Slight disagreement on how much space to give them.

### Disagreement 4: Whether to Coin "Agentic Sensing" as a Term

- **Yes, coin it**: Choudhury (#1, Title 3), Xu (#2), Campbell (#4), Narayanan (#5 as "Agentic Behavioral Signal Processing"), Kim (#10) — "memorable, defines a new category"
- **Cautious / avoid coining**: Nahum-Shani (#3), Althoff (#6), Mishra (#7), Spring (#9) — "focus on concrete contribution; coining terms risks reviewer pushback if not fully established"
- **Resolution**: The term "agentic sensing investigation" appears naturally in multiple proposals. Use it descriptively rather than as a coined category term.

### Disagreement 5: How to Frame Narayanan's "Behavioral Signal Processing" Connection

- **Explicitly connect to BSP theory**: Narayanan (#5) — "Ground the approach in BSP framework; the LLM performs contextual interpretation traditionally done through hand-crafted feature engineering"
- **Not mentioned by others**: 9/10 experts do not reference BSP
- **Resolution**: This is a unique framing from Narayanan. Worth a paragraph in the discussion if it adds theoretical depth without confusing the IMWUT audience.

### Disagreement 6: Whether to Claim CALLM is "Surpassed"

- **Position CALLM as surpassed**: Most experts frame Auto-Multi >> CALLM
- **Be careful with CALLM comparison**: Xu (#2) — "Do not claim superiority over CALLM on diary-available entries"; Althoff (#6) — "the practical margin (~3.5 BA points) is modest"; Kim (#10) — "Position CALLM as a respected baseline that PULSE extends, not something to beat"
- **Resolution**: Frame PULSE as extending CALLM's reactive approach to proactive, not simply "beating" it. The contribution is capability (works without diary) not just accuracy.

### Disagreement 7: Emphasis on Per-User Memory vs. Tool Use

- **Memory is an important component to highlight**: Choudhury (#1), Xu (#2), Campbell (#4), Barnes (#8), Kim (#10)
- **Memory contribution is not ablated; be careful**: Xu (#2, explicit), Mishra (#7) — "Unless you can show performance improves over time within a user's study period, do not make strong claims about the memory system"
- **Resolution**: Present memory as an architectural component; do not claim it improves performance without ablation evidence.

### Disagreement 8: Priority of Contribution Ordering

- **Primary = agentic paradigm, Secondary = factorial evaluation, Tertiary = clinical application**: Mishra (#7), Kim (#10)
- **Primary = clinical application + JITAI, Secondary = agentic paradigm, Tertiary = evaluation rigor**: Nahum-Shani (#3), Spring (#9)
- **Primary = paradigm shift with clinical grounding equally weighted**: Choudhury (#1), Xu (#2), Campbell (#4), Narayanan (#5), Althoff (#6), Barnes (#8)
- **Resolution**: The majority (6/10) see the paradigm and clinical grounding as equally weighted. The paper should interleave both throughout rather than subordinating one to the other.

---

## 6. Unique Insights (Raised by Only 1-2 Experts)

### From Narayanan (#5): Behavioral Signal Processing Theoretical Grounding
- Proposes explicitly connecting PULSE to the Behavioral Signal Processing framework (Narayanan & Georgiou, Proceedings of the IEEE, 2013): the LLM performs "contextual behavioral signal interpretation" that has traditionally been done through hand-crafted feature engineering.
- Also warns against language implying the LLM "understands" emotional states — frame as "contextual behavioral signal interpretation."
- **Value**: Provides theoretical depth that could distinguish this from a pure systems paper. Worth a subsection in Discussion.

### From Xu (#2): GLOBEM Generalization Challenge as PULSE Advantage
- Argues that PULSE may sidestep GLOBEM's generalization problem because LLM agents carry general knowledge + per-user memory rather than training dataset-specific models.
- **Value**: A powerful framing for Discussion — agents may generalize better than trained models *because* they bring general reasoning.

### From Xu (#2): PHIA (Nature Communications 2026) as Comparison
- Identifies PHIA ("Transforming Wearable Data into Personal Health Insights using LLM Agents", Nature Communications 2026) as the closest conceptual comparison — LLM agent with code generation for wearable data analysis. But PHIA is retrospective Q&A, PULSE is prospective prediction.
- **Value**: Important positioning paper that only Xu identified. Should be in related work.

### From Xu (#2): Penetrative AI (Xu et al., HotMobile/ACL 2024) Connection
- Suggests PULSE instantiates the "Penetrative AI" vision of LLMs "penetrating" into physical-world sensor data.
- **Value**: Nice conceptual framing for the introduction.

### From Althoff (#6): Effect Sizes Over P-values
- Specifically emphasizes reporting Wilcoxon effect sizes (r > 0.9 for several comparisons), arguing these matter more than p-values for N=50.
- **Value**: Critical methodological point for IMWUT reviewers who are statistically literate.

### From Althoff (#6): INT_availability Comparison to CALLM Not Significant (p=0.107)
- Flags that Auto-Multi vs. CALLM on INT_availability specifically has p=0.107 — not significant at p<0.05. This weakens the "proactive beats reactive" narrative for this specific construct.
- **Value**: Important honesty check. Must report this transparently. The sensing-only (Auto-Sense) vs. CALLM comparison for INT_avail is the stronger claim.

### From Althoff (#6): Voyager (NeurIPS 2023) as Agentic Architecture Reference
- Suggests citing Voyager (LLM agents with skill libraries in Minecraft) as an analogy for PULSE's tool-based autonomous investigation.
- **Value**: Novel reference for the AI audience; strengthens the "agentic paradigm" framing.

### From Mishra (#7): GLOSS Comparison Strategy
- As a co-author of GLOSS (IMWUT 2025), provides the most nuanced differentiation: GLOSS is sensemaking (open-ended), PULSE is prediction (closed-form with ground truth). GLOSS uses code-generation, PULSE uses purpose-built query tools. Frame as complementary.
- **Value**: Authoritative positioning against the closest concurrent IMWUT work.

### From Spring (#9): "Silver Tsunami" Cancer Survivorship Context
- Cites her own work (Spring et al., 2019) arguing cancer survivors face a "silver tsunami" of cardiometabolic comorbidities and need proactive health promotion systems. PULSE provides the detection layer.
- **Value**: Strengthens the clinical motivation for reviewers less familiar with cancer survivorship.

### From Spring (#9): Shiffman et al. (2008) for Diary Paradox Evidence
- Recommends citing Shiffman et al.'s foundational EMA work and MNAR literature to support the diary paradox claim.
- **Value**: Grounds what could seem like an intuitive claim in established EMA methodology literature.

### From Kim (#10): "Is the Improvement from Agentic Reasoning, or from Making More LLM Calls?"
- Identifies and preempts a subtle reviewer objection: structured pipelines use 1 LLM call, agentic uses multiple. The counter: structured prompts already contain all available information; the advantage comes from selective focus and synthesis, not more compute.
- **Value**: Subtle but important objection to preempt. Worth addressing in the evaluation section.

### From Barnes (#8): Per-User Memory Data Leakage Concern
- Flags that reviewers may worry per-user memory introduces data leakage. Recommends detailing the information boundary: memory records only past prediction outcomes and self-reflections, never current ground truth.
- **Value**: Important methodological detail to preempt a potentially damaging reviewer concern.

### From Campbell (#4): Representativeness Analysis in Main Text, Not Supplementary
- Specifically recommends placing the 50 vs. 418 representativeness analysis in the main text, not supplementary, as a credibility signal.
- **Value**: Practical formatting advice that signals confidence and transparency.

### From Spring (#9): ContextAgent (NeurIPS 2025) as Reference
- Identifies ContextAgent ("Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions") as a relevant comparison — uses egocentric wearable sensors for proactive assistance in a complementary modality space.
- **Value**: Novel reference not cited by other experts. Strengthens the proactive-agent positioning.

---

## Summary: Recommended Framing Strategy

Based on convergence across all 10 experts:

1. **Title**: Use "PULSE" + "Agentic LLM" + "Investigation" + "Passive Sensing" + "Emotional States" + "Cancer Survivors." Include intervention receptivity if space allows.

2. **Core story**: The diary paradox (clinical hook) motivates the shift from reactive to proactive. PULSE introduces agentic sensing investigation — giving LLM agents tools to autonomously investigate behavioral data. The 2x2 factorial proves agentic reasoning is the key differentiator.

3. **Lead comparison**: The 2x2 factorial (agentic vs. structured) is the primary result. ML baselines are reference points. CALLM is a respected predecessor that PULSE extends, not replaces.

4. **Highlight findings**: (a) Agentic >> Structured across both modalities, (b) INT_availability best predicted by sensing (0.706), (c) Sensing-only is viable (0.589), (d) The diary paradox and how sensing resolves it.

5. **Be transparent about**: N=50 (with representativeness analysis), ML baseline asymmetry, retrospective design, cost/latency, stochasticity.

6. **Show the agent's work**: Qualitative investigation traces are essential — they distinguish PULSE from prompt engineering and make the contribution concrete.

7. **Contribute to the field, not just a system**: Frame as a paradigm (agentic investigation of behavioral data) that the community can adopt, not just a single system demonstration.

---

*This compiled summary is ready for distribution to all 10 experts for the A2A discussion round.*
