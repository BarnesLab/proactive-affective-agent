# Meta-Review: PULSE (IMWUT)
## Panel of 10 Experts

### Score Distribution

| # | Expert | Affiliation | Score | Confidence |
|---|--------|-------------|-------|------------|
| 1 | Tanzeem Choudhury | Cornell Tech | Minor Revision | High |
| 2 | Xuhai "Orson" Xu | UW SEA Lab | Major Revision | High |
| 3 | Inbal Nahum-Shani | U Michigan | Major Revision | High |
| 4 | Andrew Campbell | Dartmouth | Minor Revision | High |
| 5 | Shrikanth Narayanan | USC | Minor Revision | High |
| 6 | Tim Althoff | Stanford | Major Revision | High |
| 7 | Varun Mishra | Northeastern | Minor Revision | High |
| 8 | Laura Barnes | UVA | Minor Revision | High |
| 9 | Bonnie Spring | Northwestern | Major Revision | High |
| 10 | Yubin Kim | MIT Media Lab | Minor Revision | High |

**Mean assessment: 6 Minor Revision / 4 Major Revision**

All 10 reviewers rated confidence as High. The split is close but leans toward conditional acceptance. The four Major Revision votes (Xu, Nahum-Shani, Althoff, Spring) share deep concern about baseline fairness, clinical framing overreach, and the untested diary paradox claim. The six Minor Revision votes acknowledge these issues but consider them addressable without fundamental restructuring.

---

### Consensus Strengths (cited by 7+ reviewers)

1. **2x2 factorial design is rigorous and well-conceived** (10/10). Every reviewer praised the within-subject factorial crossing architecture with modality. Called "the strongest methodological contribution" (Choudhury), "elegant" (Althoff), "the right approach" (Spring), and "the kind of controlled ablation the community needs" (Kim). The within-subject comparisons and effect sizes (r > 0.90) are considered convincing.

2. **Dissociation between INT_availability (behavioral, sensing-sufficient) and ER_desire (psychological, diary-dependent) is a genuinely novel and important finding** (10/10). Universally cited as the paper's most valuable conceptual contribution. Nahum-Shani: "This finding alone merits publication." Mishra: "the most important contribution from a JITAI perspective." Narayanan: "has broad implications" for behavioral signal processing.

3. **MCP tool design is thoughtful, well-motivated, and reusable** (8/10). The eight tools are praised for mirroring clinical behavioral assessment (Choudhury), spanning meaningful investigation strategies (Xu), and being a concrete, reusable contribution (Kim, Campbell). Temporal boundary enforcement noted as critical for evaluation validity.

4. **Paper is well-written with honest, transparent limitations** (8/10). Multiple reviewers specifically praised the candid limitations section (Althoff: "refreshingly candid"; Campbell: "honest about its limitations"; Barnes: "transparent about calibration limitations"). Writing quality and problem formulation called "exceptional" (Xu).

5. **Clinically meaningful population (cancer survivors) and prediction targets** (7/10). Barnes, Choudhury, Campbell, Nahum-Shani, Mishra, Spring, and Narayanan all noted the value of evaluating on a clinical population rather than convenience student samples, with prediction targets directly relevant to intervention design.

---

### Consensus Weaknesses (cited by 7+ reviewers -- MUST FIX)

1. **ML baselines are too weak and the comparison is unfair** (10/10). Every single reviewer flagged this. The baselines use default-hyperparameter RF/XGBoost/LR on daily aggregates -- no deep learning (LSTM, Transformer, temporal CNN), no personalized models, no hyperparameter tuning, no multi-task learning. Xu: "embarrassingly weak." Choudhury: "weak strawmen." Campbell: "not how anyone actually deploys these models." Additionally, the ML baselines were evaluated on all 399 users while PULSE was evaluated on 50 high-compliance users, making the comparison asymmetric (Xu, Althoff). **This is the single most cited weakness across all reviews.**

2. **N=50 high-compliance users creates selection bias that undermines core claims** (10/10). All reviewers noted that selecting the top ~12% of users by compliance (mean 82.2 vs. 34.0 entries) introduces fundamental bias. These users may have more regular, predictable behavioral patterns. Nahum-Shani noted the paradox: "the system designed to address the diary paradox is evaluated only on users who don't exhibit it." Multiple reviewers called for evaluation on 100+ users including moderate-compliance participants.

3. **The "diary paradox" is named but never empirically validated** (9/10). The paper's central motivational framing -- that self-report data is absent when most needed -- is never tested because the 50 users are high-compliance. Reviewers requested: (a) testing Auto-Sense during diary-absent periods within users, (b) including lower-compliance users, or (c) significantly softening the framing. Spring noted this is a well-documented phenomenon (MNAR/EMA compliance) that the paper re-brands without testing.

4. **GLOBEM "ceiling-breaking" framing is misleading** (9/10). GLOBEM's ~0.52 BA was established on cross-dataset transfer across populations; PULSE evaluates within-dataset on curated users. Xu: "comparing apples to oranges." Campbell: "I know these benchmarks intimately... within-dataset performance is higher." Reviewers demand either removing the ceiling-breaking narrative or validating it with cross-dataset evaluation.

5. **No failure analysis or systematic investigation trace analysis** (8/10). The single qualitative example (Section 5.6) is called "cherry-picked" by multiple reviewers. Demanded: (a) systematic tool usage statistics (distribution of calls per prediction), (b) failure case analysis (when/why the agent makes wrong predictions), (c) investigation bias detection (confirmation bias, anchoring), (d) correlation between investigation depth and accuracy.

6. **Inference cost/latency prohibitive; no cost-effectiveness analysis** (8/10). 30-90 seconds per prediction with 6-12 API calls is too slow for real-time JITAI. No dollar cost reported. Xu estimated ~$80+ for the evaluation, ~$2,600/week at deployment scale. Mishra: the "just-in-time" premise is undermined by batch processing. Multiple reviewers requested a cost-performance Pareto analysis.

7. **Continuous calibration problem (3x over-prediction of negative affect) is more serious than acknowledged** (8/10). Predicted mean ~8.5-9.5 vs. ground truth ~3.1 is not "clinically cautious" but "clinically dangerous" (Nahum-Shani). The argument that binary thresholds absorb the bias only works for binary decisions, limiting the system's applicability. Narayanan frames this as an LLM alignment issue, not just calibration. Kim requests rank-order correlations to validate the binary preservation claim.

---

### Majority Concerns (cited by 4-6 reviewers -- SHOULD ADDRESS)

1. **No session memory ablation** (6/10 -- Choudhury, Campbell, Nahum-Shani, Althoff, Kim, Narayanan). The accumulated per-user session memory may be doing "a lot of heavy lifting" (Choudhury). Without ablating memory, we cannot distinguish whether the agentic advantage comes from dynamic tool use or accumulated longitudinal context. Campbell: "An agent that accumulates personalized knowledge over ~78 predictions could develop quite rich internal models."

2. **Structured baseline may be artificially weakened / compute confound** (6/10 -- Xu, Althoff, Kim, Spring, Narayanan, Campbell). The structured agent makes 1 LLM call vs. 6-12 for agentic. Althoff: "The agentic agents receive 6-12x more LLM reasoning tokens." Kim: modern prompting (chain-of-thought, self-consistency, self-reflection) could close part of the gap. A multi-turn structured baseline that controls for compute is needed.

3. **Single-model evaluation (Claude Sonnet only) undermines generalizability** (5/10 -- Choudhury, Kim, Xu, Narayanan, Barnes). The paper claims model-agnostic architecture but provides evidence from exactly one model. Kim: "In Health-LLM, we found significant performance variation across models." At minimum, one additional model (GPT-4 or open-source) should be tested.

4. **No sensitivity/specificity breakdown for clinical targets** (5/10 -- Nahum-Shani, Mishra, Spring, Barnes, Narayanan). BA (mean of sensitivity and specificity) hides the differential clinical costs of false positives (unnecessary interruptions) vs. false negatives (missed distress). Confusion matrices for ER_desire and INT_availability are essential for clinical interpretation.

5. **Clinical framing overreaches beyond the evidence** (5/10 -- Spring, Nahum-Shani, Barnes, Mishra, Campbell). The paper is positioned as advancing JITAI design for cancer survivors, but the evaluation is retrospective, no intervention outcomes are measured, no clinical workflow integration is discussed, and no patient/stakeholder input was sought. Recommendation: reframe as "prediction framework with proactive potential" rather than a "proactive system."

6. **Statistical reporting inconsistencies** (5/10 -- Althoff, Xu, Campbell, Choudhury, Kim). Multiple CI errors: Struct-Sense CI [0.505, 0.512] does not contain point estimate 0.516; Auto-Multi CI [0.634, 0.657] does not contain 0.660. The r=1.00 effect size for PA_State is suspicious. P-values like "p < 10^{-13}" convey false precision at N=50. These must all be verified and corrected.

7. **"Forecasting" target consistently below chance (BA < 0.50)** (5/10 -- Xu, Althoff, Barnes, Kim, Choudhury). This target drags down aggregate means and indicates the model is systematically wrong. Must be discussed, explained, and arguably excluded from aggregate metrics or analyzed separately.

8. **Cross-user RAG (find_peer_cases) introduces evaluation concern** (4/10 -- Xu, Mishra, Campbell, Kim). Providing ground truth outcomes from peer cases at inference time is a form of in-context learning from labeled examples that ML baselines do not receive. This affects the fairness of the ML comparison (though not the factorial comparisons).

9. **No temporal dynamics analysis** (4/10 -- Nahum-Shani, Barnes, Narayanan, Spring). Performance may vary by time of day, day of week, or study week. Early vs. late predictions (session memory accumulation) should be analyzed. JITAI systems must handle within-day temporal dynamics.

---

### Minority Concerns (cited by 1-3 reviewers -- CONSIDER)

1. **No physiological sensing (wearables, HRV, EDA)** (2/10 -- Barnes, Narayanan). For cancer survivors with treatment-related physiological changes, physiological signals could be highly informative. At minimum, discuss why omitted and how the architecture could extend.

2. **No comparison to other agentic approaches** (2/10 -- Campbell, Narayanan). PHIA uses code generation; GLOSS uses multi-agent debate. PULSE should be compared to alternative agentic paradigms, not just structured pipelines.

3. **No comparison to receptivity-specific baselines** (1/10 -- Mishra). Receptivity detection has established that specific contextual features (time since last notification, activity type, phone usage state) are particularly informative. A receptivity-tuned RF might close the gap for INT_availability.

4. **Privacy/ethics insufficient for clinical population** (2/10 -- Barnes, Nahum-Shani). Sending cancer survivors' behavioral data to a commercial LLM API raises concerns not adequately addressed. Barnes: "for clinical populations, on-device deployment should be a prerequisite, not an afterthought."

5. **No patient/stakeholder involvement in design** (2/10 -- Barnes, Spring). No needs assessment with cancer survivors or oncology providers to validate that prediction targets and system design match clinical needs.

6. **ER_desire construct validity and operationalization** (2/10 -- Nahum-Shani, Spring). The binarization procedure is never described (median split? clinical cutoff?). Spring notes that desiring emotion regulation is not the same as wanting a digital intervention.

7. **Multimodal integration is shallow** (1/10 -- Narayanan). "Multimodal" means sensing + text only. No speech, physiological, or acoustic analysis. No formal analysis of how the agent integrates across modalities.

8. **No analysis of iOS vs. Android performance differences** (2/10 -- Barnes, Campbell). With 36% Android users having additional modalities (app usage, ambient light), platform-stratified analysis is needed.

9. **Sensing data pipeline insufficiently described** (2/10 -- Campbell, Barnes). Sampling rates, preprocessing, gap-filling, sleep detection validation, and data coverage not reported. Important for reproducibility in IMWUT.

10. **Prediction-to-outcome causal chain untested** (3/10 -- Spring, Nahum-Shani, Barnes). Better prediction of tailoring variables does not necessarily translate to better intervention outcomes. This is a decision-theoretic problem, not just a prediction problem.

---

### Priority Revision List
(Ordered by: number of reviewers x severity)

1. **[10/10] Strengthen ML baselines.** Add at minimum: (a) one tuned deep learning baseline (LSTM or Transformer on temporal features), (b) one personalized model (per-user fine-tuning or user embeddings), (c) hyperparameter tuning for existing baselines. Evaluate all baselines on the same 50 users for fair comparison. This is unanimously cited and is the single most impactful revision.

2. **[10/10] Expand evaluation sample or temper claims proportionally.** Either: (a) evaluate on 100+ users including moderate-compliance participants, or (b) substantially reframe all claims to be explicitly scoped to high-compliance users, removing "ceiling-breaking" language and repositioning as a proof-of-concept.

3. **[9/10] Remove or reframe the GLOBEM "ceiling-breaking" narrative.** Either conduct cross-dataset evaluation on GLOBEM datasets, or reframe the comparison as within-dataset performance improvement over basic ML, not a paradigm-level ceiling breakthrough.

4. **[9/10] Empirically validate the diary paradox or soften the framing.** Either: (a) test Auto-Sense on diary-absent periods within users, (b) include low-compliance users and show Auto-Sense maintains performance, or (c) reframe as "passive sensing potential during compliance gaps" rather than a validated solution.

5. **[8/10] Add systematic investigation trace and failure analysis.** Report: tool call distribution per prediction, common investigation strategies, failure mode taxonomy, correlation between investigation depth and accuracy. Replace or supplement the cherry-picked example with systematic evidence.

6. **[8/10] Report inference cost and add cost-effectiveness analysis.** Report dollar cost per prediction, total evaluation cost, projected deployment cost. Include a cost-performance Pareto plot comparing all methods.

7. **[8/10] Address calibration more seriously.** Report rank-order correlations between predicted and actual continuous values. Discuss the LLM prior bias on affect scales. Frame the 3x over-prediction as a significant limitation, not a feature.

8. **[6/10] Ablate session memory.** Run the agentic agent without session memory to isolate the contribution of dynamic tool use vs. accumulated longitudinal context.

9. **[6/10] Add a multi-turn structured baseline.** Test a structured agent with chain-of-thought, self-consistency, or self-reflection to control for the compute advantage (6-12 calls vs. 1 call).

10. **[5/10] Fix all statistical reporting errors.** Verify and correct: CI for Struct-Sense, CI for Auto-Multi, r=1.00 for PA_State, p-value precision. This is a credibility issue.

11. **[5/10] Report sensitivity/specificity separately for clinical targets.** Provide confusion matrices for at least ER_desire and INT_availability across all conditions.

12. **[5/10] Evaluate on at least one additional LLM.** Test on GPT-4 or an open-source model (Llama 3) to validate that the agentic advantage is paradigm-level, not model-specific.

13. **[5/10] Discuss or exclude the "forecasting" target.** Explain sub-chance performance or exclude from aggregate metrics with justification.

14. **[5/10] Reframe clinical claims to match evidence.** Position as "prediction framework with proactive potential" rather than "proactive system." Acknowledge that prediction improvement does not automatically translate to intervention improvement.

15. **[4/10] Analyze temporal dynamics.** Report performance by time of day, study phase (early vs. late), and session memory accumulation effect.

---

### Questions Requiring Author Response
(Deduplicated, grouped by theme)

#### Evaluation Fairness & Baselines
- Can you run ML baselines on the same 50 high-compliance users? What BA do they achieve? (Xu, Althoff, Campbell)
- Did you attempt any hyperparameter tuning for the ML baselines? (Xu, Barnes)
- Is there overlap between the peer database and test fold users? Any risk of indirect label leakage? (Xu)
- Does session memory include the agent's own prior predictions? If so, could error propagation occur? (Choudhury)

#### Sample & Generalizability
- Can you stratify results by user compliance level within the 50 users? (Choudhury)
- How does PULSE perform on users with <50% EMA compliance? (Althoff, Mishra)
- What cancer types are represented? Are there sufficient numbers for subgroup analysis? (Barnes)
- How does performance differ by platform (iOS vs. Android)? (Campbell, Barnes)

#### Clinical Relevance & Construct Validity
- What is the binarization procedure for ER_desire and INT_availability? Threshold? (Nahum-Shani)
- Can you report sensitivity and specificity separately for clinical targets? (Nahum-Shani, Mishra, Spring)
- How often do high ER_desire and high INT_availability co-occur? (Spring)
- What fraction of intervention delivery decisions would actually change between Auto-Multi and Struct-Multi? (Spring)
- How does INT_availability correlate with simple heuristics like "screen is on" or "phone moved recently"? (Mishra)

#### Cost & Scalability
- What is the total dollar cost of all 27,300 inferences? Cost per prediction? (Xu, Althoff, Kim)
- What are the token counts (input + output) per prediction for agentic vs. structured? (Xu, Kim)
- Is the system approaching context window limits after 6-12 tool calls? (Kim)

#### Investigation Architecture
- What is the distribution of tool calls per prediction? (Campbell, Kim)
- Is there a diminishing returns curve for tool calls (e.g., capping at 4 vs. 16)? (Campbell, Kim)
- Does the agent develop consistent investigation strategies for different constructs? (Kim, Narayanan)
- Have you examined failure cases where agentic performs worse than structured? (Choudhury, Mishra, Narayanan)
- How does the agent handle conflicting signals? (Narayanan)

#### Model & Technical
- Why Claude Sonnet over GPT-4 or open-source alternatives? Was any comparison done? (Choudhury, Kim)
- What exact model version was used? (Campbell)
- How is the personal baseline computed for compare_to_baseline? Lookback window? Missing data handling? (Narayanan, Kim)
- What is the sensing data coverage for the 50 users? Fraction of expected data collected per modality? (Campbell)

#### Statistical
- Can you verify that all CIs contain their point estimates? (Althoff, Xu, Campbell)
- Can you confirm r=1.00 for PA_State with the raw Wilcoxon test statistic? (Althoff)

---

### Overall Assessment

**Would this paper be accepted at IMWUT?**

In its current form, likely not -- the 4 Major Revision votes and the depth of concern about baseline fairness would prevent acceptance. However, the path to acceptance is clear and achievable. The paper has genuine strengths that no reviewer disputes: the factorial design is rigorous, the INT_availability/ER_desire dissociation is a novel and important finding, and the agentic investigation paradigm is a meaningful conceptual contribution.

**Must-fix items (acceptance blockers):**
1. Add competitive ML baselines (tuned deep learning + personalized models) evaluated on the same 50 users
2. Fix all statistical reporting errors (CIs, effect sizes)
3. Remove or substantially qualify the GLOBEM "ceiling-breaking" framing
4. Either validate the diary paradox empirically or reframe it as untested motivation
5. Report inference cost per prediction

**Should-fix items (strengthen the paper substantially):**
6. Ablate session memory
7. Add a multi-turn structured baseline to control for compute
8. Add systematic investigation trace analysis (tool usage statistics, failure modes)
9. Report sensitivity/specificity for clinical targets
10. Evaluate on at least one additional LLM

**Nice-to-haves (differentiate a strong paper from an excellent one):**
11. Expand to 100+ users including moderate-compliance participants
12. Cross-dataset evaluation on a GLOBEM dataset
13. Platform-stratified analysis (iOS vs. Android)
14. Temporal dynamics analysis (time of day, study phase)
15. Cost-performance Pareto analysis across methods

**Bottom line:** The reviewers collectively see a paper with a strong core idea and clean experimental design, weighed down by overreaching claims, weak baselines, and an evaluation sample that contradicts its own motivating narrative. A focused revision addressing the top 5 must-fix items would likely convert all 4 Major Revision votes to Minor Revision, making acceptance probable. The underlying contribution -- that agentic investigation architecture matters more than data richness for sensing-based prediction -- is considered novel and important by the full panel.
