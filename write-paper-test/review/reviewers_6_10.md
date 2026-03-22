# PULSE Paper Reviews: Reviewers 6-10

---

# Review by Tim Althoff (Stanford, formerly UW)
## IMWUT Review

### Summary (2-3 sentences)

This paper introduces PULSE, a system that uses LLM agents equipped with eight sensing query tools to autonomously investigate passive smartphone sensing data for affect prediction and intervention opportunity detection in cancer survivors. The 2x2 factorial evaluation (agentic vs. structured x sensing-only vs. multimodal) on 50 users from the BUCS study demonstrates that the agentic architecture is the primary driver of prediction accuracy, and the paper reveals an interesting dissociation between behavioral (INT_availability) and emotional (ER_desire) constructs. The work claims to break through the ~0.52 balanced accuracy ceiling established by GLOBEM for sensing-based affect prediction.

### Overall Score: Major Revision
Confidence: High

### Strengths (ranked)

- S1: **Well-designed factorial experiment.** The 2x2 factorial crossing reasoning architecture with data modality is an elegant and rigorous design that cleanly isolates contributions. The within-subject comparisons across all 50 users for all conditions eliminate many confounds that plague between-subject comparisons in this space. This is genuinely good experimental methodology for a complex system evaluation.

- S2: **Clinically meaningful finding on construct dissociation.** The demonstration that INT_availability is fundamentally behavioral (best predicted by sensing alone, 0.706 BA) while ER_desire is psychological (benefiting substantially from diary text, delta=0.093) is the paper's most important conceptual contribution. This has clear implications for JITAI design and is well-grounded in the constructs' theoretical definitions.

- S3: **Scale and comprehensiveness of evaluation.** Approximately 27,300 total LLM inferences across 7 conditions, 50 users, and 16 prediction targets is a substantial evaluation. The inclusion of representativeness analysis comparing the 50 pilot users to the remaining 349 BUCS participants demonstrates methodological care.

- S4: **Honest limitations section.** The paper transparently acknowledges retrospective evaluation, model dependency, inference cost, lack of memory ablation, and calibration issues. This is refreshingly candid for a systems paper.

### Weaknesses (ranked)

- W1: **Statistical reporting is misleading in several places.** The abstract reports "0.660 mean balanced accuracy" and "p < 10^{-11}, r = 0.95," but the 95% CI in Table 4 for Auto-Multi is [0.634, 0.657], which does not contain the point estimate of 0.660. This suggests the point estimate and the CI may come from different computation methods (entry-level vs. per-user), which is confusing. Similarly, the effect size r=1.00 for PA_State multimodal (Table 5) is mathematically suspicious -- a Wilcoxon test rarely yields a perfect rank-biserial correlation. This needs careful verification. More broadly, reporting p-values like "p < 10^{-13}" on N=50 users conveys false precision; the test statistic depends heavily on distributional assumptions at this sample size.

- W2: **Sample selection bias is underacknowledged.** The 50 users were selected for high EMA compliance (mean 82.2 entries vs. 34.0 for the rest). This is not just a minor caveat -- it fundamentally compromises the paper's motivating argument about the "diary paradox." The entire premise is that diary data is systematically missing when most needed, yet the evaluation is conducted exclusively on users who almost never miss diary entries. The representativeness analysis (Table 3) shows small effect sizes on most measures, but it does not test the key variable: these users have 2.4x the compliance rate. The paper should be far more cautious in claiming relevance to the diary paradox scenario when the evaluation sample is the antithesis of that scenario.

- W3: **ML baseline comparison is unfair and somewhat misleading.** The ML baselines were trained on 399 users via 5-fold CV, while PULSE conditions were evaluated on 50 users. The paper acknowledges this asymmetry, but then proceeds to frame the comparison as "breaking through the ML ceiling" throughout the paper. The 50 high-compliance users may be substantially easier to predict than the full 399. A fair comparison would require running the ML baselines on the same 50 users, which seems straightforward to do. Without this, the "ceiling-breaking" framing is unsupported.

- W4: **No cost-effectiveness analysis.** Each prediction requires 30-90 seconds and multiple API calls, costing real money and requiring cloud infrastructure. The paper does not quantify the dollar cost per prediction, which is critical for any clinical deployment argument. For 3,900 predictions per condition x 7 conditions = 27,300 calls on Claude Sonnet, this is likely thousands of dollars. How does this compare to a simpler model that achieves, say, 0.58 BA at near-zero marginal cost?

- W5: **Agentic "advantage" may be partially confounded with compute.** The structured agents make 1 LLM call per prediction; the agentic agents make 6-12. The paper dismisses this concern by arguing there is "no information asymmetry" since structured agents receive all data. But the agentic agents receive 6-12x more LLM reasoning tokens and have opportunities for error correction through multi-turn interaction. A fairer baseline would be a structured agent that receives multiple rounds of reasoning (e.g., chain-of-thought with self-refinement) on the same pre-formatted data, without tool use. This would isolate tool use from multi-turn reasoning.

- W6: **Confidence intervals are suspiciously narrow for some conditions.** Struct-Sense achieves 0.516 [0.505, 0.512] -- the CI doesn't even contain the point estimate. This appears to be an error. Several other CIs seem too narrow for N=50.

### Questions for Authors

1. Can you confirm that the point estimates and confidence intervals in Table 4 are computed consistently? The CI for Auto-Multi [0.634, 0.657] excludes the reported mean of 0.660.

2. What is the total dollar cost of running all 27,300 LLM inferences? How does the cost per prediction compare to running a random forest?

3. Have you evaluated the ML baselines on the same 50 high-compliance users? If the ML baselines achieve, say, 0.55 on these users instead of 0.52, that substantially changes the narrative about "ceiling-breaking."

4. For the r=1.00 effect size on PA_State multimodal: can you provide the raw Wilcoxon test statistic and confirm that every single user showed improvement?

5. The "diary paradox" motivation is compelling, but how does Auto-Sense perform on the users who actually exhibit diary non-compliance? Have you tested on users with, say, <50% compliance?

### Detailed Comments

The paper is ambitious and tackles an important problem at the intersection of sensing, LLMs, and clinical intervention. The factorial design is genuinely well-conceived. However, the statistical reporting needs significant cleanup, and the framing overpromises relative to what the evaluation actually demonstrates.

The "diary paradox" narrative is set up beautifully in the introduction but then not actually tested. You evaluate on the highest-compliance users in the dataset. A truly compelling demonstration would show that Auto-Sense maintains useful predictions on users whose compliance drops over time -- perhaps splitting each user's timeline into high-compliance and low-compliance periods and showing graceful degradation within users.

The calibration analysis (Section 5.6) is concerning. The agentic agents over-predict negative affect by a factor of ~3x (predicted mean ~9 vs. ground truth ~3.1). The paper argues this doesn't matter because binary thresholds are individualized, but this is a large systematic bias that suggests the LLM has fundamental miscalibration on continuous affect scales. This deserves more attention -- if the system were deployed, the continuous predictions would be clinically misleading.

The qualitative example (Section 5.8) of the agentic agent's investigation is illustrative but reads as cherry-picked. How representative is this? What does a failure case look like?

### Minor Issues

- Line 8: `\documentclass[acmart]{acmart}` -- the option and class are both "acmart"; should this be `\documentclass{acmart}`?
- Table 3: The CI for Struct-Sense [0.505, 0.512] excludes its own point estimate of 0.516. This appears to be a computational error.
- The paper uses ~3,900 predictions per condition but doesn't clearly state how many entries each user contributes on average, making it hard to assess per-user sample sizes.
- The "forecasting" target achieves BA below 0.50 for most conditions (Table A1), suggesting the models are actually worse than chance for this target. This deserves commentary.
- Several p-values are reported as "$< 10^{-X}$" but the exact values would be more appropriate given the sample size.

### What would change your score?

To move toward Accept: (1) Fix all statistical reporting inconsistencies (CIs that don't contain point estimates, verify r=1.00). (2) Run ML baselines on the same 50 users for fair comparison. (3) Add a multi-turn structured baseline that controls for compute. (4) Report inference cost. (5) Test on lower-compliance users to validate the diary paradox claim. To move toward Reject: If the statistical issues turn out to reflect deeper methodological problems (e.g., data leakage, incorrect test implementation), this would be disqualifying.

---

# Review by Varun Mishra (Northeastern)
## IMWUT Review

### Summary (2-3 sentences)

PULSE presents an agentic LLM system with eight MCP-based sensing tools for predicting affect states and intervention opportunities in cancer survivors. The paper uses a 2x2 factorial design to isolate the contribution of agentic investigation (vs. structured pipelines) and data modality (sensing-only vs. multimodal), evaluated on 50 users from the BUCS dataset. The key findings are that agentic reasoning is the dominant factor in prediction accuracy, and that intervention availability (INT_availability) is best predicted by passive sensing alone -- a finding with direct implications for JITAI and receptivity detection.

### Overall Score: Minor Revision
Confidence: High

### Strengths (ranked)

- S1: **Directly addresses the core receptivity detection challenge.** The decomposition of intervention receptivity into ER_desire (psychological need) and INT_availability (behavioral readiness), and the finding that they are best predicted by fundamentally different data sources, is the most important contribution of this paper from a JITAI perspective. This aligns with and extends the theoretical framework from Kunzler et al. (2019) and our own work on receptivity detection (Mishra et al., 2021), providing the first empirical demonstration that these components require different sensing modalities. This finding alone justifies publication.

- S2: **Sensing-only agent achieves clinically meaningful performance.** Auto-Sense achieving 0.706 BA on INT_availability from passive sensing alone -- exceeding all diary-dependent methods including CALLM (0.542) -- is a striking result. For real-world JITAI deployment, this is the critical capability: detecting behavioral availability without requiring any user input. This validates the passive sensing approach that our community has been pursuing for over a decade.

- S3: **The MCP tool design is thoughtful and reusable.** The eight tools (daily summary, behavioral timeline, query_sensing, query_raw_events, compare_to_baseline, receptivity_history, find_similar_days, find_peer_cases) mirror the kind of investigation a researcher would do when analyzing behavioral data. The temporal boundary enforcement preventing future data access is critical for validity. This tool set is a genuine contribution that others can build on.

- S4: **Rigorous within-subject factorial design.** Every comparison is within-subject on the same 50 users, which is the gold standard for this kind of evaluation. The Wilcoxon signed-rank tests with effect sizes are appropriate for the non-normal distributions typical of per-user balanced accuracy.

- S5: **Strong connection to JITAI theory.** The paper does an excellent job connecting empirical results to JITAI design principles (Nahum-Shani et al., 2018). The four design principles in Section 6.6 are grounded in the data and actionable.

### Weaknesses (ranked)

- W1: **No comparison to receptivity-specific baselines.** The paper cites our receptivity detection work (Mishra et al., 2021) and Kunzler et al. (2019) but does not compare against any receptivity-specific models or features. The ML baselines use generic sensing features (motion, screen, GPS, etc.) with generic classifiers. Receptivity detection has established that specific contextual features -- time since last notification, current activity type, phone usage state, social context -- are particularly informative. A random forest trained specifically on receptivity-relevant features might substantially close the gap with PULSE for INT_availability. Without this comparison, we cannot assess how much of PULSE's advantage for INT_availability comes from the agentic architecture vs. simply having a richer feature representation through the tool interface.

- W2: **Latency is prohibitive for real-time JITAI deployment.** 30-90 seconds per prediction is far too slow for a real-time intervention delivery system. When a user transitions to an available state (e.g., finishes a meeting, puts down their phone after a social media session), that window of availability may last only minutes. The paper acknowledges this in limitations but does not adequately address it. Batch processing every few hours, as suggested, fundamentally undermines the "just-in-time" premise -- you cannot deliver an intervention at the right moment if your state estimate is hours old.

- W3: **The "diary paradox" is not empirically validated in this evaluation.** The paper's strongest motivating argument -- that self-report data is absent when most needed -- is never tested. All 50 users are high-compliance. To validate the diary paradox claim, the authors should: (a) identify periods of diary non-compliance within users and test Auto-Sense during those periods, or (b) include lower-compliance users and show that Auto-Sense maintains performance where multimodal methods cannot operate.

- W4: **No analysis of when the agentic agent fails.** The paper presents a compelling success case (Section 5.8) but no failure analysis. From my experience with sensing-based systems, the most informative analysis is understanding failure modes: when does the agent make confident but wrong predictions? Are there systematic user-level or context-level factors that predict failure? This would be far more useful for the community than additional success stories.

- W5: **Cross-user RAG introduces a subtle evaluation concern.** The find_peer_cases tool provides ground truth EMA outcomes from other users. While the current user is excluded from their own peer database, this means the agent is receiving labeled training examples at inference time. This is a form of transductive learning that the ML baselines do not benefit from. To be fair, the structured agents also receive cross-user calibration examples, so this doesn't affect the factorial comparisons. But it does affect the comparison to ML baselines, which receive no labeled examples at test time.

### Questions for Authors

1. What is INT_availability's base rate in the 50-user sample? If it's high (e.g., 0.643 as in Table 3), then a BA of 0.706 means the model is only modestly above a majority-class predictor. What is the sensitivity and specificity breakdown?

2. How does the agent's INT_availability prediction correlate with simple heuristics like "screen is on" or "phone moved in last 10 minutes"? These are the kinds of signals that receptivity detection research has found most predictive, and it would be valuable to understand whether PULSE is discovering anything beyond these.

3. For find_peer_cases: how many calibration examples does the agent typically receive, and how strongly do the retrieved examples' labels correlate with the target user's actual outcome? If the correlation is high, the agent may essentially be performing a weighted k-NN lookup rather than genuine "reasoning."

4. The paper reports BA but not sensitivity/specificity separately. For clinical applications, false negatives (missing distressed states) and false positives (unnecessary interruptions) have very different costs. Can you provide the confusion matrices for ER_desire and INT_availability?

5. How does performance change across the study timeline? Does the per-user session memory improve predictions for later entries compared to earlier ones?

### Detailed Comments

This paper makes a genuine contribution to the receptivity detection and JITAI literature. The construct dissociation finding (W: different modalities for desire vs. availability) is the kind of result that should reshape how we build intervention delivery systems.

However, I am concerned that the paper oversells the "paradigm shift" narrative while underselling the practical limitations. The agentic approach is compelling in a research evaluation, but the 30-90 second latency, the dependency on a commercial API (Claude Sonnet), and the lack of any real-time deployment testing make the JITAI implications speculative. The paper would be stronger if it more honestly positioned itself as a proof-of-concept that demonstrates the value of agentic reasoning for behavioral sensing, with deployment as a future challenge.

The connection to the GLOSS system (Choube et al., 2025) deserves deeper discussion. GLOSS uses multi-agent sensing analysis with code generation -- how does this compare architecturally to PULSE's tool-use approach? Both are "agentic" in some sense, but the investigation strategies differ substantially.

I appreciate the careful representativeness analysis (Table 3), but the key missing comparison is on EMA compliance patterns, not just base rates. The high-compliance users may have more regular behavioral patterns that are inherently easier to predict, regardless of method.

### Minor Issues

- The paper should cite Mehrotra et al. (2016) on notification timing and interruptibility, which is directly relevant to INT_availability.
- Table 1 labels GLOSS as "Partial" agentic but does not explain why code generation is considered less agentic than tool use.
- The term "diary paradox" is presented as novel but the phenomenon is well-documented in the EMA literature (Stone & Shiffman, 2002; Wen et al., 2017). The contribution is naming it, not discovering it.
- Auto-Multi+ shows negligible improvement over Auto-Multi (0.661 vs. 0.660), suggesting the pre-computed behavioral narrative adds nothing. This null result deserves more discussion -- it suggests the agentic agent's own investigation is sufficient and additional pre-processing is redundant.

### What would change your score?

To Accept: (1) Add sensitivity/specificity breakdowns for the clinical targets. (2) Include a failure analysis showing when and why the agent makes wrong predictions. (3) Either validate the diary paradox claim empirically (test on low-compliance periods/users) or soften the framing. (4) Discuss latency more seriously with concrete paths to real-time operation. These are all achievable in a revision. To Major Revision: If the statistical inconsistencies flagged by other reviewers reveal deeper problems, or if the INT_availability result turns out to be largely explained by simple heuristics.

---

# Review by Laura Barnes (UVA)
## IMWUT Review

### Summary (2-3 sentences)

PULSE is an LLM agent system that performs "agentic sensing investigation" of passive smartphone data for affect prediction and intervention opportunity detection in cancer survivors. Using eight MCP sensing tools, the agent autonomously queries behavioral data, compares to personal baselines, and retrieves similar cases to make predictions about 16 emotional and behavioral targets. The 2x2 factorial evaluation on 50 BUCS participants demonstrates that the agentic architecture provides statistically significant improvements over structured pipelines, with particular strength in predicting intervention availability from sensing alone.

### Overall Score: Minor Revision
Confidence: High

### Strengths (ranked)

- S1: **Clinical population and meaningful prediction targets.** This is one of the few LLM-for-sensing papers evaluated on a clinical population (cancer survivors) rather than convenience samples of college students. The prediction targets -- emotion regulation desire and intervention availability -- are clinically grounded and directly relevant to the growing need for supportive care technologies in cancer survivorship. The paper correctly identifies that cancer survivors face unique challenges including treatment-related fatigue, fear of recurrence, and fluctuating distress that make passive monitoring particularly valuable.

- S2: **Strong methodological design.** The 2x2 factorial with within-subject comparisons is the right approach for isolating the effect of architectural choices. The representativeness analysis (Table 3) shows commendable awareness of selection bias. The temporal boundary enforcement in the tools prevents data leakage, which is a common pitfall in retrospective evaluations. The inclusion of per-user statistics (means, SDs, CIs) rather than just aggregate numbers reflects good statistical practice.

- S3: **The behavioral vs. emotional construct dissociation.** The finding that INT_availability is a behavioral construct (best captured by sensing) while ER_desire is a psychological construct (requiring diary text) is well-supported by the data and has practical implications for designing sensing-based support systems for chronic disease populations. This maps onto the clinical reality: a cancer survivor's physical availability for intervention engagement is observable through behavior, while their emotional need for support requires some form of self-expression.

- S4: **Graceful degradation is clinically important.** The demonstration that Auto-Sense maintains useful performance (0.589 overall, 0.706 for INT_availability) without any diary input addresses a real clinical need. In my experience with sensing studies in chronic disease populations, participant fatigue with self-report is the primary threat to longitudinal data collection. A system that degrades gracefully rather than failing completely when self-report stops is essential for real-world deployment.

- S5: **Transparent about calibration limitations.** The acknowledgment that the agentic agents systematically over-predict negative affect (predicted ~8.5-9.5 vs. ground truth ~3.1) is important. Many papers in this space would bury such a finding. The discussion of why binary classification is preserved despite continuous miscalibration is thoughtful.

### Weaknesses (ranked)

- W1: **No consideration of physiological sensing.** The paper uses only smartphone sensing (motion, screen, GPS, etc.) but does not discuss or evaluate wearable physiological sensors (heart rate, HRV, electrodermal activity, skin temperature) that are increasingly available through smartwatches and fitness trackers. For a clinical population like cancer survivors who experience treatment-related physiological changes (chemotherapy-induced fatigue, autonomic dysfunction, sleep architecture changes), physiological signals could be highly informative. The paper should at least discuss why physiological sensing was not included and how the tool-based architecture could extend to incorporate it.

- W2: **50 users is small for generalizability claims, especially in a heterogeneous clinical population.** Cancer survivors vary enormously by cancer type, treatment history, time since diagnosis, age, comorbidities, and socioeconomic factors. 50 users, even with ~3,900 predictions each, cannot capture this heterogeneity. The paper acknowledges sample size as a limitation but does not report any subgroup analyses -- e.g., does PULSE perform differently for breast cancer vs. prostate cancer survivors? For users closer to vs. further from treatment? For users with vs. without comorbid depression? These analyses would be essential for understanding clinical applicability.

- W3: **The ML baselines are too weak.** Random Forest, XGBoost, and Logistic Regression with default hyperparameters and daily-aggregated features represent the most basic ML pipeline. The sensing literature has moved well beyond this: deep learning approaches (LSTMs, transformers), personalized models, multi-task learning, and temporal convolutional networks have all been shown to improve over simple aggregation. The GLOBEM benchmark itself includes 18 algorithms, many more sophisticated than what is tested here. By using only the weakest baselines, the paper inflates the perceived advantage of PULSE. At minimum, the authors should include a personalized model (e.g., user-specific fine-tuning) and a temporal model (e.g., LSTM on hourly features) to establish a stronger reference point.

- W4: **No longitudinal analysis of prediction quality.** The paper treats all EMA entries equally, but in a 5-week study, prediction quality may vary systematically over time. Early predictions may be worse (less session memory, less baseline data) while later predictions may benefit from accumulated context. Alternatively, model drift or sensor fatigue could degrade later predictions. A temporal analysis of prediction accuracy would reveal important dynamics.

- W5: **Privacy and data handling discussion is insufficient for a clinical population.** The paper mentions that data is de-identified and processed with safeguards, but cancer survivors' behavioral data is particularly sensitive -- location patterns may reveal oncology visits, communication patterns may reveal disclosure decisions, and sleep disruption patterns may indicate symptom burden. Sending this data to a commercial LLM API (Claude/Anthropic) raises significant concerns that are not adequately addressed. The paper mentions on-device deployment as future work, but for clinical populations, this should be a prerequisite, not an afterthought.

- W6: **No patient/stakeholder involvement in design.** The system was designed without apparent input from cancer survivors, oncology providers, or cancer support organizations. For a system intended to serve a clinical population, participatory design or at least needs assessment with stakeholders should inform the tool design, prediction targets, and deployment architecture. What do cancer survivors actually want from a passive monitoring system? The paper assumes the prediction targets are the right ones but does not validate this assumption.

### Questions for Authors

1. The BUCS dataset was collected with the AWARE framework -- what is the battery drain and phone performance impact of continuous sensing? This is particularly relevant for cancer survivors who may rely on their phones for medical communication.

2. How does the system handle the significant platform differences between iOS and Android? With 36% Android users having app usage and ambient light data while iOS users do not, are there systematic performance differences by platform?

3. The session memory accumulates across a user's entries -- is there a risk of confirmation bias, where early (possibly incorrect) impressions about a user's patterns get reinforced in later predictions?

4. What cancer types are represented in the 50-user sample? Are there sufficient numbers of any single cancer type to analyze type-specific performance?

5. The paper mentions sleep detection based on accelerometer data. How was sleep validated? Sleep is notoriously difficult to detect from phone accelerometry alone, and errors in sleep estimation would propagate through the agent's reasoning.

### Detailed Comments

This paper addresses an important clinical need -- supporting cancer survivors' mental health through passive sensing and intelligent prediction. The agentic investigation paradigm is a creative approach that leverages LLMs' reasoning capabilities in a novel way. The construct dissociation finding has genuine clinical implications.

However, I am concerned about the gap between the paper's clinical framing and its actual evaluation. The paper is positioned as a step toward clinical deployment, but fundamental questions about clinical relevance remain unaddressed: no patient involvement, no discussion of clinical workflow integration, no analysis of clinically meaningful outcomes (does better prediction lead to better interventions?), and no consideration of the populations most at risk (those with low compliance, high distress, or complex treatment histories).

The comparison to the "ML ceiling" of ~0.52 is used repeatedly but is misleading. This ceiling was established on different datasets (GLOBEM uses college student data), different populations, different prediction targets, and different evaluation protocols. The BUCS-specific ML baselines (Table 4) achieve 0.514-0.518, which is consistent with the ceiling, but these baselines use minimally-tuned models with basic features. A dedicated ML effort on the BUCS data with modern techniques might achieve meaningfully higher performance.

The ethical considerations section, while present, reads as an afterthought. For a clinical population, data ethics should be a design driver, not a post-hoc justification.

### Minor Issues

- The paper does not mention informed consent specifically for LLM analysis of behavioral data -- was this covered in the original BUCS consent?
- Table 6: the "forecasting" target shows BA below 0.50 for most conditions, meaning the models are systematically wrong. This should be flagged and explained.
- The paper could benefit from a figure showing the temporal flow of a prediction (tools called, time spent, information gathered) to make the agentic investigation more concrete.
- Reference to "cancer survivorship" should specify that this includes active surveillance and post-treatment survivorship, as these have different behavioral profiles.

### What would change your score?

To Accept: (1) Include stronger ML baselines (at least one personalized model and one temporal model). (2) Add subgroup analyses by cancer type or treatment stage if sample permits. (3) Expand the privacy discussion with concrete data handling details for clinical deployment. (4) Add a platform-stratified analysis (iOS vs. Android). These revisions are feasible. To Major Revision: If stronger baselines substantially narrow the gap, or if platform-stratified analysis reveals that the results are driven primarily by Android users with richer sensing data.

---

# Review by Bonnie Spring (Northwestern)
## IMWUT Review

### Summary (2-3 sentences)

PULSE is an LLM agent-based system that predicts emotional states and intervention readiness in cancer survivors by autonomously investigating passive smartphone sensing data. The paper uses a factorial evaluation design to show that agentic reasoning architecture, more than data modality, drives prediction accuracy. The authors frame the work as advancing JITAI design by enabling separate estimation of desire for support (ER_desire) and behavioral availability (INT_availability) -- two components they argue should be treated as distinct tailoring variables.

### Overall Score: Major Revision
Confidence: High

### Strengths (ranked)

- S1: **Theoretically grounded in the JITAI framework.** The paper correctly identifies ER_desire and INT_availability as mapping to JITAI tailoring variables, and the finding that these require different data modalities is actionable for intervention design. The four design principles (Section 6.6) are well-articulated and grounded in the data, connecting prediction capabilities to the practical demands of adaptive intervention delivery. This is a rare paper in the computing literature that takes the intervention science framework seriously.

- S2: **Addresses the diary paradox, a genuine clinical challenge.** The observation that self-report data is systematically missing during periods of greatest clinical need is well-documented in the EMA literature and remains one of the biggest barriers to mHealth intervention deployment. While the evaluation doesn't fully test this (see weaknesses), the paper's framing correctly identifies this as the central challenge and designs the system to address it through sensing-only capabilities.

- S3: **The factorial design provides clear evidence.** Within-subject comparisons on the same users, with the same data, varying only architecture and modality, is the right approach. The consistency of the agentic advantage across sensing-only and multimodal conditions strengthens the claim that this is an architectural benefit, not a data artifact.

- S4: **Acknowledgment that binary decisions are the clinically relevant outcome.** The paper's discussion of why binary classification performance matters more than continuous calibration for JITAI deployment is correct. Intervention delivery decisions are indeed binary (intervene or not), and the individualized thresholds absorb systematic biases in continuous predictions. This shows understanding of the clinical use case.

### Weaknesses (ranked)

- W1: **No evidence that improved prediction translates to improved intervention outcomes.** This is the fundamental limitation from a behavior change perspective. The paper assumes that better prediction of tailoring variables leads to better intervention delivery, which leads to better health outcomes. But this causal chain is entirely untested. In the JITAI literature, we have seen cases where technically superior tailoring does not improve outcomes because (a) the intervention content is insufficient, (b) the decision rules are not well-calibrated, or (c) the prediction improvements are too small to change delivery decisions in practice. A BA improvement from 0.603 to 0.660 -- does this actually change when the system would deliver interventions? What fraction of delivery decisions would differ? Without this analysis, the clinical significance of the prediction improvements is unknown.

- W2: **ER_desire as a prediction target conflates measurement with intervention need.** The paper equates high emotion regulation desire with "the user wants support," but ER_desire (as measured in the EMA) captures whether the user currently desires to regulate their emotions -- not necessarily whether they want an intervention from a digital system. A cancer survivor may desire emotion regulation but prefer to achieve it through talking to their partner, going for a walk, or praying. The paper should be more careful in distinguishing between the psychological construct and its implications for digital intervention delivery.

- W3: **No consideration of the intervention component of JITAI.** The paper focuses entirely on the prediction (tailoring variable estimation) component of a JITAI but says nothing about what intervention would be delivered, how the prediction would be translated into a decision rule, or what the proximal outcomes would be. The four design principles in Section 6.6 are stated abstractly but never operationalized. What does "require evidence for both [desire and availability]" mean in terms of a threshold? What is the cost of a false positive (unnecessary interruption) vs. false negative (missed distress)? These questions cannot be answered without specifying the intervention component.

- W4: **The cancer survivorship framing adds clinical gravity but the evaluation does not validate clinical applicability.** The paper emphasizes cancer survivors throughout, but the evaluation is a retrospective replay of existing data. There is no assessment of whether cancer-specific factors (treatment phase, symptom burden, fear of recurrence, oncology appointments) influence prediction accuracy, no consultation with oncology providers about clinical relevance, and no consideration of how the system would integrate into cancer survivorship care pathways. The paper would be equally valid -- and perhaps more honest -- if it were framed as a general affect prediction evaluation that happened to use cancer survivor data.

- W5: **The "diary paradox" is named but not tested.** This is a significant gap between the paper's narrative and its evidence. The paper defines the diary paradox as "the very moments when intervention is most needed are precisely when self-report data is absent." To test this, you would need to show that (a) diary absence correlates with higher distress, and (b) Auto-Sense maintains useful predictions during these absent periods. Neither is demonstrated. The 50 high-compliance users provide the opposite test case.

- W6: **Missing operationalization of "clinically useful."** The paper repeatedly uses phrases like "clinically useful predictions" and "clinically meaningful performance" without defining what these terms mean. What BA threshold constitutes "clinically useful"? In the depression screening literature, screening tools typically require sensitivity > 0.80 and specificity > 0.80 for clinical use. The reported BAs of 0.55-0.75 may or may not meet clinical utility thresholds depending on the specific targets and use cases.

### Questions for Authors

1. For the 50 users evaluated, how many intervention delivery decisions would actually change between Auto-Multi and Struct-Multi? That is, for what fraction of EMA entries do the two conditions make different predictions?

2. Was the ER_desire measure validated specifically for cancer survivors? What scale was used, and what is its relationship to established constructs like emotional support seeking or help-seeking behavior?

3. The paper mentions that JITAI systems should intervene when both desire and availability are present. In the 50-user sample, how often do high ER_desire and high INT_availability co-occur? Is this a common or rare state?

4. Have you considered or analyzed whether the agentic agent's predictions are stable over short time intervals? If a user completes two EMAs 4 hours apart with similar sensing data, does the agent produce consistent predictions?

5. The session memory accumulates agent reasoning across entries. Does this create a dependency between predictions that violates the independence assumption in your statistical tests?

### Detailed Comments

This paper represents an interesting technical contribution -- the agentic sensing investigation paradigm is novel and the factorial design is rigorous. However, the clinical framing significantly overpromises relative to what the evaluation demonstrates.

The paper's strongest contribution is the architectural insight: LLM agents with sensing tools outperform fixed pipelines. This is a computing contribution. The clinical contribution -- advancing JITAI design for cancer survivors -- is aspirational, not demonstrated. I would encourage the authors to either (a) refocus the framing on the technical contribution, positioning clinical application as future work, or (b) include analyses that directly address clinical relevance (decision analysis, threshold-based evaluation, co-occurrence analysis of desire and availability, subgroup analysis by clinical characteristics).

The four design principles for next-generation JITAI systems (Section 6.6) are interesting but disconnected from any empirical test. In particular, "intervention delivery should require evidence for both [desire and availability]" implies an AND rule that would substantially reduce intervention delivery frequency. Has this been analyzed? In my experience with JITAI design, overly conservative delivery rules lead to insufficient intervention dose, which compromises efficacy.

The comparison to our work on cancer survivorship mHealth (cited as Spring et al., 2019) is appreciated, but the paper should note that the intervention optimization problem is not solely about better prediction -- it's about calibrating the prediction to the intervention. A system that perfectly predicts high distress is only useful if it can deliver an intervention that helps with distress.

### Minor Issues

- The abstract mentions "over 27,300 total inferences" -- this is a measure of computational cost, not a strength. It should not be in the abstract.
- Section 6.6 references "growing evidence base for JITAI effectiveness" but the cited meta-analysis (2025) found only "small but significant" effects. The paper should not overstate the evidence for JITAI efficacy.
- The term "diary paradox" is catchy but potentially misleading -- the phenomenon is EMA non-compliance or missing data at random (MNAR), which has an established literature. Introducing a new term risks obscuring the connection to existing methodological work.
- The paper does not discuss how the prediction system would be evaluated in a prospective trial. A microrandomized trial is mentioned in future work, but even the prediction system would need prospective validation before being embedded in an MRT.

### What would change your score?

To Minor Revision: (1) Include a decision analysis showing how prediction differences translate to intervention delivery differences. (2) Analyze co-occurrence of high ER_desire and high INT_availability. (3) Either soften the clinical framing or add clinically relevant analyses (subgroup, threshold-based utility). (4) Acknowledge and integrate the existing MNAR/EMA compliance literature rather than rebranding it as "diary paradox." To Accept: All of the above, plus empirical validation of the diary paradox (test on diary-absent periods) and operationalization of "clinically useful."

---

# Review by Yubin Kim (MIT Media Lab)
## IMWUT Review

### Summary (2-3 sentences)

PULSE equips LLM agents with eight MCP-based sensing tools to autonomously investigate passive smartphone data for affect prediction in cancer survivors, replacing fixed feature pipelines with agentic, hypothesis-driven behavioral investigation. The 2x2 factorial evaluation on 50 BUCS participants (~27,300 total inferences) shows that agentic reasoning is the primary driver of performance, with Auto-Multi achieving 0.660 mean BA vs. 0.603 for Struct-Multi. The paper contributes an interesting architectural paradigm and reveals a meaningful dissociation between behavioral and emotional prediction targets.

### Overall Score: Minor Revision
Confidence: High

### Strengths (ranked)

- S1: **The agentic investigation paradigm is a genuine architectural contribution.** Moving from "give the LLM all the data and ask for a prediction" to "give the LLM tools and let it investigate" is a meaningful conceptual shift. This extends the Health-LLM paradigm (Kim et al., 2024) in an important direction: rather than optimizing prompting strategies for a single-pass LLM call, PULSE optimizes the investigation process itself. The tool design -- especially compare_to_baseline for personalized z-scores and find_similar_days for analogical reasoning -- captures the kind of contextual reasoning that we found challenging to elicit through prompting alone in Health-LLM.

- S2: **The factorial design cleanly isolates the agentic contribution.** In Health-LLM, we found it difficult to disentangle the effects of model choice, prompting strategy, and data representation. PULSE's factorial design addresses this directly: holding data constant, the agentic architecture provides consistent improvements. This is the kind of controlled ablation that the LLM-for-health community needs more of.

- S3: **MCP as the tool interface is a forward-looking design choice.** Using an open standard for tool integration means the architecture is model-agnostic in principle. This addresses one of the key limitations we identified in Health-LLM: the coupling between model and evaluation. While PULSE only evaluates on Claude Sonnet, the MCP interface makes cross-model comparison straightforward.

- S4: **The cross-user RAG for calibration is a novel use of retrieval.** Using RAG not for knowledge retrieval but for empirical calibration -- grounding predictions in the distribution of actual outcomes from similar cases -- is clever and potentially addresses the calibration challenges we documented in Health-LLM. This is distinct from standard RAG and represents a meaningful methodological contribution.

- S5: **16 prediction targets with per-target analysis.** The comprehensive evaluation across 16 targets, with detailed per-target analysis in the appendix, allows the community to understand where the agentic approach helps most and where it provides marginal benefit. The "forecasting" target consistently underperforming (BA < 0.50) is an honest data point that adds credibility.

### Weaknesses (ranked)

- W1: **Single model evaluation undermines the generalizability claim.** The paper evaluates only Claude Sonnet. While it correctly notes that the architecture is model-agnostic, without empirical evidence from at least one other model (GPT-4, Llama 3, Gemini), we cannot assess whether the agentic advantage is a property of the investigation paradigm or of Claude Sonnet's specific capabilities. In Health-LLM, we found significant performance variation across models (GPT-4 vs. GPT-3.5 vs. Gemini vs. Llama 2), and the model-task interaction was substantial. The authors should evaluate on at least one open-source model to establish a lower bound on the paradigm's generalizability.

- W2: **No analysis of token cost and reasoning efficiency.** The paper reports that agentic agents make 6-12 tool calls per prediction, but does not analyze the relationship between the number of tool calls and prediction quality. Is there diminishing returns after, say, 6 calls? Do some targets benefit more from deeper investigation? In Health-LLM, we found that more reasoning (e.g., chain-of-thought) does not uniformly improve all tasks. An analysis of reasoning efficiency -- accuracy as a function of investigation depth -- would be informative and practically useful for optimizing deployment cost.

- W3: **The structured baseline may be unnecessarily weak.** The structured agents receive a pre-formatted summary and produce predictions in a single LLM call. But modern prompting techniques -- chain-of-thought, self-consistency, reflection -- could substantially improve single-call performance. In Health-LLM, we found that prompting strategy had large effects. The paper should test whether a structured agent with self-consistency (multiple samples, majority vote) or with an explicit self-reflection step can close part of the gap. Without this, we cannot distinguish "tool use helps" from "multi-turn reasoning helps."

- W4: **Calibration analysis is concerning and underexplored.** The 3x over-prediction of negative affect (predicted ~9 vs. ground truth ~3.1) is a serious issue that suggests the LLM has poor grounding in the actual scales. In Health-LLM, we found similar scale miscalibration issues, particularly for less common clinical scales. The paper argues that binary classification is unaffected due to individualized thresholds, but this is only true if the ordering is preserved -- do higher continuous predictions actually correspond to higher actual values, just on the wrong scale? Rank-order correlation between predicted and actual continuous values should be reported.

- W5: **Missing comparison to Health-LLM's approaches.** PULSE cites Health-LLM but does not implement any of its prompting strategies as baselines. Health-LLM's zero-shot and few-shot approaches on wearable data provide relevant comparison points. While the data modalities differ (Health-LLM used Fitbit/Apple Watch; PULSE uses smartphone sensing), the prompting strategies are transferable.

- W6: **The "forecasting" target consistently below chance.** Table A1 shows that the "forecasting" target (presumably future emotional state prediction?) achieves BA consistently below 0.50 across most conditions, including Auto-Multi (0.451). This means the model is systematically predicting the wrong class. This is not just a null result -- it's a failure that deserves explanation. Is the label distribution for this target unusual? Is the construct poorly defined? At minimum, this should be excluded from the mean BA computation if it's not a valid target.

### Questions for Authors

1. What is the total token count (input + output) for a typical agentic prediction vs. a structured prediction? This determines cost at scale and is essential for deployment planning.

2. Have you analyzed the reasoning traces for systematic patterns? For example, does the agent develop consistent investigation strategies for different constructs (e.g., always checking sleep for NA_State, always checking mobility for INT_availability)?

3. The compare_to_baseline tool computes z-scores against personal history. Over what time window? How sensitive is performance to this window length? Early in the study, there is little history -- does performance improve as the baseline window grows?

4. For find_peer_cases, what similarity threshold determines a "peer"? How many peers are retrieved, and what is the sensitivity of predictions to the number of retrieved peers?

5. The paper mentions ~3,900 predictions per condition. With 50 users, that's ~78 entries per user on average. With 16 binary targets per entry, each user contributes 78 x 16 = ~1,248 binary predictions. How do you handle the non-independence of predictions within a user and within an entry?

### Detailed Comments

This paper extends the direction that Health-LLM opened -- using LLMs to reason about personal health data -- in a meaningful way. The shift from single-pass prompting to agentic investigation with domain-specific tools is a natural and productive evolution. The factorial design is rigorous, and the findings about construct-specific data modality requirements are valuable.

My main concern is that the paper evaluates a paradigm (agentic investigation) but only instantiates it with one model. The whole point of a paradigm is that it should generalize. A single-model evaluation tells us that "Claude Sonnet with tools outperforms Claude Sonnet without tools," which is interesting but less impactful than showing that "agentic investigation outperforms structured analysis across models." Even one additional model -- ideally an open-source one -- would substantially strengthen the contribution.

The tool design is the paper's most concrete and reusable contribution. The eight tools are well-motivated and could be adopted by other researchers working with mobile sensing data. I encourage the authors to release the tool implementations as an open-source package.

The calibration issue (W4) connects to a broader challenge in LLM-based health prediction: LLMs have poor priors on clinical scales. When we developed Health-LLM, we observed that even GPT-4 could not reliably predict exact PHQ-9 scores, but could distinguish "high" from "low." PULSE's observation that binary classification is preserved despite continuous miscalibration is consistent with our findings and suggests that the field should focus on relative/ordinal prediction rather than absolute value prediction for LLM-based health systems.

### Minor Issues

- Table 1: Health-LLM is listed as "No" for agentic and "No" for domain tools, which is accurate, but the "10 Health Tasks" characterization is incomplete -- Health-LLM also included physical activity and sleep prediction tasks, not just health/disease tasks.
- The paper should report the LLM's context window usage -- are the 6-12 tool calls approaching the context limit? Context window overflow would silently degrade performance.
- Auto-Sense+ and Auto-Multi+ add a pre-computed behavioral narrative. The null result (no improvement) is interesting and suggests that the agent's own investigation is sufficient. But could the narrative actually hurt by anchoring the agent's investigation?
- The paper mentions "per-user session memory" but does not report its size. How many tokens of memory accumulate by user 50's 80th entry? Is this approaching context limits?

### What would change your score?

To Accept: (1) Add evaluation on at least one additional model (GPT-4 or an open-source model like Llama 3). (2) Report token costs per prediction and analyze the accuracy-cost tradeoff (accuracy vs. number of tool calls). (3) Report rank-order correlations for continuous predictions to validate the "binary classification is fine" argument. (4) Explain or exclude the "forecasting" target. These are substantial but achievable revisions. To Major Revision: If the agentic advantage disappears or is substantially reduced on a different model, it would undermine the paradigm claim and require repositioning the paper as a Claude Sonnet-specific finding.

---

*End of reviews.*
