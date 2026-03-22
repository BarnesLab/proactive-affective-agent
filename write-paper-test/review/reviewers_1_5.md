# IMWUT Peer Reviews for PULSE

Paper: "PULSE: Agentic LLM Investigation of Passive Sensing for Proactive Affect Prediction and Intervention Opportunity Detection in Cancer Survivors"

---

# Review by Tanzeem Choudhury (Cornell Tech)
## IMWUT Review

### Summary (2-3 sentences)

This paper introduces PULSE, a system that uses LLM agents equipped with eight domain-specific sensing query tools (via MCP) to autonomously investigate passive smartphone sensing data for affect prediction and intervention opportunity detection in cancer survivors. Through a 2x2 factorial design (agentic vs. structured reasoning x sensing-only vs. multimodal), the authors demonstrate that the agentic investigation architecture is the primary driver of prediction performance, achieving 0.660 mean balanced accuracy across 16 targets vs. ~0.52 for traditional ML. A notable finding is the dissociation between intervention availability (best predicted by passive sensing) and emotion regulation desire (requiring diary text), with implications for JITAI design.

### Overall Score: Minor Revision
Confidence: High

### Strengths (ranked)
- S1: **The 2x2 factorial design is exceptionally well-conceived.** This is the strongest methodological contribution. By crossing reasoning architecture with data modality, the paper cleanly isolates the contribution of agentic investigation from data richness. This is far more informative than the typical "here is our system, it beats baselines" approach. The effect sizes (r > 0.90) are compelling.
- S2: **The "diary paradox" framing is clinically important and well-articulated.** As someone who has worked extensively with EMA in clinical populations, I know this problem intimately. The systematic missingness of self-report during periods of greatest need is a real barrier to JITAI deployment. The paper addresses it head-on with the sensing-only agent pathway.
- S3: **The dissociation between INT_availability and ER_desire is a genuinely novel conceptual contribution.** The finding that intervention availability is a behavioral construct best captured by passive sensing (0.706 BA) while emotion regulation desire requires diary text has clear, actionable implications for JITAI architecture. This should be highlighted even more prominently.
- S4: **The eight MCP tools are thoughtfully designed to mirror clinical behavioral chart review.** Tools like compare_to_baseline (z-score personalization) and find_similar_days (analogical reasoning) operationalize concepts that the sensing community has struggled to implement in fixed pipelines. The tool design is a reusable contribution.
- S5: **Strong related work section that positions the contribution clearly** within the passive sensing, LLM-for-health, agentic AI, and JITAI literatures. The positioning table (Table 1) is particularly effective.

### Weaknesses (ranked)
- W1: **N=50 with a high-compliance selection bias is a significant limitation for a paper making strong claims about breaking the sensing ceiling.** The authors acknowledge this, but the framing throughout the paper implies general applicability. Users with 82.2 mean EMA entries (vs. 34.0 for others) are likely systematically different in ways beyond the representativeness analysis captures---they may have more regular routines, higher engagement with the study, and more structured behavioral patterns that are inherently easier to predict from. The "ceiling-breaking" narrative should be substantially tempered, or the authors need to show results on a broader sample.
- W2: **The ML baselines are not competitive.** Random Forest, XGBoost, and Logistic Regression with default hyperparameters on daily-aggregated features do not represent the state of the art in sensing-based prediction. Where are the deep learning baselines (CNNs, LSTMs, Transformers trained on raw sensor sequences)? Where is hyperparameter tuning? Where are the personalized models? The GLOBEM ceiling is real, but the specific ML baselines presented here are weak strawmen. A fairer comparison would include at least one neural approach and one personalized model.
- W3: **No ablation of the session memory component.** The authors acknowledge this in limitations, but it is a significant gap. The session memory accumulates "brief reflections from prior predictions," which could be doing a lot of heavy lifting---essentially providing the agent with a running behavioral summary that the structured agent lacks. Without ablating memory, we cannot distinguish whether the agentic advantage comes from dynamic tool use or from the accumulated longitudinal context in memory.
- W4: **The 30-90 second inference time per prediction is not feasible for real-time JITAI deployment.** The paper positions itself as enabling proactive mental health support, but a system that takes over a minute to make a single prediction (and requires cloud API access) cannot deliver just-in-time interventions. The discussion of "batch processing every few hours" essentially concedes that PULSE cannot be a JITAI tailoring variable estimator in the sense Nahum-Shani defined---it is too slow for real-time decision-making.
- W5: **The continuous calibration problem is swept under the rug.** The agentic agents over-predict negative affect by ~3x (predicted mean ~8.5-9.5 vs. ground truth ~3.1). This is not a minor calibration issue---it is a massive systematic bias. The paper argues this is acceptable because binary thresholds absorb the offset, but a deployed JITAI system needs well-calibrated continuous estimates for nuanced intervention dosing, not just binary flags.

### Questions for Authors
1. Can you provide results stratified by user compliance level within the 50 users? Do the agentic advantages hold for users with fewer entries, or is the advantage concentrated in the most regular users?
2. What is the cost per prediction in USD? How does the total inference cost (~27,300 calls) compare to training the ML baselines? This is relevant for scalability arguments.
3. Have you examined the failure cases where the agentic agent performs worse than the structured agent? Understanding when dynamic investigation hurts (perhaps by over-thinking straightforward cases) would be very informative.
4. The session memory never includes ground truth EMA outcomes---but does it include the agent's own prior predictions? If so, error propagation through memory could be a concern.
5. Why was Claude Sonnet chosen over GPT-4 or open-source alternatives? Was any preliminary comparison done?

### Detailed Comments

The paper is well-written and makes a compelling case for the agentic investigation paradigm. The framing around the diary paradox and the clinical motivation from cancer survivorship are effective. However, I have concerns about the strength of the claims relative to the evidence.

The positioning against GLOBEM's "0.52 ceiling" is somewhat misleading. GLOBEM evaluated generalization across datasets and populations, while PULSE evaluates within a single dataset on a curated subset of high-compliance users. The appropriate comparison is within-dataset, within-population performance of well-tuned ML models, not the cross-dataset transfer ceiling. I suspect well-tuned personalized models (e.g., multi-task learning with user embeddings, or fine-tuned per-user models) would substantially exceed 0.52 BA on these 50 high-compliance users.

The example in Section 5.6 (Structured vs. Agentic on the Same Entry) is illustrative but cherry-picked. A systematic analysis of investigation traces---what tools are called when, what patterns emerge, where the agent goes wrong---would be much more convincing than a single favorable example.

The paper claims the agentic paradigm is "model-agnostic" but presents results from exactly one model. This claim should be removed or downgraded to a hypothesis until cross-model evaluation is conducted.

The Auto-Sense+ and Auto-Multi+ variants (with pre-computed behavioral narratives) show negligible improvement over their base counterparts. This is presented as evidence of "robustness," but it could also indicate that the MCP tools are already computing similar summaries, making the pre-computed narrative redundant. This deserves more analysis.

### Minor Issues
- Table 3 (aggregate): The 95% CI for Struct-Sense [.505, .512] does not contain the point estimate of .516. This appears to be an error---please verify.
- The "forecasting" target (Table A1) has BA < 0.50 for most conditions, indicating worse-than-chance performance. This should be discussed or the target excluded.
- The abstract claims "~3,900 diary entries per condition" but not all conditions use diary entries. Clarify this is prediction instances, not diary entries.
- Some references appear to be from 2026 (Feng et al., PHIA), which would be concurrent or future work. Flag these appropriately.

### What would change your score?
- To Accept: (1) Include results on a larger, more representative sample (even 100-150 users without the compliance filter would help). (2) Add at least one competitive deep learning baseline and one personalized ML baseline. (3) Ablate the session memory component.
- To Major Revision: Evidence that the BA improvements are driven primarily by the high-compliance user selection rather than the agentic architecture (e.g., if performance on low-compliance users is much worse).

---

# Review by Xuhai "Orson" Xu (UW SEA Lab)
## IMWUT Review

### Summary (2-3 sentences)

PULSE introduces an agentic LLM framework for passive sensing-based affect prediction in cancer survivors, equipping LLM agents with eight MCP sensing tools for autonomous behavioral data investigation. The core contribution is demonstrating through a 2x2 factorial design that the agentic reasoning architecture (not data modality) is the primary driver of performance, achieving 0.660 mean BA across 16 targets. The paper also reveals that intervention availability is a behavioral construct best predicted by passive sensing alone (0.706 BA), while emotion regulation desire benefits from diary text.

### Overall Score: Major Revision
Confidence: High

### Strengths (ranked)
- S1: **The factorial design is a genuine methodological contribution to the LLM-for-sensing literature.** Most prior work (including our own GLOBEM and Health-LLM) compares systems holistically without isolating what drives performance. The 2x2 design here cleanly separates architecture from modality, producing interpretable results with clear causal implications.
- S2: **The MCP tool design is well-motivated and technically sound.** The eight tools span a meaningful range of clinical investigation strategies---from orientation (daily summary) to targeted investigation (query_sensing) to calibration (find_peer_cases). The temporal boundary enforcement is critical for valid evaluation.
- S3: **The INT_availability finding is novel and important.** Demonstrating that a behavioral construct can be predicted at 0.706 BA from passive sensing alone, with diary text adding essentially nothing, is a significant insight for JITAI design. This deserves to be a headline finding.
- S4: **The paper is exceptionally well-written** with clear problem formulation, rigorous evaluation structure, and honest discussion of limitations.

### Weaknesses (ranked)
- W1: **The comparison to GLOBEM's ceiling is misleading and borders on unfair.** GLOBEM established a cross-dataset generalization ceiling with rigorous leave-one-dataset-out evaluation across four populations. PULSE evaluates on 50 cherry-picked high-compliance users from a single dataset. These are fundamentally different evaluation settings. Claiming to "break through" the GLOBEM ceiling is comparing apples to oranges. To make this comparison valid, the authors would need to evaluate PULSE on GLOBEM's datasets with the same cross-dataset protocol. I am particularly sensitive to this since our GLOBEM work was specifically designed to expose the overly optimistic results that come from within-dataset evaluation on curated samples.
- W2: **The ML baselines are embarrassingly weak.** Default-hyperparameter Random Forest, XGBoost, and Logistic Regression on daily aggregates? In 2026? Where are:
  - (a) Deep learning approaches (LSTM, Transformer, temporal CNNs)?
  - (b) Personalized models (per-user fine-tuning, user embedding approaches)?
  - (c) Multi-task learning that jointly predicts correlated targets?
  - (d) Pre-trained models fine-tuned on sensing data?
  - (e) At minimum, Feng et al. (2026), which the paper itself cites as providing a "comprehensive comparison"---why not include their best-performing model as a baseline?
  The authors try to hedge by calling these "reference points, not direct comparisons," but then proceed to frame the entire narrative around breaking through the ML ceiling. You cannot have it both ways.
- W3: **No cross-validation or held-out evaluation for PULSE itself.** The 50 users come from test folds, but there is no description of how the peer database was constructed relative to these users. Was there any leakage? More importantly, the system is evaluated on a single split without confidence intervals from repeated splitting. The bootstrap CIs reported are over users within a single split, not over different data partitions.
- W4: **The cost and scalability analysis is absent.** Each prediction takes 30-90 seconds and 6-12 LLM API calls. For 50 users x ~78 predictions each x 7 conditions = 27,300 calls. At even conservative API pricing ($0.003/call for Sonnet input+output), that is ~$80+ for this evaluation. For deployment on 418 users with 3 daily predictions, that is ~$375/day or ~$2,600/week. The paper does not discuss this at all. For a system intended for clinical deployment, cost scalability is not optional.
- W5: **The "forecasting" target is broken.** Table A1 shows sub-chance performance (0.447-0.497) across almost all conditions for the "forecasting" target. Yet this target is included in the 16-target mean BA, dragging all conditions down equally. This should be discussed, and arguably the target should be excluded from aggregate metrics or analyzed separately. What does this target measure, and why does every method fail on it?
- W6: **The structured agent baseline may be artificially weakened.** The structured agent receives data in a "pre-formatted summary" and follows a "six-step reasoning pipeline." But the specific design of this summary and pipeline dramatically affects performance. Did the authors iterate on the structured pipeline to make it as strong as possible? Or did they design a minimal version to maximize the contrast with the agentic agent? A stronger structured baseline---e.g., with chain-of-thought prompting, self-consistency, or multiple LLM calls with different prompts---would be more convincing.

### Questions for Authors
1. What is the overlap between the peer database used for RAG and the test fold users? Is there any risk of indirect label leakage through the peer cases (e.g., peer case from same user in a different fold)?
2. For the ML baselines, did you try any hyperparameter tuning (e.g., grid search, Bayesian optimization)? If not, why?
3. Can you report the token counts per prediction for agentic vs. structured agents? This is important for understanding the computational overhead.
4. The paper mentions CALLM uses TF-IDF retrieval. Did you try using the same MCP tools with the structured agent (giving it tool access but no autonomy in tool selection)?
5. Why are some entries from Table A1 showing BA below 0.50? This implies the model is systematically wrong more often than random---what is happening?

### Detailed Comments

I appreciate the ambition and the clean experimental design of this paper. The agentic investigation concept is genuinely interesting and I believe it has legs. However, I have serious concerns about the baseline rigor that prevent me from recommending acceptance in the current form.

The paper's central narrative is: "A decade of ML has plateaued at 0.52 BA; agentic investigation breaks through to 0.66." But this narrative rests on two pillars, both of which are shaky: (1) the ML ceiling is established on cross-dataset transfer, not within-dataset prediction on curated users; (2) the ML baselines presented are not representative of modern supervised learning.

I ran some back-of-envelope calculations: 50 high-compliance users with ~78 entries each gives ~3,900 training+test instances for the ML models (in cross-validation). But the ML models were trained on 399 users---meaning the test set for ML includes users of all compliance levels, while PULSE is only evaluated on high-compliance users. This is not a fair comparison by any standard.

The cross-user RAG mechanism is interesting but raises methodological concerns. When the agent retrieves peer cases with their ground truth outcomes, it is effectively receiving labeled training examples at inference time. This is a form of in-context learning from the training set, which the ML baselines do not get. A fairer comparison would give the ML models access to similar retrieval-based features (k-NN predictions, for instance).

The paper would benefit from a cost-performance Pareto analysis: plotting BA vs. inference cost for all methods, including cheaper LLM alternatives (Haiku, GPT-4-mini) and expensive ML alternatives (ensemble of personalized models).

### Minor Issues
- Table 3: CI for Struct-Sense [.505, .512] does not contain the point estimate .516. Error?
- Table 4: "Entries-level BA from evaluation.json" is oddly specific and reads like an implementation detail rather than a methodological description.
- The paper switches between "balanced accuracy" and "BA" inconsistently in the text.
- Several self-citations (CALLM, BUCS dataset) effectively de-anonymize the submission. This is unavoidable for the dataset but could be mitigated for CALLM by citing it more neutrally.

### What would change your score?
- To Minor Revision: (1) Include competitive ML baselines (at minimum, a tuned deep learning model and a personalized model). (2) Evaluate on a broader user sample (100+ users, not filtered by compliance). (3) Report cost per prediction and discuss scalability honestly.
- To Accept: All of the above, plus (4) cross-dataset evaluation on at least one GLOBEM dataset to validate the generalization claim, or remove the GLOBEM ceiling-breaking framing entirely.

---

# Review by Inbal Nahum-Shani (U Michigan)
## IMWUT Review

### Summary (2-3 sentences)

PULSE proposes an agentic LLM framework that autonomously investigates passive smartphone sensing data to predict emotional states and intervention opportunities in cancer survivors, framed within the JITAI tailoring variable estimation problem. The paper demonstrates through a 2x2 factorial design that agentic reasoning outperforms structured pipelines, and identifies a dissociation between emotion regulation desire (psychological, diary-dependent) and intervention availability (behavioral, sensing-sufficient). The work positions itself as addressing the "diary paradox"---the systematic absence of self-report data when it is most clinically needed.

### Overall Score: Major Revision
Confidence: High

### Strengths (ranked)
- S1: **The decomposition of intervention receptivity into desire vs. availability is this paper's most important contribution, and it maps cleanly onto the JITAI framework.** In my own work on JITAIs, we have emphasized that tailoring variables should capture distinct constructs that inform different aspects of the decision rule. The finding that ER_desire and INT_availability are not only conceptually distinct but empirically dissociable---requiring different data sources for optimal prediction---is exactly the kind of insight the JITAI community needs. This finding alone merits publication.
- S2: **The "diary paradox" is well-articulated and addresses a real methodological challenge in EMA-based intervention research.** The systematic missing-not-at-random nature of EMA non-response during high-distress periods is a known problem. Demonstrating that passive sensing can maintain predictive performance (0.589 BA overall, 0.706 for INT_availability) during diary gaps is practically important.
- S3: **The factorial design is appropriate and well-executed** for isolating the mechanism of interest (agentic vs. structured reasoning). The within-subject design on the same 50 users provides clean comparisons.
- S4: **The JITAI design implications (Section 6.6) are concrete and actionable**, particularly the two-channel architecture recommendation and the principle of separating desire from availability.

### Weaknesses (ranked)
- W1: **The paper conflates prediction accuracy with clinical utility.** A balanced accuracy of 0.706 for INT_availability sounds impressive, but what does it mean for JITAI delivery? What is the false positive rate? What is the false negative rate? In JITAI design, these have very different costs: a false positive (predicting available when not) leads to an ignored or annoying notification; a false negative (predicting unavailable when actually available) is a missed intervention opportunity. The paper reports BA (mean of sensitivity and specificity) but never reports these components separately, despite their differential clinical significance. For a paper grounded in the JITAI framework, this is a significant omission.
- W2: **There is no validation that these predictions would actually improve intervention outcomes.** The paper implicitly assumes that better prediction of tailoring variables leads to better intervention delivery, but this is an empirical question. The JITAI framework is explicit that the value of tailoring variables is determined by their impact on proximal outcomes when used in decision rules, not by their prediction accuracy in isolation. A prediction that is 70% accurate but poorly calibrated might produce worse intervention timing than a simpler rule. Without embedding PULSE in an actual JITAI and measuring intervention outcomes, the clinical claims remain speculative.
- W3: **The operationalization of ER_desire and INT_availability as binary constructs is problematic.** The paper binarizes these using "individualized thresholds" (personal baselines), but the specific binarization procedure is never described. What threshold is used? Median split? Clinical cutoff? Standard deviation-based? This is critical because the BA values are entirely dependent on this binarization. A different threshold could yield very different results. In the JITAI context, the decision of whether to intervene is indeed binary, but the tailoring variable itself is better modeled as continuous (or at least ordinal), with the decision threshold being part of the decision rule design, not the prediction evaluation.
- W4: **The paper does not adequately address the temporal dynamics of tailoring variables.** JITAIs must adapt to within-day temporal dynamics---receptivity varies substantially across the day. Does PULSE's performance vary by time of day (morning vs. afternoon vs. evening EMA)? Does it capture temporal patterns like "available in the morning but not the evening"? The factorial evaluation aggregates across all time points, potentially masking important temporal heterogeneity.
- W5: **The "proactive" framing is aspirational, not demonstrated.** The evaluation is entirely retrospective---replaying past data. A truly proactive system must: (a) run in real-time, (b) generate predictions before decision points, (c) be evaluated on whether its predictions lead to well-timed interventions. None of these are demonstrated. The paper should be framed as a prediction study with proactive potential, not as a proactive system.
- W6: **Sample size and compliance selection bias.** 50 users selected for high compliance from a pool of 418 represents the top ~12% of the study population. These users are not representative of the target clinical population. Cancer survivors with the greatest need for support are likely those with lower compliance (due to distress, treatment burden, etc.)---precisely the users excluded from evaluation. This creates a paradox: the system designed to address the diary paradox is evaluated only on users who don't exhibit it.

### Questions for Authors
1. Can you report sensitivity (true positive rate) and specificity (true negative rate) separately for ER_desire and INT_availability across all conditions? This is essential for clinical interpretation.
2. How exactly are the binary targets constructed? What is the binarization threshold for each target?
3. Does PULSE performance vary by time of day? By day of week? By study week (early vs. late)?
4. Have you considered evaluating the system on users who subsequently dropped out of the study, to test whether it can predict the approach of dropout (a proxy for high distress)?
5. What happens when the peer retrieval returns cases with conflicting outcomes (e.g., 3 peers report high NA, 2 report low NA for similar behavioral patterns)?
6. How would the two-channel architecture (continuous sensing for availability, opportunistic diary for desire) work in practice? What is the expected decision rule?

### Detailed Comments

I want to be clear: I think the core idea of using agentic LLM investigation for passive sensing is promising, and the ER_desire/INT_availability dissociation is a genuine contribution. My concerns are primarily about the gap between the paper's clinical framing and the actual evidence provided.

The paper cites my work on JITAIs extensively and correctly, but I think it also overextends the JITAI framing. JITAI tailoring variable estimation is not just a prediction problem---it is a decision-theoretic problem. The value of a tailoring variable is not its prediction accuracy but its ability to improve the expected outcome when used in a decision rule. The paper never engages with this decision-theoretic perspective. For example: if INT_availability is predicted at 0.706 BA, and the base rate is 0.643, what is the lift in correctly-timed interventions compared to a simpler strategy (e.g., always intervene in the afternoon)?

The emotion regulation desire construct deserves more psychometric scrutiny. The paper references Gross (2015) but does not describe how ER_desire was measured in the BUCS study. Is it a single item? A validated scale? How reliable is it at the within-person level? The prediction accuracy ceiling is bounded by the reliability of the ground truth measure.

The continuous calibration problem (Section 5.7) is more serious than the paper suggests. Over-predicting negative affect by 3x is not "clinically cautious"---it is clinically dangerous if these continuous estimates are ever used for dosing decisions. The paper's argument that binary thresholds absorb the bias only works if you never need the continuous estimate, which limits the system's applicability to binary go/no-go decisions.

I appreciate the thorough limitations section, which addresses many of my concerns honestly. But several limitations (retrospective evaluation, sample selection, model dependency, no memory ablation, calibration issues) are sufficiently serious that they should be resolved before this work is presented as a contribution to JITAI design.

### Minor Issues
- The term "diary paradox" is catchy but may be confusing---it is really an EMA compliance paradox, not specific to diaries.
- "Approximately 3,900 diary entries per condition"---these are predictions, not diary entries.
- The ethical considerations section mentions IRB approval but does not discuss the ethics of using LLM-based inference for clinical decision-making specifically (e.g., accountability for wrong predictions).

### What would change your score?
- To Minor Revision: (1) Report sensitivity/specificity separately for clinical targets. (2) Describe the binarization procedure fully. (3) Analyze temporal dynamics (time-of-day effects). (4) Reframe from "proactive system" to "prediction framework with proactive potential."
- To Accept: All of the above, plus (5) a simulation study showing that using PULSE predictions in a decision rule leads to better intervention timing than simpler strategies (e.g., random timing, time-based rules, threshold-based rules).

---

# Review by Andrew Campbell (Dartmouth)
## IMWUT Review

### Summary (2-3 sentences)

PULSE proposes using LLM agents with tool-use capabilities for affect prediction from passive smartphone sensing in cancer survivors, evaluated via a 2x2 factorial crossing agentic vs. structured reasoning with sensing-only vs. multimodal data. The paper reports that the agentic architecture is the primary driver of prediction performance, achieving 0.660 mean BA across 16 targets compared to ~0.52 for traditional ML, and demonstrates that intervention availability is best predicted by passive sensing alone. The work is positioned as breaking through the decade-long sensing prediction ceiling established by studies like StudentLife and GLOBEM.

### Overall Score: Minor Revision
Confidence: High

### Strengths (ranked)
- S1: **The agentic investigation paradigm is a genuinely novel approach to the sensing-to-prediction problem.** Having worked on passive sensing for mental health since StudentLife, I have watched the field iterate on increasingly complex feature engineering with diminishing returns. The idea of giving an LLM agent investigation tools and letting it reason about behavioral data adaptively is a paradigm shift worth exploring. The MCP tool design is sensible and mirrors how we actually think about behavioral data when reviewing it manually.
- S2: **The 2x2 factorial is the right experimental design.** Too many papers in this space conflate system components. The clean separation of architecture from modality, with within-subject comparisons, produces interpretable results. The effect sizes are convincing---r > 0.90 for the aggregate comparison is not easily dismissed.
- S3: **The finding about INT_availability as a behavioral construct resonates strongly with our experience.** In StudentLife and subsequent work, we observed that contextual availability signals (time, location, activity) were among the most reliable sensing features. That an LLM agent can synthesize these into a 0.706 BA prediction from sensing alone, outperforming diary-based methods, is consistent with and extends this observation.
- S4: **The paper is honest about its limitations,** including the retrospective evaluation, sample selection, model dependency, and continuous calibration issues. This is refreshing in a field that often oversells results.
- S5: **The BUCS dataset adds value as a clinical population.** Most sensing studies evaluate on college students (including ours). Cancer survivors have different behavioral patterns, clinical needs, and study engagement profiles, making this a meaningful extension of the literature.

### Weaknesses (ranked)
- W1: **The comparison to StudentLife and GLOBEM's "sensing ceiling" is misleading.** I know these benchmarks intimately. The ~0.52 BA figure from GLOBEM reflects cross-dataset transfer with minimal adaptation. Within-dataset performance is higher. Moreover, the ML baselines in this paper are not tuned. When we built StudentLife, we spent significant effort on feature engineering and model selection for each target. The ML baselines here use default hyperparameters, which is not how anyone actually deploys these models. A well-tuned ensemble with personalized features would likely exceed 0.52 on these 50 high-compliance users.
- W2: **The sensing data description is insufficient for reproducibility.** The paper mentions eight modalities but does not describe: (a) the sampling rate for each sensor, (b) the preprocessing pipeline (noise removal, resampling, gap-filling), (c) how the raw sensor data is transformed into the representations the MCP tools provide, (d) what happens during sensing gaps (phone off, battery dead, sensor failures). For a paper published in IMWUT, the sensing infrastructure details matter. I would want to know, for example, what percentage of the expected sensing data was actually collected for these 50 users.
- W3: **The 50-user evaluation is too small for the claims being made.** You have 418 users in BUCS. Running PULSE on 50 (selected for compliance) and claiming to break through a decade-long ceiling is a stretch. I understand LLM inference is expensive, but the claims should be proportional to the evidence. At minimum, run the evaluation on 100+ users, or report results on a stratified sample that includes moderate-compliance users.
- W4: **No analysis of sensing data quality or coverage.** The paper mentions iOS vs. Android platform differences but does not report: (a) what fraction of sensing modalities were actually available for each user, (b) how data completeness affects prediction accuracy, (c) whether the agentic advantage is larger or smaller when sensing data is sparse. Given the 36% Android representation (app usage and ambient light only on Android), platform-specific analyses are needed.
- W5: **The qualitative trace analysis (Section 5.6) is a single cherry-picked example.** Where is the systematic analysis? How many tools does the agent call on average? What is the distribution of tool usage across the 3,900 predictions? What percentage of predictions use all 8 tools vs. a subset? The paper mentions "6-12 tools per prediction" in one sentence but provides no further quantification. For a paper claiming the investigation strategy is the key contribution, this is remarkably thin.
- W6: **No comparison to other agentic approaches.** PHIA (cited in the paper) uses code generation for wearable data analysis. GLOSS uses multiple LLMs for sensemaking. How does PULSE's tool-use approach compare to code generation (where the agent writes Python to analyze the data) or a multi-agent debate approach? The agentic space is not homogeneous, and PULSE should be compared to other agentic paradigms, not just the structured pipeline.

### Questions for Authors
1. What is the actual sensing data coverage for these 50 users? What fraction of expected data points (per modality) were collected?
2. Can you provide a systematic tool usage analysis---distribution of tool calls per prediction, conditional tool usage patterns (e.g., does the agent call compare_to_baseline more often when the daily summary reveals anomalies)?
3. How does the agentic advantage vary by platform (iOS vs. Android)? Given that Android users have two additional sensing modalities, does the agent leverage these when available?
4. What happens when you run the agent with fewer tool-use turns (e.g., cap at 4 instead of 16)? Is there a diminishing returns curve?
5. Have you considered using the agent's reasoning traces for post-hoc feature importance analysis? The traces could reveal which sensing signals the agent finds most informative for each target.

### Detailed Comments

This paper tackles an important problem with a novel approach, and I am generally supportive of the direction. The agentic investigation paradigm is compelling, and the factorial design is the right way to evaluate it. My main concerns are about baseline rigor and evaluation scope.

Having built and deployed several sensing studies, I know that data quality and coverage are at least as important as algorithmic sophistication. The paper says almost nothing about the sensing data pipeline---how raw accelerometer data becomes "walking minutes," how GPS traces become "location variance," how sleep is detected from accelerometer data. These preprocessing decisions embed assumptions that directly affect what the MCP tools report to the agent. If the sleep detection algorithm has 80% accuracy, then the agent is reasoning about noisy data, and its investigation advantage may partly come from being more robust to noise than fixed features (which would be a different kind of contribution).

The auto-ethnographic nature of the agentic trace in Section 5.6 is interesting but insufficient. I would like to see: (1) a corpus analysis of all reasoning traces identifying common investigation strategies, (2) a classification of prediction errors by investigation quality (did wrong predictions correlate with shallow investigation?), and (3) examples where the agentic agent's investigation led it astray (confirmation bias, overthinking, etc.).

The session memory mechanism is described briefly but seems potentially very powerful. An agent that accumulates personalized knowledge about each user over ~78 predictions could develop quite rich internal models. This is a form of few-shot learning that the structured agent does not have. The lack of an ablation here is a significant gap.

### Minor Issues
- StudentLife should be cited as [Wang et al., 2014] not just [wang2014studentlife] in the related work.
- The paper mentions "Claude Sonnet" without specifying the exact version (e.g., claude-3.5-sonnet-v2, claude-sonnet-4). Model versions matter for reproducibility.
- The "+" variants (Auto-Sense+, Auto-Multi+) are described as adding a "pre-computed behavioral narrative" but this is never shown or described in detail. What does it look like?
- Table A1: "interact. qual." is abbreviated inconsistently with other targets.

### What would change your score?
- To Accept: (1) Evaluate on at least 100 users including moderate-compliance participants. (2) Provide systematic tool usage and investigation trace analysis. (3) Include at least one tuned ML baseline and one alternative agentic approach (e.g., code generation). (4) Describe sensing data pipeline and coverage.
- Score would stay the same even without these if the claims are appropriately scoped to the evidence (i.e., reframe as a proof-of-concept on high-compliance users rather than a ceiling-breaking result).

---

# Review by Shrikanth Narayanan (USC)
## IMWUT Review

### Summary (2-3 sentences)

PULSE introduces agentic LLM investigation of passive smartphone sensing data for affect prediction and JITAI opportunity detection in cancer survivors, operationalizing the concept of contextual behavioral signal interpretation through eight domain-specific MCP tools. A 2x2 factorial evaluation demonstrates that the agentic architecture (not data modality) is the primary driver of prediction performance, with the key finding that intervention availability is a behavioral construct best captured by passive sensing. The work sits at the intersection of behavioral signal processing, agentic AI, and mobile health, addressing the "diary paradox" of missing self-report data during high-need periods.

### Overall Score: Minor Revision
Confidence: High

### Strengths (ranked)
- S1: **The operationalization of Behavioral Signal Processing through agentic LLM investigation is conceptually significant.** The BSP framework (which I co-developed) has long argued that the bottleneck in behavioral data analysis is signal interpretation, not signal capture. PULSE provides the first concrete instantiation of this idea using LLM agents as the interpretation engine---dynamically selecting what to analyze, contextualizing signals against personal baselines, and synthesizing multimodal evidence. This is a meaningful advance over fixed feature engineering pipelines.
- S2: **The factorial design is rigorous and produces clean, interpretable results.** The within-subject 2x2 design with effect sizes (r > 0.90) convincingly demonstrates that it is the investigation architecture, not the data, that drives the performance improvement. This is an important distinction that most papers in this space fail to make.
- S3: **The dissociation between psychological (ER_desire) and behavioral (INT_availability) constructs is a finding with broad implications.** This maps onto a distinction we have explored in behavioral signal processing---that behavioral and affective signals have fundamentally different information structures. The finding that passive sensing captures behavioral constructs while text captures psychological constructs is consistent with theory and provides empirical validation.
- S4: **The tool design captures key principles of clinical behavioral assessment.** The compare_to_baseline (within-person normalization), find_similar_days (analogical reasoning), and find_peer_cases (population calibration) tools operationalize principles from both clinical psychology and signal processing. The z-score personalization through compare_to_baseline is particularly well-motivated.
- S5: **The graceful degradation from multimodal to sensing-only is practically important** for real-world deployment where data availability is unpredictable.

### Weaknesses (ranked)
- W1: **The multimodal integration is surprisingly shallow.** For a paper that positions itself within behavioral signal processing and multimodal affective computing, the "multimodal" condition simply means "sensing + text." There is no speech prosody, no physiological signals, no facial affect, no acoustic environment analysis. Even within the smartphone modalities available, the paper does not analyze how the agent integrates across modalities---does it reason about the joint distribution of sleep disruption + reduced mobility + increased screen time, or does it process each modality independently and then aggregate? The paper claims "integrative reasoning" but does not demonstrate it formally.
- W2: **The affective constructs are narrowly operationalized.** PANAS captures a specific dimensional model of affect, and the binary state indicators are coarse. The field of affective computing has moved toward continuous, fine-grained affect recognition (valence-arousal space, appraisal dimensions, emotion dynamics). The binary binarization of continuous affect into "unusually high" vs. "not" loses the richness that makes affect prediction clinically useful. Does the agent capture affect dynamics (rate of change, variability, inertia) or only static snapshots?
- W3: **No analysis of when and why the agentic agent fails.** The paper presents average performance metrics but does not analyze failure modes. Behavioral signal processing emphasizes that understanding errors is as important as achieving accuracy. When does the agent's investigation lead to worse predictions than the structured baseline? Does the agent exhibit systematic biases (e.g., anchoring on salient but non-diagnostic signals, confirmation bias in its investigation)? The single qualitative example in Section 5.6 is insufficient.
- W4: **The continuous calibration problem reveals a fundamental limitation of the LLM-as-predictor approach.** Over-predicting negative affect by ~3x is not a calibration issue---it is an alignment issue. The LLM's prior over emotional states (trained on text data where negative events are disproportionately discussed) conflicts with the actual distribution of affect in this population (which is presumably skewed toward neutral/positive most of the time). This is a known problem in sentiment analysis and affective computing. The paper should engage with the literature on LLM calibration for affective tasks more deeply.
- W5: **The paper does not leverage any acoustic or prosodic features**, despite smartphone microphones being available and speech being one of the richest channels for affect recognition. Music listening metadata is included, but actual audio features are not. This is a missed opportunity, especially given the diary entries that could potentially include voice recordings.
- W6: **No formal analysis of the agent's reasoning quality.** The paper claims the agent performs "integrative reasoning" and "contextual behavioral signal interpretation," but this is demonstrated through a single example. A systematic analysis of reasoning traces---coding them for reasoning quality, identifying common strategies, measuring the correlation between reasoning depth and prediction accuracy---would substantially strengthen the paper.

### Questions for Authors
1. When the agent calls compare_to_baseline, how is the personal baseline computed? What lookback window? How are missing data periods handled in the baseline computation?
2. Does the agent's investigation strategy adapt over the course of the study (i.e., does it learn which tools are most informative for a given user through session memory)?
3. Have you analyzed the agent's reasoning traces for evidence of multimodal integration vs. modality-by-modality processing? This is important for the BSP framing.
4. What is the temporal resolution of predictions? Can PULSE predict affect at sub-EMA granularity (e.g., hourly), or is it tied to the EMA schedule?
5. How does the agent handle conflicting signals (e.g., good sleep but high screen time and low mobility)? Does it articulate these conflicts in its reasoning?
6. Have you considered using the agent's confidence estimates as a meta-feature for identifying uncertain predictions that should trigger a brief EMA check-in?

### Detailed Comments

This paper represents an interesting convergence of ideas from behavioral signal processing, agentic AI, and mobile health. The core contribution---showing that the investigation architecture matters more than the data---resonates with the BSP framework's emphasis on signal interpretation over signal capture. However, I have concerns about the depth of the affective computing and multimodal analysis.

The connection to BSP could be much deeper. The BSP framework emphasizes: (1) dynamic, context-sensitive signal processing, (2) multimodal integration at the feature, decision, and model levels, (3) temporal modeling of behavioral dynamics, and (4) computational models of human behavior that link observable signals to latent states. PULSE addresses (1) through the agentic architecture, partially addresses (2) through multimodal data access, but largely ignores (3) and (4). There is no formal temporal modeling---the agent processes each EMA point largely independently (with only the session memory providing longitudinal context). There is no computational model of affect dynamics---the agent relies on the LLM's implicit model of emotions rather than explicit psychological models.

The paper could benefit from a more formal analysis of multimodal information fusion. When the agent has access to both sensing and diary text, how does it weight evidence from each source? Does the weighting vary by target (as the aggregate results suggest it should)? Is the agent's implicit fusion strategy consistent with known principles of multimodal integration (e.g., inverse effectiveness, temporal congruence)?

The music listening feature is intriguing but under-explored. Does the agent reason about music genre, tempo, or listening patterns as affective signals? Music is one of the most direct behavioral indicators of emotional state, and a deeper analysis of how the agent uses music data would be interesting.

The paper would benefit from a comparison with formal computational models of affect (e.g., appraisal models, dynamical systems models of emotion) to contextualize what the LLM agent is doing implicitly. Is the agent performing something like appraisal-based emotion recognition, or is it doing pattern matching against its training data?

### Minor Issues
- The reference to BSP (Narayanan and Georgiou, 2013) in the introduction could include more recent work on multimodal behavioral analysis.
- The paper mentions "ambient light (lux levels; Android only)" as a sensing modality but does not discuss how this is used by the agent or whether its absence on iOS affects performance.
- The "penetrative AI" concept (Xu et al., 2024) deserves more than a one-sentence mention, given the conceptual overlap.
- Table 1 positioning table could include a "Multimodal Fusion" column to better differentiate approaches.

### What would change your score?
- To Accept: (1) A systematic analysis of agentic reasoning traces demonstrating multimodal integration quality. (2) An analysis of failure modes and investigation biases. (3) Temporal dynamics analysis (how the agent handles affect changes over the study period). (4) A brief comparison with at least one formal affective computing model (e.g., a trained emotion recognition model on the sensing features).
- To Reject: Evidence that the agentic advantage is primarily driven by the session memory (longitudinal context) rather than the dynamic investigation paradigm, which would undermine the core contribution.

---

*End of Reviews*
