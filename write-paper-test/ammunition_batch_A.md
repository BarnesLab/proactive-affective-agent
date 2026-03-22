# Ammunition Round — Batch A (5 Experts)

**Paper**: PULSE — Agentic LLM Investigation of Passive Sensing for Proactive Affect Prediction and Intervention Opportunity Detection in Cancer Survivors
**Target**: IMWUT, ~25 pages

---

# Expert 1: Tanzeem Choudhury (Cornell Tech)

*Domain: Mobile sensing for mental health, closing the sensing-to-intervention loop*

---

## Section 1 — Introduction

### Key Arguments
- The sensing-to-intervention loop remains broken: a decade of mobile sensing has produced detection models, but translation to actionable clinical tools lags far behind. PULSE bridges detection to intervention by predicting *when a person wants and is available for support* — not just whether they are distressed.
- The diary paradox is a clinically observed phenomenon: patients most in need disengage from self-report. This is not a data-collection inconvenience — it is a fundamental barrier to deploying JITAIs in real clinical populations.
- Traditional ML on passive sensing features achieves ~0.52 BA (RF 0.518, XGBoost 0.514) — barely above chance. This ceiling has persisted across studies (StudentLife, GLOBEM, CrossCheck). PULSE shatters this ceiling with agentic investigation (0.661 BA).

### Evidence to Cite
- ML baselines: RF 0.518, XGBoost 0.514, Logistic 0.515 (Table from project-digest)
- Auto-Multi+ 0.661 vs ML best 0.518 — large practical gap
- INT_avail from sensing alone: Auto-Sense 0.706 >> CALLM diary-only 0.542

### Phrasing Suggestions
- "...closing the loop from behavioral sensing to proactive intervention delivery..."
- "...the very moments when intervention is most needed are precisely when self-report data is absent..."
- "...agentic investigation enables contextual interpretation that fixed feature pipelines fundamentally cannot provide..."

### Pitfalls to Avoid
- Do not frame PULSE as a clinical intervention system — it is a *prediction layer* that informs JITAIs.
- Do not claim real-time readiness — evaluation is retrospective.

### Reviewer Preemption
- **"How is this different from prior mobile sensing work?"** — Emphasize the paradigm shift: prior work extracts fixed features then classifies; PULSE gives the LLM autonomous investigative tools. The 2x2 factorial cleanly isolates this effect.
- **"N=50 is small"** — Present representativeness analysis in main text (PA_State, ER_desire, INT_avail: p > 0.05 vs full 418).

### Related Work & BibTeX

```bibtex
@inproceedings{Wang2014StudentLife,
  author = {Wang, Rui and Chen, Fanglin and Chen, Zhenyu and Li, Tianxing and Harari, Gabriella and Tignor, Stefanie and Zhou, Xia and Ben-Zeev, Dror and Campbell, Andrew T.},
  title = {{StudentLife}: Assessing Mental Health, Academic Performance and Behavioral Trends of College Students using Smartphones},
  booktitle = {Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)},
  year = {2014},
  pages = {3--14},
  publisher = {ACM},
  doi = {10.1145/2632048.2632054}
}

@inproceedings{Wang2016CrossCheck,
  author = {Wang, Rui and Aung, Min S. H. and Abdullah, Saeed and Brian, Rachel and Campbell, Andrew T. and Choudhury, Tanzeem and Hauser, Marta and Kane, John and Merrill, Michael and Scherer, Emily A. and Tseng, Vincent W. S. and Ben-Zeev, Dror},
  title = {{CrossCheck}: Toward Passive Sensing and Detection of Mental Health Changes in People with Schizophrenia},
  booktitle = {Proceedings of the 2016 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '16)},
  year = {2016},
  pages = {886--897},
  publisher = {ACM},
  doi = {10.1145/2971648.2971740}
}

@article{Abdullah2018SensingMI,
  author = {Abdullah, Saeed and Choudhury, Tanzeem},
  title = {Sensing Technologies for Monitoring Serious Mental Illnesses},
  journal = {IEEE MultiMedia},
  year = {2018},
  volume = {25},
  pages = {61--75},
  doi = {10.1109/MMUL.2018.011921236}
}

@article{Adler2022Generalization,
  author = {Adler, Daniel A. and Wang, Fei and Mohr, David C. and Choudhury, Tanzeem},
  title = {Machine Learning for Passive Mental Health Symptom Prediction: Generalization Across Different Longitudinal Mobile Sensing Studies},
  journal = {PLoS ONE},
  year = {2022},
  volume = {17},
  number = {4},
  pages = {e0266516},
  doi = {10.1371/journal.pone.0266516}
}

@article{Adler2024BeyondDetection,
  author = {Adler, Daniel A. and Yang, Yuewen and Viranda, Thalia and Xu, Xuhai and Mohr, David C. and Van Meter, Anna R. and Tartaglia, Julia C. and Jacobson, Nicholas C. and Wang, Fei and Estrin, Deborah and Choudhury, Tanzeem},
  title = {Beyond Detection: Towards Actionable Sensing Research in Clinical Mental Healthcare},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2024},
  volume = {8},
  number = {4},
  pages = {160},
  doi = {10.1145/3699755}
}
```

---

## Section 2 — Related Work

### Key Arguments
- Position PULSE at the intersection of three converging threads: (1) passive sensing for mental health, (2) LLMs for affective computing, (3) agentic AI with tool use. No prior work occupies this intersection.
- Choudhury's own trajectory illustrates the field's evolution: StudentLife (2014) -> CrossCheck (2016) -> generalization studies (2022) -> actionability (2024). PULSE is the next logical step: from *detecting* states to *investigating* them with autonomous agents.
- The "detection plateau" — after a decade, ML baselines on sensing data hover at ~0.52 BA. This is not a data problem; it is a *reasoning* problem. Features lack contextual interpretation.

### Evidence to Cite
- GLOBEM benchmark: 18 algorithms tested, modest generalization (Xu et al. 2022)
- CrossCheck: 7.6% mean error in schizophrenia symptom prediction, but on aggregated scores
- Adler et al. 2022: models trained on combined data may generalize, but accuracy remains modest
- Adler et al. 2024: clinicians want actionable sensing, not just detection

### Phrasing Suggestions
- "...a decade of mobile sensing research has established the feasibility of detecting behavioral correlates of mental health states, yet translation to clinical action remains elusive..."
- "...the shift from feature-based detection to agentic investigation represents a qualitative change in how computational systems engage with behavioral data..."

### Pitfalls to Avoid
- Do not dismiss prior sensing work — PULSE builds on it. Acknowledge the foundation explicitly.
- Do not frame LLMs as replacing clinical judgment — frame as augmenting clinical workflows.

### Reviewer Preemption
- **"The related work seems disconnected from cancer survivorship"** — Include 2-3 cancer-specific sensing references to ground the clinical motivation. The diary paradox is especially acute in cancer populations where symptom burden fluctuates with treatment cycles.

### Related Work & BibTeX

```bibtex
@article{Saeb2015,
  author = {Saeb, Sohrab and Zhang, Mi and Karr, Christopher J. and Schueller, Stephen M. and Corden, Marya E. and Kording, Konrad P. and Mohr, David C.},
  title = {Mobile Phone Sensor Correlates of Depressive Symptom Severity in Daily-Life Behavior: An Exploratory Study},
  journal = {Journal of Medical Internet Research},
  year = {2015},
  volume = {17},
  number = {7},
  pages = {e175},
  doi = {10.2196/jmir.4273}
}

@article{Xu2023GLOBEM,
  author = {Xu, Xuhai and Liu, Xin and Zhang, Han and Wang, Weichen and Nepal, Subigya and Kuehn, Kevin S. and Huckins, Jeremy F. and Morris, Margaret E. and Nurius, Paula S. and Riskin, Eve A. and Patel, Shwetak and Althoff, Tim and Campbell, Andrew and Dey, Anind K. and Mankoff, Jennifer},
  title = {{GLOBEM}: Cross-Dataset Generalization of Longitudinal Human Behavior Modeling},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2022},
  volume = {6},
  number = {4},
  pages = {1--34},
  doi = {10.1145/3569485}
}

@article{McClaine2024Engagement,
  author = {McClaine, Kimberly and others},
  title = {Engagement With Daily Symptom Reporting, Passive Smartphone Sensing, and Wearable Device Data Collection During Chemotherapy: Longitudinal Observational Study},
  journal = {JMIR Cancer},
  year = {2024},
  doi = {10.2196/57347}
}
```

---

## Section 3 — System Design

### Key Arguments
- The 8 MCP tools mirror the clinical chart review process: a clinician doesn't look at every data point — they selectively investigate based on emerging hypotheses. The agentic design codifies this investigative reasoning.
- Cross-user RAG for calibration is novel: it serves as an *empirical grounding mechanism*, not a knowledge retrieval system. This distinction is critical for positioning.
- Session memory captures longitudinal patterns without ground truth leakage — a crucial design constraint.

### Evidence to Cite
- 8 sensing modalities: motion, GPS, screen, keyboard, app usage (Android), light (Android), music, sleep
- 8 MCP tools: daily summary, behavioral timeline, targeted hourly query, raw events, baseline comparison, receptivity history, similar days, peer cases
- 2x2 factorial: {Structured, Agentic} x {Sensing, Multimodal} — 7 total versions

### Phrasing Suggestions
- "...purpose-built sensing query tools expose behavioral data streams to the LLM agent in a manner analogous to how a clinical dashboard presents patient data to a care provider..."
- "...the agent's investigation strategy emerges from the interaction between its clinical reasoning and the available tools, rather than being pre-specified by the system designer..."

### Pitfalls to Avoid
- Do not anthropomorphize the agent excessively — it is not "thinking like a clinician" but performing tool-augmented reasoning.
- Do not understate the MCP technical contribution — these are novel domain-specific tools.

### Reviewer Preemption
- **"How do you prevent data leakage through session memory?"** — Memory stores self-reflections referencing receptivity signals only; no ground truth from prediction targets is ever stored.
- **"Why MCP and not custom API?"** — MCP is an open standard enabling reproducibility and interoperability.

---

## Section 4 — Evaluation Methodology

### Key Arguments
- The 50-user evaluation is sufficient when coupled with thorough representativeness analysis. Present this in the main text, not supplementary.
- ML baselines on 399 users vs LLM on 50 is an asymmetry that must be acknowledged transparently — but note the ML baselines are *reference points* for the field, not head-to-head comparisons.
- Per-user BA distributions are essential: they show the agent helps *most* users, not just a few outliers.

### Evidence to Cite
- Representativeness: PA_State, ER_desire, INT_avail p > 0.05; NA_State p = 0.028 (small effect r = 0.19)
- EMA count: significantly higher in pilot (82 vs 34 — by design)

### Phrasing Suggestions
- "...while the ML baselines were evaluated on a larger cohort (N=399), they serve as calibration reference points for the sensing modality rather than as direct comparisons..."

### Pitfalls to Avoid
- Do not bury the evaluation asymmetry — state it clearly and explain why it does not invalidate the agentic vs structured comparison (which is apples-to-apples on the same 50 users).

### Reviewer Preemption
- **"Why only 50 users for LLM?"** — Cost and API constraints for ~3,900 entries x 7 versions = ~27,300 LLM inferences. The 2x2 factorial comparison is internally valid.

---

## Section 5 — Results

### Key Arguments
- **Lead with the 2x2 factorial result**: Agentic >> Structured is the headline finding. Auto-Multi 0.660 >> Struct-Multi 0.603; Auto-Sense 0.589 >> Struct-Sense 0.516.
- INT_avail is the Choudhury "closing the loop" finding: sensing alone (0.706) massively outperforms diary (0.542). This means intervention availability is *behaviorally observable* — you don't need to ask the person.
- The diary paradox finding: when diary is absent, sensing-only still delivers viable prediction (0.589).

### Evidence to Cite
- Agentic vs Structured: p < 10^-10, r > 0.9
- ER_desire: Auto-Multi+ 0.751, best in study
- INT_avail: Auto-Sense 0.706 >> CALLM 0.542
- PA_State: Auto-Multi+ 0.733; NA_State: Auto-Multi+ 0.722

### Phrasing Suggestions
- "...the agentic investigation paradigm is the primary driver of prediction accuracy, independent of data modality..."
- "...intervention availability, a construct fundamental to JITAI deployment, is best captured through behavioral sensing rather than self-report..."

### Pitfalls to Avoid
- Do not overclaim on filtering: Auto-Multi+ 0.661 ≈ Auto-Multi 0.660. Mention but do not emphasize.
- Do not present absolute BA numbers without context — always pair with baselines and effect sizes.

### Reviewer Preemption
- **"0.66 BA is still modest"** — Compare to prior art in sensing (~0.52) and note these are 16-target averages including difficult constructs. Individual targets reach 0.75.

---

## Section 6 — Discussion

### Key Arguments
- Connect to Adler et al. (2024) "Beyond Detection" — PULSE operationalizes the vision of actionable sensing by predicting intervention-relevant constructs.
- The diary paradox has implications for JITAI system design: systems must gracefully degrade from multimodal to sensing-only.
- Separate desire (ER_desire — emotional, diary-aided) from availability (INT_avail — behavioral, sensing-captured) as distinct JITAI tailoring variables.

### Evidence to Cite
- INT_avail behavioral vs diary: Auto-Sense 0.706 vs CALLM 0.542
- Graceful degradation: Auto-Sense 0.589 is viable; Struct-Sense 0.516 is not

### Phrasing Suggestions
- "...PULSE demonstrates that the sensing-to-intervention gap can be narrowed not by better sensors, but by smarter reasoning over existing data streams..."
- "...the field has focused on whether sensing data *contains* mental health signals; PULSE shifts the question to how an intelligent agent can *investigate* those signals in context..."

### Pitfalls to Avoid
- Do not claim generalizability beyond cancer survivors without evidence.
- Do not speculate about real-time deployment without acknowledging 30-90s inference latency.

### Reviewer Preemption
- **"This is just prompt engineering"** — The 2x2 factorial proves otherwise: structured prompts with identical data produce 0.603; agentic investigation produces 0.660. The architecture, not the prompt, is the differentiator.

---

## Sections 7-8 — Future Work & Conclusion

### Key Arguments
- Future: close the loop completely — from prediction to intervention generation. PULSE predicts *when*; next step is *what* to deliver.
- Real-time deployment requires edge LLMs or streaming architectures.
- Conclusion should emphasize paradigm contribution: the community can adopt agentic investigation for behavioral data beyond cancer survivors.

---

# Expert 2: Xuhai "Orson" Xu (UW SEA Lab / Columbia)

*Domain: GLOBEM, Health-LLM, computational well-being*

---

## Section 1 — Introduction

### Key Arguments
- Position PULSE in the lineage of computational well-being systems: GLOBEM benchmarked traditional ML; Health-LLM showed LLMs can process wearable data; Mental-LLM showed LLMs can process text for mental health. PULSE synthesizes all three threads with autonomous tool use.
- The ML ceiling (~0.52 BA) is well-documented in GLOBEM — 18 algorithms, modest results across multiple datasets. PULSE does not just incrementally improve; it changes the computational paradigm.
- Health-LLM showed promise but used fixed prompts and structured inputs. PULSE demonstrates that giving LLMs *tools to investigate* is the key unlock.

### Evidence to Cite
- GLOBEM: 18 algorithms, 497 participants, 4 years — established the ceiling
- Health-LLM: 12 LLMs evaluated on wearable data, structured prompts
- PULSE Auto-Multi+ 0.661 vs GLOBEM-style ML ~0.52

### Phrasing Suggestions
- "...where prior benchmarks revealed the ceiling of feature-based approaches, agentic investigation reveals a new trajectory for computational well-being..."
- "...PULSE extends the Health-LLM paradigm by replacing fixed data formatting with autonomous investigation..."

### Pitfalls to Avoid
- Do not position PULSE as competing with GLOBEM — frame as building on GLOBEM's diagnostic of the field's limitations.

### Reviewer Preemption
- **"Have you compared against Health-LLM?"** — Note different evaluation setup (Health-LLM used public datasets, not cancer-specific). The contribution is architectural (agentic vs structured), not a leaderboard comparison.

### Related Work & BibTeX

```bibtex
@inproceedings{Kim2024HealthLLM,
  author = {Kim, Yubin and Xu, Xuhai and McDuff, Daniel and Breazeal, Cynthia and Park, Hae Won},
  title = {{Health-LLM}: Large Language Models for Health Prediction via Wearable Sensor Data},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  pages = {522--539},
  year = {2024},
  volume = {248},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}

@article{Xu2024MentalLLM,
  author = {Xu, Xuhai and Yao, Bingsheng and Dong, Yuanzhe and Gabriel, Saadia and Yu, Hong and Hendler, James and Ghassemi, Marzyeh and Dey, Anind K. and Wang, Dakuo},
  title = {{Mental-LLM}: Leveraging Large Language Models for Mental Health Prediction via Online Text Data},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2024},
  volume = {8},
  number = {1},
  pages = {31},
  doi = {10.1145/3643540}
}

@article{Xu2022GLOBEMDataset,
  author = {Xu, Xuhai and Zhang, Han and Sefidgar, Yasaman S. and Ren, Yiyi and Liu, Xin and Seo, Woosuk and Brown, Jennifer and Kuehn, Kevin S. and Merrill, Michael and Nurius, Paula S. and Patel, Shwetak N. and Althoff, Tim and Morris, Margaret E. and Riskin, Eve A. and Mankoff, Jennifer and Dey, Anind K.},
  title = {{GLOBEM} Dataset: Multi-Year Datasets for Longitudinal Human Behavior Modeling Generalization},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2022},
  volume = {abs/2211.02733}
}

@article{Xu2025LENS,
  author = {Xu, Wenxuan and others},
  title = {{LENS}: {LLM}-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models},
  journal = {arXiv preprint arXiv:2512.23025},
  year = {2025}
}
```

---

## Section 2 — Related Work

### Key Arguments
- Build a progression narrative: (1) sensing data → fixed features → ML (GLOBEM ceiling), (2) sensing data → text description → LLM prompt (Health-LLM, Mental-LLM), (3) sensing data → autonomous tool use → agentic investigation (PULSE). Each step adds reasoning capability.
- LENS (Xu et al. 2025) aligns multimodal sensing with language models for narrative synthesis — complementary to PULSE but does not give agents investigative autonomy.
- The GLOBEM benchmark specifically revealed that generalization across populations is a key failure mode. PULSE's cross-user RAG addresses this by grounding predictions in empirical population data.

### Evidence to Cite
- GLOBEM: 4 years, 700+ user-years, 9 domain generalization techniques tested
- Health-LLM: 10 health prediction tasks, 12 LLMs — showed structured prompting works but is limited
- Mental-LLM: zero-shot, few-shot, fine-tuning — best results from instruction fine-tuning
- CALLM: 72.96% PA, 73.29% NA, 73.72% ER_desire — diary-only baseline

### Phrasing Suggestions
- "...GLOBEM established that traditional ML approaches to longitudinal behavior modeling face fundamental generalization limits; PULSE proposes an alternative computational paradigm..."
- "...while Health-LLM and Mental-LLM demonstrated that LLMs can reason about health data when properly prompted, PULSE demonstrates that LLM agents can autonomously investigate behavioral data..."

### Pitfalls to Avoid
- Do not conflate GLOBEM (benchmark platform) with the ML models tested within it.
- Be precise about CALLM relationship — PULSE extends CALLM, not replaces it.

### Reviewer Preemption
- **"How does PULSE relate to CALLM?"** — CALLM is the diary-only baseline (v0 in the factorial). PULSE extends it by adding passive sensing and agentic investigation. The same team built both systems.

### Related Work & BibTeX

```bibtex
@article{Wang2025CALLM,
  author = {Wang, Zhiyuan and Daniel, Katharine E. and Barnes, Laura E. and Chow, Philip I.},
  title = {{CALLM}: Understanding Cancer Survivors' Emotions and Intervention Opportunities via Mobile Diaries and Context-Aware Language Models},
  journal = {arXiv preprint arXiv:2503.10707},
  year = {2025}
}

@article{Zhang2024LLMAffect,
  author = {Zhang, Tianyi and Teng, Songyan and Jia, Hong and D'Alfonso, Simon},
  title = {Leveraging {LLMs} to Predict Affective States via Smartphone Sensor Features},
  booktitle = {Companion of the 2024 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp Companion '24)},
  year = {2024},
  publisher = {ACM},
  doi = {10.1145/3675094.3678420}
}
```

---

## Section 3 — System Design

### Key Arguments
- The factorial design is a major methodological contribution from the computational well-being perspective. Prior LLM-for-health papers compare against ML baselines but fail to isolate *what about the LLM matters*. The 2x2 design answers this cleanly.
- Cross-user RAG for calibration connects to GLOBEM's insight about cross-population generalization. Rather than training a generalizable model, PULSE retrieves similar cases at inference time.
- The 8 MCP tools should be presented as a reusable toolkit for the sensing community.

### Evidence to Cite
- 7 versions tested: CALLM, Struct-Sense, Auto-Sense, Struct-Multi, Auto-Multi, Auto-Sense+, Auto-Multi+
- Three RAG modes: text-based, sensing-based, tool-based
- 5-fold across-subject CV with temporal boundary enforcement

### Phrasing Suggestions
- "...the factorial design enables a controlled comparison rarely seen in LLM-for-health research, where most evaluations conflate model architecture with data modality..."
- "...cross-user RAG operationalizes the intuition behind population-level calibration: rather than learning a fixed mapping from features to labels, the system retrieves empirical examples that ground its predictions..."

### Pitfalls to Avoid
- Do not oversell the RAG as "knowledge retrieval" — it is *calibration*, drawing a distinction from typical RAG use cases.

### Reviewer Preemption
- **"Is the 2x2 sufficient to make causal claims?"** — Frame as a quasi-experimental design that isolates two factors. Not causal in the RCT sense, but the strongest design possible for system comparison at this scale.

---

## Section 4 — Evaluation Methodology

### Key Arguments
- Balanced accuracy as primary metric is critical for imbalanced health targets — standard accuracy would be misleading.
- Per-user evaluation (not per-entry averaging) prevents high-compliance users from dominating.
- The 50 vs 418 representativeness analysis follows GLOBEM-style methodological rigor.

### Evidence to Cite
- ~3,900 entries per version across 50 users
- Wilcoxon signed-rank tests, bootstrap CIs, effect sizes

### Pitfalls to Avoid
- Do not compare LLM and ML baselines as if they are apples-to-apples — different N, different feature sets.

---

## Section 5 — Results

### Key Arguments
- The GLOBEM lens: Auto-Multi+ 0.661 is the highest BA ever reported for passive-sensing-based affect prediction in a clinical population. Frame against GLOBEM-era baselines.
- Multimodal >> Sensing-only confirms the value of diary text when available — but the *graceful degradation* finding (sensing-only still works) is equally important.
- Per-user distributions should look like GLOBEM-style analyses — show variance across individuals.

### Evidence to Cite
- Auto-Multi+ 0.661, Auto-Multi 0.660
- CALLM 0.611 (diary-only, the PULSE predecessor)
- Auto-Sense 0.589 >> Struct-Sense 0.516 (agentic effect in sensing-only)
- ER_desire 0.751 — highest individual target

### Phrasing Suggestions
- "...the agentic architecture achieves prediction accuracy that exceeds the ceiling established by a decade of feature-engineering approaches to behavioral sensing..."

### Reviewer Preemption
- **"These numbers are from a single LLM (Claude Sonnet)"** — Acknowledge as limitation. The contribution is the *architecture*, and future work should test across models.

---

## Section 6 — Discussion

### Key Arguments
- Connect to the computational well-being roadmap: GLOBEM diagnosed the problem (ceiling), Health-LLM explored LLMs for health data, PULSE introduces the agentic paradigm. The next step is real-time deployment.
- Cross-user RAG as calibration (not knowledge retrieval) is a contribution to the broader LLM-for-health community.
- The diary paradox finding directly informs JITAI system design: build for sensing-only, enhance with diary when available.

### Pitfalls to Avoid
- Do not claim the agentic paradigm will generalize to all well-being prediction tasks without evidence.

---

# Expert 3: Inbal Nahum-Shani (U Michigan)

*Domain: JITAI framework, micro-randomized trials, receptivity*

---

## Section 1 — Introduction

### Key Arguments
- Frame through the JITAI lens: the core challenge is estimating *tailoring variables* in real time. PULSE directly addresses this by predicting two key tailoring variables — vulnerability/opportunity (ER_desire) and receptivity (INT_avail).
- The JITAI framework (Nahum-Shani et al. 2018) identifies six core components. Tailoring variables are the hardest to operationalize because they require real-time estimation. PULSE provides this.
- Receptivity has been studied through EMA-based self-report (Mishra 2021, Kunzler 2019). PULSE demonstrates that passive sensing can predict intervention availability — a receptivity proxy — without asking the user.

### Evidence to Cite
- INT_avail: Auto-Sense 0.706 — predicted from sensing alone
- ER_desire: Auto-Multi+ 0.751 — best with multimodal data
- CALLM INT_avail 0.542 (diary-only) vs Auto-Sense 0.706 (sensing-only) — sensing dramatically outperforms diary for availability

### Phrasing Suggestions
- "...PULSE operationalizes the JITAI framework's tailoring variables by predicting both vulnerability/opportunity (emotion regulation desire) and receptivity (intervention availability) from passive behavioral data..."
- "...the finding that intervention availability is better predicted by sensing than by self-report has direct implications for JITAI deployment: the system need not interrupt the user to determine whether to interrupt the user..."

### Pitfalls to Avoid
- Do not conflate INT_avail with receptivity in the strict Nahum-Shani sense — INT_avail is a *proxy* for availability, one component of receptivity.
- Do not claim PULSE implements a full JITAI — it is the prediction layer that informs JITAI decision rules.

### Reviewer Preemption
- **"This isn't a JITAI study"** — Correct. PULSE provides the prediction layer that JITAIs need. The 2x2 factorial tests prediction accuracy, not intervention efficacy.
- **"How do you define receptivity?"** — Be precise: we predict INT_avail (self-reported availability for intervention) as a behavioral construct, distinct from but related to Nahum-Shani's receptivity construct.

### Related Work & BibTeX

```bibtex
@article{NahumShani2018JITAI,
  author = {Nahum-Shani, Inbal and Smith, Shawna N. and Spring, Bonnie J. and Collins, Linda M. and Witkiewitz, Katie and Tewari, Ambuj and Murphy, Susan A.},
  title = {Just-in-Time Adaptive Interventions ({JITAIs}) in Mobile Health: Key Components and Design Principles for Ongoing Health Behavior Support},
  journal = {Annals of Behavioral Medicine},
  year = {2018},
  volume = {52},
  number = {6},
  pages = {446--462},
  doi = {10.1007/s12160-016-9830-8}
}

@article{Mishra2021Receptivity,
  author = {Mishra, Varun and K{\"u}nzler, Florian and Kramer, Jan-Niklas and Fleisch, Elgar and Kowatsch, Tobias and Kotz, David},
  title = {Detecting Receptivity for {mHealth} Interventions in the Natural Environment},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2021},
  volume = {5},
  number = {2},
  pages = {74},
  doi = {10.1145/3463492}
}

@article{Kunzler2019Receptivity,
  author = {K{\"u}nzler, Florian and Mishra, Varun and Kramer, Jan-Niklas and Fleisch, Elgar and Kotz, David F. and Kowatsch, Tobias},
  title = {Exploring the State-of-Receptivity for {mHealth} Interventions},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2019},
  volume = {3},
  number = {4},
  pages = {140},
  doi = {10.1145/3369805}
}

@article{Klasnja2015MRT,
  author = {Klasnja, Predrag and Hekler, Eric B. and Shiffman, Saul and Boruvka, Audrey and Almirall, Daniel and Tewari, Ambuj and Murphy, Susan A.},
  title = {Micro-randomized Trials: An Experimental Design for Developing Just-in-Time Adaptive Interventions},
  journal = {Health Psychology},
  year = {2015},
  volume = {34},
  number = {Suppl},
  pages = {1220--1228},
  doi = {10.1037/hea0000305}
}

@incollection{NahumShani2023Vulnerability,
  author = {Nahum-Shani, Inbal and Wetter, David W. and Murphy, Susan A.},
  title = {Adapting Just-in-Time Interventions to Vulnerability and Receptivity: Conceptual and Methodological Considerations},
  booktitle = {Digital Therapeutics for Mental Health and Addiction},
  publisher = {Elsevier},
  year = {2023},
  pages = {77--87}
}
```

---

## Section 2 — Related Work (2.4 JITAIs)

### Key Arguments
- JITAI literature has extensively theorized about tailoring variables but operationalization lags. Most JITAI studies use self-report EMA for tailoring — creating a circular dependency (you must bother the user to decide whether to bother the user).
- Micro-randomized trials (Klasnja et al. 2015) test intervention components but assume tailoring variables are available. PULSE provides the missing prediction layer.
- Nahum-Shani et al. (2023) distinguish vulnerability, opportunity, and receptivity. PULSE's targets map: ER_desire → vulnerability/opportunity; INT_avail → receptivity.

### Evidence to Cite
- Nahum-Shani (2018): 6 JITAI components, tailoring variables as hardest to estimate
- Mishra (2021): 189 participants, 6 weeks, receptivity prediction from context
- Kunzler (2019): state-of-receptivity from phone sensors — PULSE extends this with richer sensing and LLM reasoning

### Phrasing Suggestions
- "...while the JITAI framework identifies tailoring variables as the lynchpin of adaptive intervention, most implementations rely on self-report for these variables — creating an ironic dependency on the very data source that is absent when most needed..."
- "...PULSE addresses the gap between JITAI theory and JITAI practice by demonstrating that passive sensing, when investigated by an agentic LLM, can estimate tailoring variables with clinically meaningful accuracy..."

### Pitfalls to Avoid
- Do not oversimplify the JITAI framework — it is a multi-component system with decision points, proximal outcomes, etc. PULSE addresses the tailoring variable component specifically.

### Reviewer Preemption
- **"Receptivity prediction has been done before"** — Yes, but with fixed ML features (Mishra 2021, Kunzler 2019). PULSE achieves INT_avail 0.706 through agentic reasoning — and crucially demonstrates that diary is *not needed* for this construct.

---

## Section 3 — System Design

### Key Arguments
- The distinction between ER_desire and INT_avail is theoretically grounded in JITAI framework: desire is an *emotional* tailoring variable (vulnerability/opportunity), availability is a *contextual-behavioral* tailoring variable (receptivity).
- The 2x2 factorial maps to a JITAI design question: what data sources and reasoning architectures best estimate tailoring variables?
- The finding that sensing predicts availability better than diary aligns with JITAI theory — behavioral context (what is the person doing?) is more reliably captured by sensors than by self-report.

### Evidence to Cite
- INT_avail sensing-only: 0.706 (Auto-Sense) vs 0.542 (CALLM diary)
- ER_desire needs diary: 0.751 (Auto-Multi+) vs 0.653 (Auto-Sense+)

### Phrasing Suggestions
- "...the differential sensitivity of desire and availability to sensing vs. diary data has direct design implications: JITAI systems should estimate availability from passive sensing and reserve self-report for assessing emotional desire when available..."

---

## Section 5 — Results

### Key Arguments
- The INT_avail finding is the JITAI headline: passive sensing predicts intervention availability better than self-report diary. This means JITAIs can assess receptivity without interrupting the user.
- ER_desire requires multimodal data — this makes sense because desire for emotion regulation is an internal state best captured through self-expression (diary text).
- The agentic vs structured comparison has JITAI design implications: a structured pipeline (like a fixed decision rule) underperforms an adaptive reasoning system.

### Evidence to Cite
- INT_avail: Auto-Sense 0.706, Auto-Multi 0.716 — improvement from diary is minimal
- ER_desire: Auto-Multi+ 0.751, Auto-Sense+ 0.653 — diary matters here
- Structured Multi 0.551 for INT_avail — structured reasoning fails even with diary

### Phrasing Suggestions
- "...intervention availability, the construct most directly relevant to JITAI receptivity, is best estimated from behavioral sensing — not self-report..."
- "...the agentic architecture's advantage over structured reasoning is especially pronounced for intervention availability, suggesting that contextual behavioral interpretation benefits from flexible investigation rather than fixed analysis pipelines..."

### Reviewer Preemption
- **"INT_avail is self-reported — how is it behavioral?"** — It asks about behavioral availability ("are you available for an intervention right now?"), not emotional state. Sensing captures the behavioral context (location, phone usage, activity) that determines availability.

---

## Section 6 — Discussion (6.3, 6.6)

### Key Arguments
- 6.3: INT_avail as a behavioral construct — the most important JITAI finding. Design implication: separate desire from availability in JITAI decision rules. Use sensing for availability, reserve diary for desire.
- 6.6: Four implications for JITAI design:
  1. Separate desire from availability as distinct tailoring variables
  2. Use agentic reasoning for state detection (not fixed rules)
  3. Design graceful degradation (sensing-only fallback)
  4. Leverage population data for calibration (cross-user RAG)
- Connect to future MRTs: PULSE's predictions could serve as tailoring variables in a micro-randomized trial of JITAIs for cancer survivors.

### Phrasing Suggestions
- "...our results suggest that JITAI decision rules should treat intervention desire and intervention availability as orthogonal dimensions with different optimal data sources..."

### Reviewer Preemption
- **"You don't test whether these predictions improve intervention outcomes"** — Correct. That requires an MRT. PULSE provides the prediction infrastructure; testing outcomes is future work.

---

# Expert 4: Andrew Campbell (Dartmouth)

*Domain: StudentLife, Mindscape, mobile sensing deployment*

---

## Section 1 — Introduction

### Key Arguments
- Ground in the StudentLife legacy: a decade ago, StudentLife demonstrated that passive smartphone sensing correlates with mental health. PULSE asks the next question: what if an intelligent agent could *interpret* these correlations autonomously?
- Deployment realism: Campbell's work emphasizes ecological validity and real-world deployment. PULSE uses data from the BUCS study — 418 cancer survivors, 5 weeks, 3x daily EMA, 8 sensing modalities. This is large-scale, real-world data.
- The sensing modality coverage maps to StudentLife's heritage: motion, GPS, screen, keyboard — the core smartphone sensing stack.

### Evidence to Cite
- BUCS: 418 cancer survivors, ~5 weeks, 3x daily EMA
- 8 sensing modalities with platform-specific coverage (iOS vs Android)
- 50 users evaluated, 82 mean EMA count vs 34 in full population

### Phrasing Suggestions
- "...building on a decade of mobile sensing infrastructure pioneered by StudentLife and its successors, PULSE demonstrates that the value of passive sensing data can be dramatically amplified by agentic LLM interpretation..."
- "...the BUCS dataset represents one of the largest longitudinal passive sensing studies in a clinical cancer population..."

### Pitfalls to Avoid
- Do not minimize the engineering effort of data collection — acknowledge the BUCS study team.
- Do not present 8 modalities as if all are equally available — note iOS vs Android coverage.

### Reviewer Preemption
- **"How does sensing coverage vary by platform?"** — iOS lacks app usage and light sensor data. Present a coverage table and note that the agent adapts its investigation to available data.

### Related Work & BibTeX

```bibtex
@article{Nepal2024CollegeExperience,
  author = {Nepal, Subigya and Liu, Wenjun and Pillai, Arvind and Wang, Weichen and Vojdanovski, Vlado and Huckins, Jeremy F. and Rogers, Courtney and Meyer, Meghan L. and Campbell, Andrew T.},
  title = {Capturing the College Experience: A Four-Year Mobile Sensing Study of Mental Health, Resilience and Behavior of College Students during the Pandemic},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2024},
  volume = {8},
  number = {1},
  doi = {10.1145/3643501}
}

@article{Nepal2024MindScape,
  author = {Nepal, Subigya and Pillai, Arvind and Campbell, William and Massachi, Talie and Heinz, Michael V. and Kunwar, Ashmita and Choi, Eunsol Soul and Xu, Xuhai and Kuc, Joanna and Huckins, Jeremy F. and Holden, Jason and Preum, Sarah M. and Depp, Colin and Jacobson, Nicholas and Czerwinski, Mary P. and Granholm, Eric and Campbell, Andrew T.},
  title = {{MindScape} Study: Integrating {LLM} and Behavioral Sensing for Personalized {AI}-Driven Journaling Experiences},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2024},
  volume = {8},
  number = {4},
  pages = {186},
  doi = {10.1145/3699761}
}

@article{Huckins2020COVID,
  author = {Huckins, Jeremy F. and daSilva, Alex W. and Wang, Weichen and Hedlund, Elin and Rogers, Courtney and Nepal, Subigya K. and Wu, Jialing and Obuchi, Mikio and Murphy, Eilis I. and Meyer, Meghan L. and Wagner, Dylan D. and Holtzheimer, Paul E. and Campbell, Andrew T.},
  title = {Mental Health and Behavior of College Students During the Early Phases of the {COVID-19} Pandemic: Longitudinal Smartphone and Ecological Momentary Assessment Study},
  journal = {Journal of Medical Internet Research},
  year = {2020},
  volume = {22},
  number = {6},
  pages = {e20185},
  doi = {10.2196/20185}
}
```

---

## Section 2 — Related Work

### Key Arguments
- The StudentLife lineage: Wang et al. (2014) → Huckins et al. (2020, COVID) → Nepal et al. (2024, 4-year study) → MindScape (Nepal et al. 2024, LLM + sensing). PULSE extends this trajectory from descriptive correlation to predictive investigation.
- MindScape is the most direct comparison from Campbell's group: it integrates LLM and behavioral sensing for journaling. PULSE differs by using autonomous agent tools rather than LLM-powered content generation.
- The "four-year college experience" study (Nepal 2024) shows that long-term sensing is feasible but interpretation at scale remains manual. PULSE automates the interpretation.

### Evidence to Cite
- StudentLife: 48 students, 10 weeks, first continuous sensing study
- MindScape: 20 students, 8 weeks, LLM + sensing for journaling — positive affect +7%, negative affect -11%
- Nepal 2024: 4-year study, longest longitudinal mobile sensing study

### Phrasing Suggestions
- "...MindScape demonstrated that LLMs can generate meaningful behavioral narratives from sensing data; PULSE extends this by giving LLMs the tools to autonomously investigate sensing data for clinical prediction..."

### Pitfalls to Avoid
- Do not claim PULSE replaces MindScape — they serve different purposes (prediction vs journaling).

### Reviewer Preemption
- **"How does this compare to MindScape?"** — MindScape uses LLM for content generation; PULSE uses LLM agents for clinical prediction. Complementary, not competing.

---

## Section 3 — System Design

### Key Arguments
- Data engineering is a first-class concern in sensing research. The 8 MCP tools abstract away the complexity of heterogeneous sensing data streams, making them accessible to the LLM agent.
- Platform-specific coverage (iOS vs Android) is a deployment reality. The agentic design handles this naturally: the agent queries available tools and adapts.
- The BUCS dataset's 3x daily EMA protocol provides rich ground truth. Describe data collection infrastructure.

### Evidence to Cite
- 8 sensing modalities with platform-specific availability
- 5-fold across-subject CV prevents data leakage between users
- ~3,900 entries per version = ~27,300 total LLM inferences

### Phrasing Suggestions
- "...the MCP tool abstraction layer transforms heterogeneous, multi-platform sensing data into a uniform investigation interface..."
- "...the agent's adaptive investigation strategy naturally handles platform-specific data availability — when app usage data is absent on iOS, the agent directs its investigation to available modalities..."

### Pitfalls to Avoid
- Do not gloss over platform differences — they affect ~25% of users.

### Reviewer Preemption
- **"What about missing data?"** — Present data completeness metrics. The agentic approach is inherently robust to missing modalities — it queries what is available.

---

## Section 5 — Results

### Key Arguments
- Compare to StudentLife-era baselines: StudentLife found correlations between sensing and mental health but did not produce prediction models with BA metrics. PULSE produces actionable predictions.
- The per-user distributions should show that the agent helps across diverse users — echoing Campbell's emphasis on studying individual differences in sensing studies.
- Qualitative investigation traces are essential: show the agent's tool-use sequence for a specific user, demonstrating how it *reasons* about behavioral data.

### Evidence to Cite
- Auto-Multi+ 0.661 mean BA across 16 targets
- Per-user BA distributions (violin/box plots)
- Agent investigation traces: which tools called, in what order, what found

### Phrasing Suggestions
- "...the agentic investigation traces reveal a reasoning process remarkably similar to how researchers manually analyze sensing data in studies like StudentLife: examining trends, comparing against baselines, and synthesizing multiple behavioral signals into a coherent interpretation..."

### Reviewer Preemption
- **"Qualitative traces are cherry-picked"** — Present summary statistics of tool usage patterns (which tools are most called, average investigation depth) alongside example traces.

---

## Section 6 — Discussion

### Key Arguments
- 6.1: Why agentic investigation works — selective attention, anomaly detection, cross-modality reasoning. The agent decides what to examine based on what it finds, mimicking the scientific method.
- 6.2: Deployment implications — 30-90s inference is too slow for real-time JITAI. But for batch processing (e.g., every few hours) it is feasible.
- Scalability: discuss how MCP tools could be deployed on a server for real-time access, how edge LLMs could reduce latency.

### Phrasing Suggestions
- "...a decade of sensing deployment experience suggests that the bottleneck in mobile mental health is not data collection but data interpretation; PULSE addresses this bottleneck by delegating interpretation to an agentic LLM..."

---

## Section 7 — Future Work

### Key Arguments
- Real-time deployment is the next step — Campbell's group has deployed sensing apps at scale (StudentLife, MindScape). PULSE's server-side architecture could integrate with existing sensing infrastructure.
- Cross-population evaluation: test on students (StudentLife-like populations), other chronic illness groups.
- Edge LLM deployment for reducing latency and improving privacy.

---

# Expert 5: Shrikanth Narayanan (USC)

*Domain: Behavioral signal processing, multimodal affective computing*

---

## Section 1 — Introduction

### Key Arguments
- Frame through BSP lens: PULSE is behavioral signal processing with LLM agents. Traditional BSP extracts features from speech/physiological signals and maps to behavioral states. PULSE extends this by using LLM agents to perform *contextual signal interpretation*.
- The multimodal dimension is key: PULSE combines sensing data (behavioral signals) with diary text (self-reported affect) and cross-user data (population calibration) — a multimodal fusion that BSP has long advocated.
- The "behavioral informatics" perspective: sensing data captures behavioral correlates of mental health; the agent interprets these correlates in the context of the individual's history and peer population.

### Evidence to Cite
- 8 sensing modalities = 8 behavioral signal channels
- Multimodal >> Sensing-only: 0.660 vs 0.589
- PA_State 0.733, NA_State 0.722 — direct affect prediction from behavioral signals

### Phrasing Suggestions
- "...PULSE operationalizes behavioral signal processing for passive smartphone sensing, replacing hand-crafted feature extraction with autonomous agentic investigation of behavioral data streams..."
- "...the agentic architecture performs contextual behavioral signal interpretation — examining behavioral patterns in the context of individual history, circadian rhythms, and population norms..."

### Pitfalls to Avoid
- Do not claim PULSE processes raw sensor signals — it processes aggregated behavioral features. The "signal processing" analogy is at the behavioral level, not the electrical level.
- Do not conflate smartphone sensing with the speech/physiological signals that are BSP's traditional domain.

### Reviewer Preemption
- **"This is not behavioral signal processing"** — Acknowledge the broader definition: BSP = computational analysis of human behavioral signals, which includes smartphone-derived behavioral patterns, not just speech/physiological signals.

### Related Work & BibTeX

```bibtex
@article{Narayanan2013BSP,
  author = {Narayanan, Shrikanth S. and Georgiou, Panayiotis G.},
  title = {Behavioral Signal Processing: Deriving Human Behavioral Informatics from Speech and Language},
  journal = {Proceedings of the IEEE},
  year = {2013},
  volume = {101},
  number = {5},
  pages = {1203--1233},
  doi = {10.1109/JPROC.2012.2236291}
}

@article{Gross2015EmotionRegulation,
  author = {Gross, James J.},
  title = {Emotion Regulation: Current Status and Future Prospects},
  journal = {Psychological Inquiry},
  year = {2015},
  volume = {26},
  number = {1},
  pages = {1--26},
  doi = {10.1080/1047840X.2014.940781}
}

@article{Watson1988PANAS,
  author = {Watson, David and Clark, Lee Anna and Tellegen, Auke},
  title = {Development and Validation of Brief Measures of Positive and Negative Affect: The {PANAS} Scales},
  journal = {Journal of Personality and Social Psychology},
  year = {1988},
  volume = {54},
  number = {6},
  pages = {1063--1070},
  doi = {10.1037/0022-3514.54.6.1063}
}
```

---

## Section 2 — Related Work

### Key Arguments
- BSP provides the theoretical framework for understanding *why* agentic investigation works: behavioral signals are inherently contextual — their meaning depends on when, where, and in what personal context they occur. Fixed feature extraction strips this context; agentic investigation preserves it.
- Connect PANAS measurement to sensing: Watson et al. (1988) defined PA and NA as orthogonal dimensions. PULSE predicts both from behavioral sensing, which BSP theory supports — behavioral outputs (activity, sociality, sleep) reflect affective states.
- Multimodal fusion in BSP: the field has long argued that combining multiple behavioral channels improves affect recognition. PULSE's multimodal condition (sensing + diary) confirms this.

### Evidence to Cite
- PA_State Auto-Multi+ 0.733, NA_State Auto-Multi+ 0.722
- Multimodal consistently outperforms sensing-only across all architectures
- Agent investigation traces show cross-modality reasoning

### Phrasing Suggestions
- "...behavioral signal processing theory predicts that contextual interpretation of behavioral signals should outperform decontextualized feature extraction; the 2x2 factorial result confirms this prediction..."
- "...the agentic architecture performs multimodal behavioral signal fusion at the reasoning level rather than the feature level — integrating information from motion, location, screen use, and text through contextual interpretation..."

### Pitfalls to Avoid
- Do not overstretch the BSP analogy — PULSE works at a higher abstraction level than traditional BSP.

### Reviewer Preemption
- **"Where is the signal processing?"** — BSP is about computational extraction of behavioral information from human signals. PULSE does this at the behavioral pattern level using LLM reasoning rather than DSP techniques.

---

## Section 3 — System Design

### Key Arguments
- Each MCP tool is a "channel" in BSP terms — providing a specific view of the behavioral signal. The agent performs *channel selection and fusion* dynamically.
- The agent's investigation strategy is analogous to a BSP pipeline: observe → hypothesize → investigate → synthesize. But it is adaptive rather than fixed.
- Continuous prediction targets (PANAS-PA, PANAS-NA) are directly in BSP's domain. Binary states (ER_desire, INT_avail) bridge BSP with JITAI.

### Evidence to Cite
- 8 MCP tools = 8 behavioral signal channels
- Agent decides investigation strategy per-prediction
- Continuous + binary prediction targets

### Phrasing Suggestions
- "...the MCP tools expose behavioral signals at multiple temporal granularities (hourly, daily, multi-day), enabling the agent to perform multi-scale behavioral signal analysis..."

### Pitfalls to Avoid
- Do not use DSP jargon where it doesn't map cleanly — the tools provide behavioral data, not raw signals.

---

## Section 4 — Evaluation Methodology

### Key Arguments
- Affect measurement: PANAS (Watson et al. 1988) is the gold standard for momentary affect. The study uses 3x daily EMA with PANAS items — methodologically sound from the affect measurement perspective.
- The binarization of continuous affect into states (PA_State, NA_State) may lose information but is necessary for clinical decision-making (JITAI triggers require binary decisions).
- Calibration analysis is essential from the affective computing perspective: continuous predictions show negativity bias for NA and mean regression for PA — known issues in affective computing systems.

### Evidence to Cite
- PANAS-based EMA, 3x daily
- Calibration: negativity bias for NA, mean regression for PA
- Binary robust despite continuous miscalibration

### Phrasing Suggestions
- "...the system exhibits negativity bias for negative affect predictions and mean regression for positive affect — patterns well-documented in affective computing systems that can inform calibration strategies..."

### Pitfalls to Avoid
- Do not claim the continuous predictions are well-calibrated — they are not. The binary thresholding rescues accuracy.

### Reviewer Preemption
- **"Your continuous predictions are poorly calibrated"** — Yes, and we report this transparently. The clinical utility comes from binary state detection, where the system is robust.

---

## Section 5 — Results

### Key Arguments
- Affect detection accuracy: PA_State 0.733, NA_State 0.722 — competitive with state-of-the-art affective computing from curated lab data, achieved here from in-the-wild smartphone sensing.
- The negative affect finding is particularly notable: NA is harder to predict from behavior (people mask distress). Agentic investigation achieves 0.722 by examining subtle behavioral patterns.
- Multimodal >> Sensing for NA: diary text carries direct emotional expression. But sensing-only still achieves meaningful prediction through behavioral indicators.

### Evidence to Cite
- PA_State: Auto-Multi+ 0.733 vs Struct-Sense 0.505
- NA_State: Auto-Multi+ 0.722 vs Struct-Sense 0.510
- PA improvement from multimodal: 0.598 → 0.733
- NA improvement from multimodal: 0.592 → 0.722

### Phrasing Suggestions
- "...achieving 0.722 balanced accuracy for negative affect state from in-the-wild passive sensing and diary text is notable in the affective computing literature, where comparable accuracy has typically required controlled laboratory conditions with richer signal modalities..."

### Pitfalls to Avoid
- Do not compare directly to lab-based affective computing systems — different settings, different signals.

### Reviewer Preemption
- **"These numbers are lower than speech-based affect recognition"** — Different domain (in-the-wild vs lab), different signals (smartphone vs speech/physiological), different population (cancer survivors with heterogeneous emotional profiles).

---

## Section 6 — Discussion

### Key Arguments
- 6.1: Why agentic investigation works — from a BSP perspective, the agent performs *contextual behavioral signal interpretation*. It doesn't just extract features; it reasons about what behavioral patterns *mean* in context.
- 6.5: Calibration as a research opportunity: the continuous prediction biases (negativity for NA, mean regression for PA) are addressable through post-hoc calibration techniques from the affective computing literature.
- 6.7: Ethical considerations — BSP has long grappled with privacy and surveillance concerns. The same applies here: passive sensing + LLM API creates a particularly sensitive pipeline.

### Evidence to Cite
- Agent investigation traces showing contextual reasoning
- Calibration analysis results
- IRB, consent, de-identification details

### Phrasing Suggestions
- "...the agentic architecture performs behavioral signal interpretation at a level of contextual sophistication that fixed feature extraction cannot achieve — examining temporal patterns, cross-modal correlations, and individual baselines adaptively..."
- "...from a behavioral signal processing perspective, the key insight is that contextual interpretation of behavioral signals, not just their extraction, is the bottleneck for affect prediction from in-the-wild data..."

### Pitfalls to Avoid
- Do not claim PULSE is the first system to reason about behavioral signals contextually — BSP has always emphasized context. PULSE automates it.

### Reviewer Preemption
- **"An LLM is a black box — how do you know what it's reasoning about?"** — The agentic traces provide observability into the reasoning process. Each tool call and its result are logged. This is more interpretable than traditional black-box ML.

---

## Sections 7-8 — Future Work & Conclusion

### Key Arguments
- Future: integrate richer behavioral signals (speech prosody, physiological data from wearables) as additional MCP tools. This would bring PULSE closer to full BSP with agentic LLM reasoning.
- Cross-model comparison: test with different LLMs to verify the architecture-level contribution.
- The paradigm contribution from BSP perspective: agentic investigation is a new modality of behavioral signal processing — one that uses natural language reasoning over behavioral data.

### Phrasing Suggestions
- "...agentic sensing investigation represents a new paradigm for behavioral signal processing, where the interpretation of behavioral signals is delegated to an LLM agent with domain-specific tools rather than hand-crafted feature pipelines..."

---

# Shared BibTeX — Core References (All Experts)

These BibTeX entries are referenced by multiple experts and should be included in the paper's bibliography.

```bibtex
@inproceedings{Yao2023ReAct,
  author = {Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  title = {{ReAct}: Synergizing Reasoning and Acting in Language Models},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}

@inproceedings{Schick2023Toolformer,
  author = {Schick, Timo and Dwivedi-Yu, Jane and Dess{\`i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  title = {Toolformer: Language Models Can Teach Themselves to Use Tools},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2023}
}

@misc{Anthropic2024MCP,
  author = {{Anthropic}},
  title = {Model Context Protocol},
  year = {2024},
  howpublished = {\url{https://modelcontextprotocol.io}},
  note = {Open standard for LLM-tool integration}
}

@article{Merrill2026PHIA,
  author = {Merrill, Mike A. and Paruchuri, Akshay and Rezaei, Naghmeh and Kovacs, Geza and Perez, Javier and Liu, Yun and Schenck, Erik and Hammerquist, Nova and Sunshine, Jake and Tailor, Shyam and others},
  title = {Transforming Wearable Data into Personal Health Insights Using Large Language Model Agents},
  journal = {Nature Communications},
  year = {2026},
  doi = {10.1038/s41467-025-67922-y}
}

@article{Choube2025GLOSS,
  author = {Choube, Akshat and Le, Ha and Li, Jiachen and Ji, Kaixin and Das Swain, Vedant and Mishra, Varun},
  title = {{GLOSS}: Group of {LLMs} for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing},
  journal = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  year = {2025},
  volume = {9},
  number = {3},
  pages = {1--32},
  doi = {10.1145/3749474}
}
```

---

# Cross-Expert Positioning Table Reference

For the positioning table in Section 2, compare PULSE against these systems on these dimensions:

| System | Year | Data Source | LLM? | Agentic? | Tools? | Clinical Pop? | Prediction Targets |
|--------|------|------------|------|----------|--------|--------------|-------------------|
| StudentLife | 2014 | Sensing | No | No | No | Students | Correlations only |
| CrossCheck | 2016 | Sensing | No | No | No | Schizophrenia | Symptom scores |
| GLOBEM | 2022 | Sensing | No | No | No | Students | Depression |
| Health-LLM | 2024 | Wearable | Yes | No | No | General | 10 health tasks |
| Mental-LLM | 2024 | Text | Yes | No | No | Online | Mental health |
| Zhang et al. | 2024 | Sensing | Yes | No | No | Students | Affect |
| MindScape | 2024 | Sensing+LLM | Yes | No | No | Students+SMI | Journaling |
| PHIA | 2026 | Wearable | Yes | Yes | Code gen | General | Health Q&A |
| GLOSS | 2025 | Sensing | Yes | Yes* | Code gen | General | Sensemaking |
| CALLM | 2025 | Diary | Yes | No | No | Cancer | Affect+ER |
| **PULSE** | **2025** | **Sensing+Diary** | **Yes** | **Yes** | **8 MCP tools** | **Cancer** | **Affect+ER+INT** |

*GLOSS uses multi-agent debate, not tool-use agentic investigation.

---

# Universal Reviewer Preemption Strategies

These concerns will arise regardless of expert framing:

1. **"LLM inference is expensive"** — Acknowledge 30-90s per prediction. Frame as proof-of-concept; cost decreases with model optimization and edge deployment.

2. **"This only works with Claude"** — Acknowledge model dependency. The architecture (tool use, factorial design) is model-agnostic; results may vary across LLMs.

3. **"N=50 is small"** — Present representativeness analysis prominently. The 2x2 comparison is internally valid. The ML baselines on 399 users provide population-level reference.

4. **"Retrospective evaluation"** — Yes. Frame as necessary first step before real-time deployment. Prospective evaluation is future work.

5. **"Privacy concerns with LLM API"** — Discuss in ethics section. Data is de-identified before API calls. Future work: on-device LLMs.

6. **"The ML comparison is unfair"** — Agree. ML baselines are reference points, not head-to-head comparisons. The apples-to-apples comparison is agentic vs structured (same data, same LLM, same 50 users).

7. **"Where is the ablation for memory?"** — Acknowledge as limitation. Memory adds longitudinal context but we cannot quantify its isolated contribution in this evaluation.

8. **"Why not fine-tune instead of prompting?"** — Agentic investigation cannot be fine-tuned — it requires tool use and multi-turn reasoning. Fine-tuning would collapse the architecture to structured prompting.
