# PULSE Citation Map (Expanded)

Maps each paper section to all references. Reference keys match `references_expanded.bib`.
Total unique references: **84**

---

## 1. Introduction (~2 pages)

### Opening hook: cancer survivors and mental health burden
- `mitchell2011depression` -- meta-analysis: 16-38% depression prevalence in cancer settings (Lancet Oncology)
- `andersen2023asco` -- ASCO guideline: management of anxiety/depression in cancer survivors
- `spring2019cancersurvivors` -- cancer survivor health promotion needs
- `mattingly2025predictors` -- same BUCS cohort, intervention preferences vary with momentary affect

### The diary paradox
- `shiffman2008ema` -- foundational EMA methodology; supports diary compliance issues
- `wang2025callm` -- CALLM demonstrates diary-based prediction but exposes the paradox

### Passive sensing opportunity
- `wang2014studentlife` -- seminal passive sensing for mental health
- `saeb2015mobile` -- early digital phenotyping linking sensors to depression
- `onnela2016harnessing` -- defined "digital phenotyping" concept
- `xu2022globem` -- reveals ceiling of traditional ML on sensing (~0.52 BA, poor generalization)

### Core insight: agentic LLM investigation
- `yao2023react` -- foundational agentic reasoning paradigm
- `xu2024penetrative` -- LLMs "penetrating" into physical-world sensor data

### Contributions list
- `nahumshani2018jitai` -- JITAI framework that PULSE operationalizes
- `mishra2021receptivity` -- receptivity prediction that PULSE extends
- `gross2015emotion` -- emotion regulation framework relevant to ER_desire prediction

---

## 2. Related Work (~3 pages)

### 2.1 Passive Sensing for Mental Health (~15-20 refs)
- `wang2014studentlife` -- foundational (2014), UbiComp 10-Year Impact Award
- `saeb2015mobile` -- foundational (2015), phone sensors correlate with depression
- `canzian2015trajectories` -- mobility traces predict depressive states (UbiComp 2015, 10-Year Impact Award 2025)
- `ferreira2015aware` -- AWARE framework for mobile context sensing
- `onnela2016harnessing` -- coined "digital phenotyping" (Neuropsychopharmacology)
- `wang2016crosscheck` -- CrossCheck: passive sensing for schizophrenia in clinical populations
- `torous2017new` -- digital phenotyping for RDoC and psychiatry
- `wang2018tracking` -- depression dynamics via smartphone+wearable sensing
- `jacobson2019digital` -- digital biomarkers of mood disorders from actigraphy
- `huckins2020mental` -- COVID-19 impact on college students via smartphone sensing
- `chikersal2021detecting` -- depression detection from longitudinal passive sensing
- `adler2022machine` -- ML generalization across longitudinal sensing studies
- `xu2022globem` -- GLOBEM cross-dataset generalization benchmark; reveals ML ceiling
- `meegahapola2023generalization` -- mood inference generalization across 8 countries
- `dpmentalhealth2023review` -- systematic review: digital phenotyping for monitoring mental disorders
- `nepal2024college` -- four-year mobile sensing study of college students (IMWUT 2024)
- `adler2024beyond` -- "Beyond Detection" -- field must move from detection to action
- `dpstressanxiety2024` -- digital phenotyping for stress, anxiety, mild depression
- `feng2026comparative` -- ML vs DL vs LLM benchmark on smartphone sensing

### 2.2 LLMs for Affective Computing and Health (~15-20 refs)
- `brown2020language` -- GPT-3: foundational few-shot learning capabilities
- `ouyang2022instructgpt` -- InstructGPT: RLHF alignment for instruction following
- `singhal2023medpalm` -- Med-PaLM: LLMs encode clinical knowledge (Nature)
- `cosentino2024phllm` -- PH-LLM: personal health LLM for sleep/fitness (Nature Medicine)
- `kim2024healthllm` -- Health-LLM: 12 LLMs on 10 wearable health tasks
- `xu2024mentalllm` -- Mental-LLM: LLMs for mental health from online text (IMWUT)
- `zhang2024leveraging` -- first LLM+smartphone sensing for affect prediction
- `zhang2024sentiment` -- sentiment analysis reality check in LLM era (NAACL 2024)
- `wang2025callm` -- CALLM: direct predecessor, diary+RAG
- `choube2025gloss` -- GLOSS: multi-LLM sensemaking of passive sensing (IMWUT 2025)
- `phia2026` -- PHIA: LLM agent for wearable data Q&A (Nature Communications)
- `nepal2024mindscape` -- MindScape: LLM+sensing for journaling intervention
- `li2025vitalinsight` -- Vital Insight: human-in-the-loop LLM for sensing sensemaking
- `xu2025lens` -- LENS: sensor-to-LLM alignment for mental health narratives
- `an2024iotllm` -- IoT-LLM: framework for LLM reasoning over IoT sensor data
- `wang2024slm` -- small language models for on-device mHealth prediction
- `zhang2024ondevice` -- on-device LLM personalization with smartphone sensing
- `sensorllm2025` -- SensorLLM: aligning LLMs with motion sensors (EMNLP 2025)
- `llm_wearable_survey2024` -- survey: LLMs for wearable sensor-based health monitoring
- `feng2026comparative` -- direct methodological comparison (ML/DL/LLM on sensing)

### 2.3 Agentic AI and Tool Use (~10 refs)
- `wei2022chain` -- chain-of-thought prompting enables reasoning (NeurIPS 2022)
- `yao2023react` -- ReAct: reasoning + acting paradigm
- `schick2023toolformer` -- Toolformer: self-supervised tool use (NeurIPS 2023 oral)
- `wang2023voyager` -- Voyager: embodied agent with skill library (NeurIPS 2023)
- `patil2023gorilla` -- Gorilla: LLM connected with massive APIs for function calling
- `lewis2020rag` -- RAG: retrieval-augmented generation (foundational, NeurIPS 2020)
- `yang2025contextagent` -- ContextAgent: proactive LLM agent with wearable sensing (NeurIPS 2025)
- `anthropic2024mcp` -- Model Context Protocol specification
- `wang2024agentsurvey` -- comprehensive survey of LLM-based autonomous agents
- `huang2024planning` -- survey: understanding LLM agent planning
- `gao2024ragsurvey` -- RAG for healthcare: systematic review

### 2.4 Just-in-Time Adaptive Interventions (~12 refs)
- `nahumshani2014jitai` -- original six-component JITAI organizing model
- `nahumshani2018jitai` -- foundational JITAI framework
- `klasnja2015mrt` -- microrandomized trials for JITAI development
- `klasnja2019heartsteps` -- HeartSteps MRT for physical activity JITAI
- `kunzler2019receptivity` -- state-of-receptivity for mHealth (IMWUT 2019)
- `mishra2021receptivity` -- receptivity detection in the wild (IMWUT 2021, Distinguished Paper)
- `mohr2017intellicare` -- IntelliCare: skills-based app suite for depression/anxiety
- `lattie2019digital` -- digital mental health interventions for college students (review)
- `jitai_effectiveness2025` -- meta-analysis: JITAI effectiveness for mental health
- `heron2010emi` -- ecological momentary interventions framework
- `emi_scoping2021` -- EMI for mental health: scoping review
- `mello_emt2023` -- notification timing in behavior change apps (MRT)

### Positioning table
Compare PULSE vs:
- `wang2025callm` (CALLM)
- `kim2024healthllm` (Health-LLM)
- `xu2024mentalllm` (Mental-LLM)
- `zhang2024leveraging` (Zhang et al. 2024)
- `choube2025gloss` (GLOSS)
- `phia2026` (PHIA)
- `nepal2024mindscape` (MindScape)
- `feng2026comparative` (Feng et al.)
- `mishra2021receptivity` (Mishra et al.)
- `xu2022globem` (GLOBEM traditional ML)

---

## 3. System Design (~5 pages)

### 3.1 Problem Formulation
- `nahumshani2018jitai` -- JITAI tailoring variables (decision points, intervention options)
- `kunzler2019receptivity` -- receptivity = desire + availability decomposition
- `mishra2021receptivity` -- receptivity prediction constructs
- `gross2015emotion` -- emotion regulation: ER_desire as desire to change emotional state
- `watson1988panas` -- PANAS: affect measurement instrument
- `wang2025callm` -- same BUCS dataset definition and prediction targets
- `mattingly2025predictors` -- same cohort, establishes clinical constructs

### 3.2 Factorial Design
- `wang2025callm` -- CALLM baseline condition

### 3.3 BUCS Dataset
- `wang2025callm` -- dataset description (418 cancer survivors, 5 weeks, 3x daily EMA)
- `mattingly2025predictors` -- same dataset, complementary analysis
- `shiffman2008ema` -- EMA methodology grounding

### 3.4 Structured Agents
- `wang2025callm` -- RAG design inherited from CALLM
- `lewis2020rag` -- RAG: foundational retrieval-augmented generation approach

### 3.5 Agentic Agents
- `yao2023react` -- ReAct paradigm underlying the agentic investigation loop
- `anthropic2024mcp` -- MCP tool interface protocol
- `wang2023voyager` -- skill library analogy for tool-based investigation
- `wei2022chain` -- chain-of-thought reasoning in agent traces

### 3.6 Cross-User RAG
- `lewis2020rag` -- RAG foundational approach
- `xu2022globem` -- cross-user generalization challenge

---

## 4. Evaluation Methodology (~2 pages)

### 4.1 Evaluation Setup
- `shiffman2008ema` -- EMA methodology grounding
- `watson1988panas` -- PANAS scales used for affect measurement

### 4.2 Baselines
- `xu2022globem` -- contextualizes ML baseline performance levels
- `wang2025callm` -- CALLM as LLM baseline
- `zhang2024leveraging` -- LLM+sensing baseline context
- `feng2026comparative` -- comparative ML/DL/LLM context

### 4.3 Metrics
- (standard metrics; statistical references as needed)

### 4.4 Implementation
- `anthropic2024mcp` -- MCP tool protocol
- `vaswani2017attention` -- transformer architecture underlying LLMs

---

## 5. Results (~5 pages)

### 5.1 Aggregate Performance
- `xu2022globem` -- contextualizes BA ~0.52 ML ceiling
- `wang2025callm` -- CALLM BA=0.611 comparison
- `feng2026comparative` -- contextualizes LLM performance on sensing
- `adler2022machine` -- ML generalization baseline context

### 5.2 Intervention Opportunity Detection
- `kunzler2019receptivity` -- receptivity prediction context
- `mishra2021receptivity` -- receptivity detection baselines
- `nahumshani2018jitai` -- JITAI tailoring variable relevance
- `gross2015emotion` -- emotion regulation desire construct

### 5.3 Affect State Detection
- `kim2024healthllm` -- Health-LLM performance context
- `zhang2024leveraging` -- LLM affect prediction context
- `watson1988panas` -- PANAS scale interpretation

### 5.4 Agentic vs. Structured (2x2 result)
- (primary empirical section; cite comparison systems from Section 2)

### 5.5 Sensing-Only vs. Multimodal
- `adler2024beyond` -- actionable sensing beyond detection
- `shiffman2008ema` -- EMA missing data patterns

### 5.7 Agentic Investigation Patterns
- `yao2023react` -- ReAct trace structure
- `wei2022chain` -- chain-of-thought reasoning patterns
- `narayanan2013bsp` -- behavioral signal interpretation lens

---

## 6. Discussion (~3 pages)

### 6.1 Why Agentic Investigation Works
- `narayanan2013bsp` -- Behavioral Signal Processing framework: LLM as behavioral signal interpreter
- `xu2024penetrative` -- Penetrative AI: LLMs comprehending physical-world sensor data
- `yao2023react` -- agent reasoning traces analogous to ReAct
- `wei2022chain` -- chain-of-thought enables selective attention

### 6.2 The Diary Paradox and Graceful Degradation
- `shiffman2008ema` -- EMA compliance and MNAR issues
- `wang2025callm` -- CALLM's diary dependency as the problem PULSE solves
- `adler2024beyond` -- actionable sensing even when diaries missing

### 6.3 INT_availability as a Behavioral Construct
- `kunzler2019receptivity` -- receptivity decomposition
- `mishra2021receptivity` -- availability as a contextual/behavioral signal
- `nahumshani2018jitai` -- JITAI tailoring variables

### 6.4 Cross-User RAG as Calibration
- `xu2022globem` -- GLOBEM generalization challenge; agents may bypass it
- `lewis2020rag` -- RAG framework adapted for calibration
- `meegahapola2023generalization` -- generalization challenges in mood inference

### 6.5 Limitations
- `feng2026comparative` -- model dependency, temporal modeling limitations
- `adler2022machine` -- generalization concerns in sensing ML
- `touvron2023llama` -- open-source LLM alternatives for reproducibility

### 6.6 Implications for JITAI Design
- `nahumshani2018jitai` -- JITAI design principles
- `klasnja2015mrt` -- MRT as next-step evaluation methodology
- `klasnja2019heartsteps` -- HeartSteps as example of deployed JITAI
- `spring2019cancersurvivors` -- cancer survivor health promotion systems
- `andersen2023asco` -- ASCO guideline for cancer survivor mental health management
- `jitai_effectiveness2025` -- meta-analysis supports JITAI effectiveness

### 6.7 Ethical Considerations
- `adler2024beyond` -- actionable sensing and clinical safety concerns
- `coravos2019digital` -- safe and effective digital biomarkers framework
- `privacy_fl_health2024` -- privacy preservation in health data
- `explainability_llm2025` -- explainability challenges in LLM-based health AI
- `vaidyam2019chatbots` -- psychiatric chatbot landscape and safety considerations

---

## 7. Future Work (~0.5 pages)

- `yang2025contextagent` -- proactive agents with egocentric sensing (complementary modality)
- `wang2024slm` -- edge/small LMs for on-device deployment
- `xu2025lens` -- fine-tuned sensor-LLM alignment as alternative approach
- `edge_llm_survey2024` -- mobile edge intelligence for LLMs
- `zhang2024ondevice` -- on-device LLM personalization framework
- `touvron2023llama` -- open-source LLMs for reproducibility and on-device deployment
- `sensorllm2025` -- sensor-language foundation models
- `digital_cancer_meta2025` -- digital interventions for cancer survivors
- `smartphone_cancer_survivors2022` -- smartphone interventions for cancer survivors

---

## 8. Conclusion (~0.5 pages)

- `wang2014studentlife` -- passive sensing tradition PULSE extends
- `nahumshani2018jitai` -- JITAI framework PULSE operationalizes
- `yao2023react` -- agentic paradigm PULSE instantiates
- `watson1988panas` -- affect measurement PULSE predicts

---

## Summary: Papers by Topic Area

### Passive sensing for mental health: 19 papers
`wang2014studentlife`, `saeb2015mobile`, `canzian2015trajectories`, `ferreira2015aware`, `onnela2016harnessing`, `wang2016crosscheck`, `torous2017new`, `wang2018tracking`, `jacobson2019digital`, `huckins2020mental`, `chikersal2021detecting`, `adler2022machine`, `xu2022globem`, `meegahapola2023generalization`, `dpmentalhealth2023review`, `nepal2024college`, `adler2024beyond`, `dpstressanxiety2024`, `feng2026comparative`

### LLMs for health/affective computing: 20 papers
`brown2020language`, `ouyang2022instructgpt`, `singhal2023medpalm`, `cosentino2024phllm`, `kim2024healthllm`, `xu2024mentalllm`, `zhang2024leveraging`, `zhang2024sentiment`, `wang2025callm`, `choube2025gloss`, `phia2026`, `nepal2024mindscape`, `li2025vitalinsight`, `xu2025lens`, `an2024iotllm`, `wang2024slm`, `zhang2024ondevice`, `sensorllm2025`, `llm_wearable_survey2024`, `feng2026comparative`

### Predecessor/same-dataset work: 2 papers
`wang2025callm`, `mattingly2025predictors`

### JITAI, receptivity, and intervention: 12 papers
`nahumshani2014jitai`, `nahumshani2018jitai`, `klasnja2015mrt`, `klasnja2019heartsteps`, `kunzler2019receptivity`, `mishra2021receptivity`, `mohr2017intellicare`, `lattie2019digital`, `jitai_effectiveness2025`, `heron2010emi`, `emi_scoping2021`, `mello_emt2023`

### Agentic AI and tool use: 11 papers
`wei2022chain`, `yao2023react`, `schick2023toolformer`, `wang2023voyager`, `patil2023gorilla`, `lewis2020rag`, `yang2025contextagent`, `anthropic2024mcp`, `wang2024agentsurvey`, `huang2024planning`, `gao2024ragsurvey`

### Behavioral signal processing / physical world: 2 papers
`narayanan2013bsp`, `xu2024penetrative`

### EMA methodology: 1 paper
`shiffman2008ema`

### Cancer survivorship and mHealth: 8 papers
`mitchell2011depression`, `spring2019cancersurvivors`, `andersen2023asco`, `wang2022mhealth`, `smartphone_cancer_survivors2022`, `digital_cancer_meta2025`, `emi_depression_meta2026`, `ema_cancer_delivery2021`

### Emotion regulation and affect measurement: 2 papers
`gross2015emotion`, `watson1988panas`

### Foundation models and transformers: 2 papers
`vaswani2017attention`, `touvron2023llama`

### Digital health ethics/safety: 5 papers
`coravos2019digital`, `vaidyam2019chatbots`, `benzeev2018focus`, `privacy_fl_health2024`, `explainability_llm2025`

### Stress/multimodal sensing: 1 paper
`dpstressanxiety2024`

### Other: 1 paper
`edge_llm_survey2024`

---

## Total unique references: 84

### New additions (48 papers added to original 36):
1. `canzian2015trajectories` -- Mobility traces for depression monitoring (UbiComp 2015)
2. `ferreira2015aware` -- AWARE sensing framework
3. `onnela2016harnessing` -- Digital phenotyping definition
4. `wang2016crosscheck` -- CrossCheck for schizophrenia
5. `torous2017new` -- Digital phenotyping for RDoC
6. `jacobson2019digital` -- Digital biomarkers of mood disorders
7. `huckins2020mental` -- COVID-19 impact on students via sensing
8. `chikersal2021detecting` -- Depression detection from passive sensing
9. `dpmentalhealth2023review` -- Digital phenotyping systematic review
10. `nepal2024college` -- Four-year college student sensing study
11. `dpstressanxiety2024` -- Stress/anxiety digital phenotyping review
12. `brown2020language` -- GPT-3 (NeurIPS 2020)
13. `ouyang2022instructgpt` -- InstructGPT / RLHF (NeurIPS 2022)
14. `singhal2023medpalm` -- Med-PaLM clinical knowledge (Nature 2023)
15. `cosentino2024phllm` -- PH-LLM for personal health (Nature Medicine)
16. `zhang2024sentiment` -- LLM sentiment analysis reality check (NAACL)
17. `zhang2024ondevice` -- On-device LLM personalization (UbiComp 2024)
18. `sensorllm2025` -- SensorLLM for motion sensors (EMNLP 2025)
19. `llm_wearable_survey2024` -- LLM + wearable survey
20. `wei2022chain` -- Chain-of-thought prompting (NeurIPS 2022)
21. `patil2023gorilla` -- Gorilla: LLM + API function calling
22. `lewis2020rag` -- RAG original paper (NeurIPS 2020)
23. `huang2024planning` -- LLM agent planning survey
24. `gao2024ragsurvey` -- RAG for healthcare review
25. `watson1988panas` -- PANAS scales (1988)
26. `gross2015emotion` -- Emotion regulation review (2015)
27. `vaswani2017attention` -- Transformer architecture (NeurIPS 2017)
28. `touvron2023llama` -- LLaMA open models (Meta 2023)
29. `mitchell2011depression` -- Depression prevalence in cancer (Lancet Oncology)
30. `andersen2023asco` -- ASCO guideline for cancer survivor mental health
31. `coravos2019digital` -- Digital biomarkers safety framework
32. `vaidyam2019chatbots` -- Chatbots in psychiatry review
33. `benzeev2018focus` -- FOCUS smartphone intervention for schizophrenia
34. `mohr2017intellicare` -- IntelliCare app suite
35. `lattie2019digital` -- Digital mental health for college students review
36. `heron2010emi` -- Ecological momentary interventions framework
37. `emi_scoping2021` -- EMI scoping review
38. `jitai_effectiveness2025` -- JITAI effectiveness meta-analysis
39. `wang2022mhealth` -- mHealth app meta-analysis for cancer survivors
40. `smartphone_cancer_survivors2022` -- Smartphone psychotherapy for cancer survivors
41. `digital_cancer_meta2025` -- Digital interventions for cancer mental health
42. `emi_depression_meta2026` -- EMI for depression meta-analysis
43. `ema_cancer_delivery2021` -- EMA/EMI delivery review
44. `mello_emt2023` -- Notification timing in mHealth MRT
45. `ai_chatbot_meta2024` -- AI chatbot for mental health meta-analysis
46. `privacy_fl_health2024` -- Privacy-preserving federated learning
47. `explainability_llm2025` -- LLM explainability in healthcare
48. `edge_llm_survey2024` -- Mobile edge intelligence for LLMs
49. `huckins2020mental` -- COVID-era college sensing
50. `nepal2024college` -- Four-year sensing longitudinal study
51. `dpmentalhealth2023review` -- Digital phenotyping systematic review
52. `dpstressanxiety2024` -- Stress/anxiety phenotyping review
