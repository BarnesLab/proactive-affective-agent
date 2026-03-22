# PULSE Citation Map

Maps each paper section to the references it should cite. Reference keys match `references.bib`.

---

## 1. Introduction (~2 pages)

### Opening hook: cancer survivors and mental health burden
- `spring2019cancersurvivors` -- cancer survivor health promotion needs
- `mattingly2025predictors` -- same BUCS cohort, intervention preferences vary with momentary affect

### The diary paradox
- `shiffman2008ema` -- foundational EMA methodology; supports diary compliance issues
- `wang2025callm` -- CALLM demonstrates diary-based prediction but exposes the paradox

### Passive sensing opportunity
- `wang2014studentlife` -- seminal passive sensing for mental health
- `saeb2015mobile` -- early digital phenotyping linking sensors to depression
- `xu2022globem` -- reveals ceiling of traditional ML on sensing (~0.52 BA, poor generalization)

### Core insight: agentic LLM investigation
- `yao2023react` -- foundational agentic reasoning paradigm
- `xu2024penetrative` -- LLMs "penetrating" into physical-world sensor data (conceptual framing)

### Contributions list
- `nahumshani2018jitai` -- JITAI framework that PULSE operationalizes
- `mishra2021receptivity` -- receptivity prediction that PULSE extends

---

## 2. Related Work (~3 pages)

### 2.1 Passive Sensing for Mental Health
- `wang2014studentlife` -- foundational (2014)
- `saeb2015mobile` -- foundational (2015)
- `wang2018tracking` -- depression dynamics via smartphone+wearable sensing
- `xu2022globem` -- cross-dataset generalization benchmark; reveals ML ceiling
- `meegahapola2023generalization` -- mood inference generalization across 8 countries
- `adler2022machine` -- ML generalization across longitudinal sensing studies
- `adler2024beyond` -- "Beyond Detection" -- field must move from detection to action
- `feng2026comparative` -- ML vs DL vs LLM benchmark on smartphone sensing

### 2.2 LLMs for Affective Computing and Health
- `kim2024healthllm` -- Health-LLM: 12 LLMs on 10 wearable health tasks
- `xu2024mentalllm` -- Mental-LLM: LLMs for mental health from online text (IMWUT)
- `zhang2024leveraging` -- first LLM+smartphone sensing for affect prediction
- `wang2025callm` -- CALLM: direct predecessor, diary+RAG
- `choube2025gloss` -- GLOSS: multi-LLM sensemaking of passive sensing (IMWUT 2025)
- `phia2026` -- PHIA: LLM agent for wearable data Q&A (Nature Communications)
- `nepal2024mindscape` -- MindScape: LLM+sensing for journaling intervention
- `li2025vitalinsight` -- Vital Insight: human-in-the-loop LLM for sensing sensemaking
- `xu2025lens` -- LENS: sensor-to-LLM alignment for mental health narratives
- `an2024iotllm` -- IoT-LLM: framework for LLM reasoning over IoT sensor data
- `wang2024slm` -- small language models for on-device mHealth prediction
- `feng2026comparative` -- direct methodological comparison (ML/DL/LLM on sensing)

### 2.3 Agentic AI and Tool Use
- `yao2023react` -- ReAct: reasoning + acting paradigm
- `schick2023toolformer` -- Toolformer: self-supervised tool use
- `wang2023voyager` -- Voyager: embodied agent with skill library (NeurIPS 2023)
- `yang2025contextagent` -- ContextAgent: proactive LLM agent with wearable sensing (NeurIPS 2025)
- `anthropic2024mcp` -- Model Context Protocol specification
- `wang2024agentsurvey` -- comprehensive survey of LLM-based autonomous agents

### 2.4 Just-in-Time Adaptive Interventions
- `nahumshani2018jitai` -- foundational JITAI framework
- `nahumshani2014jitai` -- original six-component JITAI organizing model
- `klasnja2015mrt` -- microrandomized trials for JITAI development
- `klasnja2019heartsteps` -- HeartSteps MRT for physical activity JITAI
- `kunzler2019receptivity` -- state-of-receptivity for mHealth (IMWUT 2019)
- `mishra2021receptivity` -- receptivity detection in the wild (IMWUT 2021, Distinguished Paper)

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
- `wang2025callm` -- same BUCS dataset definition and prediction targets
- `mattingly2025predictors` -- same cohort, establishes clinical constructs

### 3.2 Factorial Design
- `wang2025callm` -- CALLM baseline condition

### 3.3 BUCS Dataset
- `wang2025callm` -- dataset description (418 cancer survivors, 5 weeks, 3x daily EMA)
- `mattingly2025predictors` -- same dataset, complementary analysis

### 3.4 Structured Agents
- `wang2025callm` -- RAG design inherited from CALLM

### 3.5 Agentic Agents
- `yao2023react` -- ReAct paradigm underlying the agentic investigation loop
- `anthropic2024mcp` -- MCP tool interface protocol
- `wang2023voyager` -- skill library analogy for tool-based investigation

### 3.6 Cross-User RAG
- (no additional citations beyond those in 3.1)

---

## 4. Evaluation Methodology (~2 pages)

### 4.1 Evaluation Setup
- `shiffman2008ema` -- EMA methodology grounding

### 4.2 Baselines
- `xu2022globem` -- contextualizes ML baseline performance levels
- `wang2025callm` -- CALLM as LLM baseline
- `zhang2024leveraging` -- LLM+sensing baseline context

### 4.3 Metrics
- (standard metrics; no special citations needed beyond statistical method references)

### 4.4 Implementation
- `anthropic2024mcp` -- MCP tool protocol

---

## 5. Results (~5 pages)

### 5.1 Aggregate Performance
- `xu2022globem` -- contextualizes BA ~0.52 ML ceiling
- `wang2025callm` -- CALLM BA=0.611 comparison
- `feng2026comparative` -- contextualizes LLM performance on sensing

### 5.2 Intervention Opportunity Detection
- `kunzler2019receptivity` -- receptivity prediction context
- `mishra2021receptivity` -- receptivity detection baselines
- `nahumshani2018jitai` -- JITAI tailoring variable relevance

### 5.3 Affect State Detection
- `kim2024healthllm` -- Health-LLM performance context
- `zhang2024leveraging` -- LLM affect prediction context

### 5.4 Agentic vs. Structured (2x2 result)
- (primary empirical section; cite comparison systems from Section 2)

### 5.7 Agentic Investigation Patterns
- `yao2023react` -- ReAct trace structure

---

## 6. Discussion (~3 pages)

### 6.1 Why Agentic Investigation Works
- `narayanan2013bsp` -- Behavioral Signal Processing framework: LLM as behavioral signal interpreter
- `xu2024penetrative` -- Penetrative AI: LLMs comprehending physical-world sensor data
- `yao2023react` -- agent reasoning traces analogous to ReAct

### 6.2 The Diary Paradox and Graceful Degradation
- `shiffman2008ema` -- EMA compliance and MNAR issues
- `wang2025callm` -- CALLM's diary dependency as the problem PULSE solves

### 6.3 INT_availability as a Behavioral Construct
- `kunzler2019receptivity` -- receptivity decomposition
- `mishra2021receptivity` -- availability as a contextual/behavioral signal
- `nahumshani2018jitai` -- JITAI tailoring variables

### 6.4 Cross-User RAG as Calibration
- `xu2022globem` -- GLOBEM generalization challenge; agents may bypass it

### 6.5 Limitations
- `feng2026comparative` -- model dependency, temporal modeling limitations
- `adler2022machine` -- generalization concerns in sensing ML

### 6.6 Implications for JITAI Design
- `nahumshani2018jitai` -- JITAI design principles
- `klasnja2015mrt` -- MRT as next-step evaluation methodology
- `klasnja2019heartsteps` -- HeartSteps as example of deployed JITAI
- `spring2019cancersurvivors` -- cancer survivor health promotion systems

### 6.7 Ethical Considerations
- `adler2024beyond` -- actionable sensing and clinical safety concerns

---

## 7. Future Work (~0.5 pages)
- `yang2025contextagent` -- proactive agents with egocentric sensing (complementary modality)
- `wang2024slm` -- edge/small LMs for on-device deployment
- `xu2025lens` -- fine-tuned sensor-LLM alignment as alternative approach

---

## 8. Conclusion (~0.5 pages)
- `wang2014studentlife` -- passive sensing tradition PULSE extends
- `nahumshani2018jitai` -- JITAI framework PULSE operationalizes
- `yao2023react` -- agentic paradigm PULSE instantiates

---

## Summary: Papers by Expert Citation Count

| Paper | Key | Expert Count | Sections |
|-------|-----|:---:|----------|
| CALLM (Wang 2025) | `wang2025callm` | 10/10 | 1, 2.2, 3, 4, 5, 6 |
| JITAI framework (Nahum-Shani 2018) | `nahumshani2018jitai` | 10/10 | 1, 2.4, 3, 5, 6, 8 |
| StudentLife (Wang 2014) | `wang2014studentlife` | 10/10 | 1, 2.1, 8 |
| Detecting Receptivity (Mishra 2021) | `mishra2021receptivity` | 10/10 | 1, 2.4, 3, 5, 6 |
| LLMs for Affective States (Zhang 2024) | `zhang2024leveraging` | 10/10 | 2.2, 4, 5 |
| State-of-Receptivity (Kunzler 2019) | `kunzler2019receptivity` | 9/10 | 2.4, 3, 5, 6 |
| Saeb et al. (2015) | `saeb2015mobile` | 8/10 | 1, 2.1 |
| ReAct (Yao 2023) | `yao2023react` | 8/10 | 1, 2.3, 3, 5, 6, 8 |
| GLOBEM (Xu 2022) | `xu2022globem` | 7/10 | 1, 2.1, 4, 5, 6 |
| Health-LLM (Kim 2024) | `kim2024healthllm` | 7/10 | 2.2, 5 |
| Mental-LLM (Xu 2024) | `xu2024mentalllm` | 5/10 | 2.2 |
| MindScape (Nepal 2024) | `nepal2024mindscape` | 4/10 | 2.2 |
| Meegahapola (2023) | `meegahapola2023generalization` | 4/10 | 2.1 |
| GLOSS (Choube 2025) | `choube2025gloss` | 4/10 | 2.2 |
| Feng et al. (2026) | `feng2026comparative` | 4/10 | 2.1, 2.2, 5, 6 |
| PHIA (2026) | `phia2026` | 1/10 | 2.2 |
| ContextAgent (2025) | `yang2025contextagent` | 1/10 | 2.3, 7 |
| Voyager (2023) | `wang2023voyager` | 1/10 | 2.3, 3 |
| BSP (Narayanan 2013) | `narayanan2013bsp` | 1/10 | 6.1 |
| Penetrative AI (Xu 2024) | `xu2024penetrative` | 1/10 | 1, 6.1 |
| Shiffman EMA (2008) | `shiffman2008ema` | 1/10 | 1, 4, 6 |
| Spring cancer (2019) | `spring2019cancersurvivors` | 1/10 | 6.6 |
| Adler Beyond Detection (2024) | `adler2024beyond` | 2/10 | 2.1, 6.7 |
| Vital Insight (Li 2025) | `li2025vitalinsight` | 1/10 | 2.2 |
| LENS (Xu 2025) | `xu2025lens` | 3/10 | 2.2, 7 |
| IoT-LLM (An 2024) | `an2024iotllm` | 1/10 | 2.2 |
| Wang SLM (2024) | `wang2024slm` | 1/10 | 2.2, 7 |
| Toolformer (Schick 2023) | `schick2023toolformer` | 0/10 | 2.3 |
| MCP (Anthropic 2024) | `anthropic2024mcp` | 0/10 | 2.3, 3, 4 |
| Agent survey (Wang 2024) | `wang2024agentsurvey` | 0/10 | 2.3 |
| Nahum-Shani (2014) | `nahumshani2014jitai` | 1/10 | 2.4 |
| Klasnja MRT (2015) | `klasnja2015mrt` | 1/10 | 2.4, 6 |
| Klasnja HeartSteps (2019) | `klasnja2019heartsteps` | 1/10 | 6.6 |
| Mattingly (2025) | `mattingly2025predictors` | 1/10 | 1, 3 |
| Adler ML (2022) | `adler2022machine` | 1/10 | 2.1, 6 |
| Wang tracking (2018) | `wang2018tracking` | 1/10 | 2.1 |

---

## Total unique references: 36

### By topic area:
- **Passive sensing for mental health**: 7 papers
- **LLMs for health/affective computing**: 11 papers
- **Predecessor/same-dataset work**: 2 papers
- **JITAI and receptivity**: 6 papers
- **Agentic AI and tool use**: 6 papers
- **Behavioral signal processing**: 1 paper
- **Penetrative AI**: 1 paper
- **EMA methodology**: 1 paper
- **Cancer survivorship**: 1 paper

### Gap analysis papers (not mentioned by any expert, added by citation researcher):
- `schick2023toolformer` -- Toolformer (seminal tool-use work, NeurIPS 2023 oral)
- `anthropic2024mcp` -- MCP specification (the actual protocol PULSE uses)
- `wang2024agentsurvey` -- comprehensive LLM agent survey (contextualizes agentic AI field)
