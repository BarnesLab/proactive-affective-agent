# Expert Panel for PULSE (IMWUT)

## Scoring: IMWUT Journal — Accept / Minor Revision / Major Revision / Reject

## Expert Roster (10 researchers)

### 1. Tanzeem Choudhury (Cornell Tech)
- **Themes**: Mobile sensing for mental health, behavioral health interventions, closing the sensing-to-intervention loop
- **Perspective**: ACM Fellow (2021), pioneer in mobile health sensing. Co-founded HealthRhythms/Dapple. Advocates for translating passive sensing into real-world clinical interventions. Would push hard on clinical translational potential.
- **Relevance**: The godmother of mobile sensing for mental health — this paper's entire framing of passive sensing → affect prediction → intervention fits her research agenda.

### 2. Xuhai "Orson" Xu (University of Washington → SEA Lab)
- **Themes**: GLOBEM benchmark for behavioral modeling generalization, computational well-being ecosystems, passive sensing for affect
- **Perspective**: Created the most rigorous benchmark for cross-dataset generalization in behavioral sensing. Co-authored Health-LLM. Would scrutinize evaluation methodology, generalizability, and sensing feature engineering.
- **Relevance**: GLOBEM is the benchmark for sensing-based prediction; Health-LLM is the closest prior work on LLMs + sensor data. This paper must position against both.

### 3. Inbal Nahum-Shani (University of Michigan)
- **Themes**: JITAIs, micro-randomized trials, adaptive interventions, health behavior optimization
- **Perspective**: Co-inventor of the JITAI framework and MRT methodology. Views intervention timing and receptivity as first-class research problems. Would evaluate whether this paper's prediction targets (ER_desire, INT_availability) are properly defined per JITAI theory.
- **Relevance**: This paper's focus on predicting intervention opportunities (desire + availability) directly addresses JITAI decision points. She defines the theory.

### 4. Andrew Campbell (Dartmouth College)
- **Themes**: StudentLife, mobile sensing for student mental health, Mindscape (generative AI + sensing)
- **Perspective**: ACM UbiComp 10-Year Impact Award (2024) for StudentLife. Recently launched Mindscape, the first generative AI app embedding smartphone behavioral sensing. Would evaluate sensing data quality, longitudinal validity, and practical deployment feasibility.
- **Relevance**: StudentLife pioneered smartphone sensing for mental health; Mindscape is the most directly comparable system combining generative AI + behavioral sensing.

### 5. Shrikanth Narayanan (USC)
- **Themes**: Behavioral signal processing, multimodal behavior analysis, affective computing, computational media intelligence
- **Perspective**: NAE member (2026), University Professor. Approaches affective computing from signal processing fundamentals. Would scrutinize whether the agentic approach has principled advantages over feature engineering, and evaluate multimodal fusion rigor.
- **Relevance**: Senior figure in multimodal affective computing — his behavioral signal processing lens provides a complementary (potentially skeptical) view of LLM-based approaches to what is traditionally a signal processing problem.

### 6. Tim Althoff (UW → Stanford Biomedical Data Science)
- **Themes**: Computational health, large-scale behavioral data, empathic AI conversations for mental health, digital phenotyping
- **Perspective**: SIGKDD Dissertation Award, NSF CAREER. Works with massive behavioral datasets. Would evaluate statistical methodology, sample size adequacy, and whether findings generalize beyond the study population.
- **Relevance**: His work on large-scale behavioral data analysis and AI for mental health support provides a data science perspective on whether N=50 is sufficient and whether the evaluation methodology is sound.

### 7. Varun Mishra (Northeastern University)
- **Themes**: Receptivity to mHealth interventions, passive sensing, JITAI delivery, UbiWell Lab
- **Perspective**: PhD from Dartmouth (Campbell lab). Published on detecting receptivity in natural environments. Co-author of GLOSS (multi-agent LLM for passive sensing). Would evaluate receptivity prediction and compare to his GLOSS approach.
- **Relevance**: GLOSS is the most directly comparable system (multi-agent LLM + passive sensing, IMWUT 2025). His receptivity detection work directly relates to this paper's INT_availability prediction.

### 8. Laura Barnes (University of Virginia)
- **Themes**: Mobile sensing for health, affective computing, social anxiety detection, machine learning for chronic disease
- **Perspective**: Associate Director of UVA Link Lab. Has applied sensing + ML to cancer, anxiety, depression, TBI. Understands the clinical deployment challenges and IRB/data considerations unique to cancer survivor populations.
- **Relevance**: As UVA faculty, she knows the BUCS study ecosystem. Her work on sensing for chronic disease populations provides clinical grounding for the paper's cancer survivorship application.

### 9. Bonnie Spring (Northwestern University)
- **Themes**: mHealth behavior change, cancer survivorship interventions, JITAI/MRT optimization, multiple health behavior change
- **Perspective**: Distinguished Scientist Award (SBM 2021). First to succeed using mobile tech for multiple simultaneous behavior changes. Works with cancer survivors on physical activity interventions. Would evaluate clinical significance and real-world applicability for cancer care.
- **Relevance**: Brings the clinical/behavioral science perspective that the IMWUT reviewers may lack. Ensures the paper's claims about intervention opportunities are clinically meaningful, not just statistically significant.

### 10. Yubin Kim (MIT Media Lab)
- **Themes**: Health-LLM, LLMs for health prediction from wearable data, contextual health reasoning
- **Perspective**: Lead author of Health-LLM (PMLR 2024), which evaluated 12 LLMs on 10 health prediction tasks using wearable sensor data. Would compare PULSE's agentic approach to Health-LLM's prompting/fine-tuning paradigm.
- **Relevance**: Health-LLM is the primary prior work on using LLMs to reason about sensor data for health prediction. This paper must clearly differentiate its agentic investigation approach from Health-LLM's fixed prompting.

---

## Panel Coverage Matrix

| Dimension | Experts |
|-----------|---------|
| Mobile/passive sensing | Choudhury, Xu, Campbell, Mishra, Barnes |
| Affective computing | Narayanan, Barnes, Xu |
| LLM agents / tool-use | Mishra (GLOSS), Kim (Health-LLM), Campbell (Mindscape) |
| JITAI / intervention theory | Nahum-Shani, Spring, Mishra |
| Cancer / clinical populations | Spring, Barnes |
| Evaluation methodology / data science | Althoff, Xu, Nahum-Shani |
| Writing strength / high-impact papers | Choudhury, Narayanan, Althoff, Nahum-Shani |
| Senior / field-shapers | Choudhury (ACM Fellow), Narayanan (NAE), Nahum-Shani, Spring |
| Mid-career / innovators | Xu, Althoff, Mishra, Kim, Das Swain |
