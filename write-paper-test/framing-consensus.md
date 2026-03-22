# Framing Consensus — PULSE (IMWUT)

## Title
**PULSE: Agentic LLM Investigation of Passive Sensing for Proactive Affect Prediction and Intervention Opportunity Detection in Cancer Survivors**

## Core Story (2 sentences)
Cancer survivors who most need mental health support are least likely to self-report — the diary paradox. PULSE introduces agentic sensing investigation, where LLM agents autonomously query passive smartphone sensing data through purpose-built tools, and a 2×2 factorial evaluation on 50 cancer survivors proves that agentic reasoning — not just data modality — is the key driver of prediction accuracy for emotional states and intervention receptivity.

## Contribution Framing (ordered)
1. **Paradigm**: Agentic sensing investigation — LLM agents with tools autonomously investigate behavioral data, replacing fixed feature pipelines
2. **Evaluation methodology**: 2×2 factorial design (structured/agentic × sensing/multimodal) cleanly isolating the agentic effect
3. **Clinical application**: Predicting intervention receptivity (ER_desire + INT_availability) in cancer survivors from passive sensing alone
4. **System**: 8 MCP sensing query tools, cross-user RAG for calibration, per-user session memory

## Narrative Arc
1. **Problem**: Cancer survivors need proactive mental health support. Current JITAIs depend on active self-report (diary/EMA). The diary paradox: people who most need help are least likely to respond.
2. **Opportunity**: Passive smartphone sensing captures behavioral signals continuously. But traditional ML on fixed features achieves only ~0.52 BA — barely above chance.
3. **Insight**: What if the LLM could investigate sensing data autonomously — choosing what to examine, how far back to look, which comparisons to make — like a clinician reviewing a patient's behavioral chart?
4. **Approach**: PULSE gives LLM agents 8 sensing query tools. 2×2 factorial: {Structured, Agentic} × {Sensing-only, Multimodal}. 7 versions total including CALLM baseline and filtered variants.
5. **Results**: Agentic >> Structured (0.660 vs 0.603 multimodal, 0.589 vs 0.516 sensing). INT_availability best predicted by sensing (Auto-Sense 0.706 >> CALLM 0.542). Sensing-only viable without diary.
6. **Impact**: Paradigm shift from feature engineering to agentic investigation. Practical implications for JITAI design: separate desire from availability; use sensing for the behavioral, diary for the emotional.

## Key Claims (supported)
- Agentic >> Structured on identical data (p < 10^-10, r > 0.9)
- Multimodal >> Sensing-only across both architectures
- LLM agents >> ML baselines (with caveat: different evaluation setups)
- Sensing-only prediction is viable (Auto-Sense 0.589 >> ML ~0.52)
- INT_availability is fundamentally behavioral (sensing outperforms diary)
- The diary paradox motivates proactive sensing-first prediction

## Do NOT Overclaim
- Real-time deployment readiness (retrospective only)
- Generalizability beyond cancer survivors
- Fair ML baseline comparison (different N)
- Filtering adds meaningful value (0.661 ≈ 0.660)
- Cost efficiency or scalability
- Memory system improves performance (no ablation)
- CALLM is "surpassed" — PULSE extends it to proactive

## Emphasis Strategy
- **Lead with**: 2×2 factorial results, agentic vs. structured comparison
- **Highlight**: INT_avail as behavioral construct, diary paradox, agent traces
- **Include but contextualize**: ML baselines (reference points, not head-to-head)
- **De-emphasize**: Filtering variants, absolute BA numbers in isolation
- **Show**: Qualitative investigation traces (essential for contribution clarity)
- **Report**: Effect sizes over p-values, per-user distributions, representativeness in main text

## Lead Author Election
Based on framing alignment: **Tanzeem Choudhury** — her "closing the sensing-to-intervention loop" framing most naturally matches the agreed narrative. Mobile sensing pioneer with the strongest voice for IMWUT audience. Paradigm-forward but clinically grounded.

## Target Venue Profile
- IMWUT: 20-25 pages, strong system contribution + thorough evaluation
- Scoring: Accept / Minor Revision / Major Revision / Reject
- Audience: Mobile sensing, ubicomp, HCI researchers
