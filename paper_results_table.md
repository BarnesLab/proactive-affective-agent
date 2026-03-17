# Paper Results Tables — Proactive Affective Agent (BUCS Study)

Generated: 2026-03-17
Status: 18-user primary evaluation set (V2∩V4∩V5∩V6 clean complete). CALLM/V1 have 13/18; V3 has 10/18.

---

## Table 1: Main Results — Aggregate Metrics

All baselines use 5-fold across-subject CV on full dataset (~15,984 EMA entries, 399 users).
LLM agents are evaluated on the 18-user primary set (V2/V4/V5/V6 = 18 users, CALLM/V1 = 13, V3 = 10).
All LLM agent results use claude-sonnet.

| System | Input Modality | Method | Users | Entries | Mean BA ↑ | Mean F1 ↑ | Notes |
|--------|---------------|--------|-------|---------|----------|----------|-------|
| **Traditional ML Baselines (Sensing)** |||||||||
| RF | Sensing | 5-fold CV | 399 | ~15,984 | 0.501 | 0.365 | |
| XGBoost | Sensing | 5-fold CV | 399 | ~15,984 | 0.502 | 0.391 | |
| Logistic | Sensing | 5-fold CV | 399 | ~15,984 | 0.500 | 0.302 | Classifier only |
| MLP | Sensing | 4-fold CV† | 399 | ~12,800 | 0.507 | 0.440 | †Fold 5 diverged |
| **Text Baselines (Diary)** |||||||||
| TF-IDF + SVM | Diary text | 5-fold CV | 399 | ~15,984 | 0.613 | 0.570 | |
| BoW + SVM | Diary text | 5-fold CV | 399 | ~15,984 | 0.607 | 0.561 | |
| MiniLM | Diary embeddings | 5-fold CV | 399 | ~15,984 | 0.629 | 0.588 | |
| **Combined Baselines (Sensing + Diary)** |||||||||
| Combined RF | Sensing + diary | 5-fold CV | 399 | ~15,984 | 0.620 | 0.568 | |
| Combined Logistic | Sensing + diary | 5-fold CV | 399 | ~15,984 | 0.615 | 0.575 | |
| **LLM Agent Systems (N=18 primary set)** |||||||||
| CALLM | Diary + TF-IDF RAG | Pilot (13/18) | 13 | 1,137 | 0.626 | 0.618 | |
| V1 (Structured) | Sensing only | Pilot (13/18) | 13 | 1,131 | 0.521 | 0.453 | |
| V2 (Agentic) | Sensing only | Pilot (18/18) | 18 | 1,567 | 0.598 | 0.591 | |
| V3 (Structured) | Sensing + diary | Pilot (10/18) | 10 | 862 | 0.607 | 0.590 | |
| **V4 (Agentic)** | **Sensing + diary** | **Pilot (18/18)** | **18** | **1,567** | **0.666** | **0.661** | |
| V5 (Agentic+filtered) | Sensing only | Pilot (18/18) | 18 | 1,567 | 0.601 | 0.596 | |
| **V6 (Agentic+filtered)** | **Sensing + diary** | **Pilot (18/18)** | **18** | **1,567** | **0.669** | **0.664** | **BEST BA** |

**Ranking by Mean BA:** V6 (0.669) > V4 (0.666) > MiniLM (0.629) > CALLM (0.626) > Combined RF (0.620) > TF-IDF (0.613) > V3 (0.607) > V5 (0.601) > V2 (0.598) > V1 (0.521) > MLP (0.507) > ML (~0.50)

---

## Table 2: Per-Target Binary Metrics — Balanced Accuracy (Key Targets)

LLM agent systems on respective user subsets; baselines on full 5-fold CV.

| Target | CALLM | V1 | V2 | V3 | V4 | V5 | V6 |
|--------|-------|-----|-----|-----|-----|-----|-----|
| happy | .723 | .532 | .620 | .646 | .727 | .631 | **.735** |
| PA_State | .544 | .520 | .607 | .572 | .719 | .604 | **.731** |
| NA_State | .666 | .521 | .616 | .686 | .750 | .637 | **.760** |
| sad | .650 | .513 | .603 | .657 | .681 | .601 | **.688** |
| worried | .677 | .547 | .605 | .659 | .696 | .611 | **.712** |
| INT_avail | .560 | .513 | **.738** | .530 | .733 | .712 | .733 |
| ER_desire | .644 | .518 | .660 | .683 | **.749** | .649 | **.749** |
| **Mean BA** | .626 | .521 | .598 | .607 | .666 | .601 | **.669** |

Bold = best in row among LLM agent systems.

---

## Table 3: Hypotheses Evaluation

| Hypothesis | Comparison | Result | Evidence |
|------------|-----------|--------|----------|
| **H1**: Sensing data enhances diary-based prediction | V4 vs CALLM, V6 vs CALLM | ✅ **Supported** | V4 BA=0.666, V6 BA=0.669 > CALLM BA=0.626 |
| **H2**: LLM agents outperform traditional ML baselines | V4/V6 vs ML baselines | ✅ **Supported** (multimodal) | V6 BA=0.669 > MiniLM BA=0.629, Combined RF BA=0.620 |
| **H3**: Passive sensing alone enables meaningful prediction | V2/V5 vs random | ⚠️ **Partially supported** | V2 BA=0.598, V5 BA=0.601 — above chance (0.50) but below multimodal |
| **H4**: Agentic > structured | V2 vs V1, V4 vs V3 | ✅ **Supported** | V4 BA=0.666 > V3 BA=0.607; V2 BA=0.598 > V1 BA=0.521 |
| **H5**: Diary text adds value beyond sensing | V4 vs V2, V6 vs V5 | ✅ **Supported** | V4 BA=0.666 > V2 BA=0.598; V6 BA=0.669 > V5 BA=0.601 |

---

## Notes

1. **Evaluation set**: Primary set = 18 users with clean complete V2/V4/V5/V6 results. CALLM and V1 have 13/18 users; V3 has only 10/18 (bottleneck due to earlier model contamination, now resolved for most users).
2. **All LLM agent systems** use claude-sonnet for inference.
3. **Baseline–agent comparison caveat**: Baselines use 5-fold CV on full dataset (399 users); LLM agents use 13–18 user subsets. Direct comparison requires caution, though agent results on fewer users still outperform baselines on full data.
4. **V5/V6 (filtered)**: Agentic variants with additional data quality filtering applied to sensing features.
5. **Ridge excluded**: 3/5 folds diverged catastrophically. Not suitable for this dataset.
6. **MLP 4-fold**: Fold 5 diverged even with gradient clipping. Reported as 4-fold mean.
7. **V6 is the best system** by Mean BA (0.669) and Mean F1 (0.664), marginally ahead of V4 (0.666 / 0.661). Both substantially outperform all non-LLM baselines.
