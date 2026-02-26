# Paper Results Tables — Proactive Affective Agent (BUCS Pilot)

Generated: 2026-02-26
Status: V4 pilot in progress (user310 + user458 running). Tables will be updated when V4 completes.

---

## Table 1: Main Results — Aggregate Metrics

All baselines use 5-fold across-subject CV on full dataset (~15,984 EMA entries, 399 users).
LLM agents are evaluated on a 5-user pilot subset (user71, user119, user164, user310, user458).

| System | Input Modality | Method | Mean MAE ↓ | Mean BA ↑ | Mean F1 ↑ | N | Notes |
|--------|---------------|--------|-----------|----------|----------|--:|-------|
| **Traditional ML Baselines (Sensing)** ||||||||
| RF | Sensing | 5-fold CV | 5.923 | 0.501 | 0.365 | ~15,984 | |
| XGBoost | Sensing | 5-fold CV | 9.374 | 0.502 | 0.391 | ~15,984 | |
| Logistic | Sensing | 5-fold CV | — | 0.500 | 0.302 | ~15,984 | Classifier only |
| MLP | Sensing | 4-fold CV† | 4.699 | 0.507 | 0.440 | ~12,800 | †Fold 5 diverged |
| **Text Baselines (Diary)** ||||||||
| TF-IDF + SVM | Diary text | 5-fold CV | 3.999 | 0.613 | 0.570 | ~15,984 | |
| BoW + SVM | Diary text | 5-fold CV | 4.043 | 0.607 | 0.561 | ~15,984 | |
| MiniLM | Diary embeddings | 5-fold CV | 3.898 | 0.629 | 0.588 | ~15,984 | |
| **Combined Baselines (Sensing + Diary)** ||||||||
| Combined RF | Sensing + diary | 5-fold CV | 3.935 | 0.620 | 0.568 | ~15,984 | |
| Combined Logistic | Sensing + diary | 5-fold CV | — | 0.615 | 0.575 | ~15,984 | |
| **Autocorrelation Baselines** ||||||||
| AR last_value | Prior EMA | 5-fold CV | 2.758 | 0.658 | 0.617 | ~15,984 | |
| AR rolling_mean | Prior EMA | 5-fold CV | 2.552 | 0.658 | 0.617 | ~15,984 | |
| **LLM Agent Systems (Pilot)** ||||||||
| CALLM | Diary + TF-IDF RAG | Pilot | 1.167 | 0.645 | 0.478 | 427 | Structured output |
| V1 (Structured) | Sensing only | Pilot | 6.977 | 0.539 | 0.316 | 427 | Mean regression |
| V2 (Agentic) | Sensing only | Pilot | 7.062 | 0.531 | 0.284 | 427 | Mean regression |
| **V3 (Structured)** | **Sensing + diary** | **Pilot** | **0.866** | **0.674** | **0.514** | **335** | **Best system** |
| V4 (Agentic) | Sensing + diary | Pilot | — | — | — | — | In progress |

**Ranking by Mean BA:** V3 (0.674) > AR (0.658) > CALLM (0.645) > MiniLM (0.629) > Combined RF (0.620) > TF-IDF (0.613) > V1 (0.539) > V2 (0.531) > MLP (0.507) > ML (~0.50)

---

## Table 2: Per-Target Continuous Metrics (MAE)

Prediction targets: PANAS_Pos (0–30), PANAS_Neg (0–30), ER_desire (0–10).

| System | PANAS_Pos MAE | PANAS_Neg MAE | ER_desire MAE | Mean MAE |
|--------|--------------|--------------|---------------|----------|
| **V3** | **1.313** | **1.045** | **0.239** | **0.866** |
| CALLM | 1.850 | 1.288 | 0.363 | 1.167 |
| AR rolling_mean | 3.820 | 2.272 | 1.574 | 2.552 |
| AR last_value | 4.227 | 2.410 | 1.646 | 2.758 |
| MiniLM | 6.019 | 3.492 | 2.185 | 3.898 |
| Combined RF | — | — | — | 3.935 |
| TF-IDF | 6.164 | 3.581 | 2.252 | 3.999 |
| BoW | 6.208 | 3.633 | 2.289 | 4.043 |
| XGBoost | 6.682 | 3.887 | 2.550 | 4.373 |
| MLP | — | — | — | 4.699 |
| RF | 6.798 | 7.884 | 3.250 | 5.978 |
| V1 | 8.016 | 9.794 | 3.119 | 6.977 |
| V2 | 8.834 | 9.131 | 3.222 | 7.062 |

---

## Table 3: Per-Target Binary Metrics — Balanced Accuracy

16 binary targets. LLM pilot systems on 5-user subset; baselines on full 5-fold CV.

| Target | V3 | CALLM | AR | MiniLM | TF-IDF | BoW | Comb. RF | V1 | V2 |
|--------|-----|-------|-----|--------|--------|-----|----------|----|----|
| PA_State | .550 | .606 | .614 | .689 | .666 | .652 | — | .533 | .523 |
| NA_State | .595 | .610 | .627 | .680 | .652 | .645 | — | .549 | .546 |
| happy | **.794** | .709 | .628 | .672 | .651 | .642 | — | .547 | .551 |
| sad | .611 | .639 | .651 | .639 | .612 | .612 | — | .620 | .577 |
| afraid | .646 | .556 | .707 | .587 | .579 | .574 | — | .585 | .581 |
| miserable | .606 | .558 | .698 | .598 | .578 | .578 | — | .494 | .476 |
| worried | **.792** | .686 | .641 | .657 | .643 | .634 | — | .562 | .553 |
| cheerful | **.849** | .711 | .623 | .678 | .652 | .648 | — | .549 | .549 |
| pleased | **.781** | .732 | .618 | .674 | .650 | .639 | — | .557 | .554 |
| grateful | **.766** | .711 | .640 | .623 | .613 | .604 | — | .562 | .569 |
| lonely | .469 | .516 | **.717** | .558 | .567 | .556 | — | .435 | .403 |
| int._quality | **.672** | .622 | .627 | .616 | .602 | .602 | — | .522 | .498 |
| pain | **.686** | .542 | **.691** | .545 | .546 | .539 | — | .507 | .493 |
| forecasting | .373 | .465 | **.631** | .597 | .580 | .582 | — | .508 | .515 |
| ER_desire | .750 | **.751** | **.731** | .671 | .649 | .638 | — | .615 | .612 |
| INT_avail. | .846 | **.908** | — | .573 | .571 | .570 | — | .486 | .498 |
| **Mean BA** | **.674** | .645 | .658 | .629 | .613 | .607 | .620 | .539 | .531 |

Bold = best in row (among primary systems).

---

## Table 4: Per-Target Binary Metrics — F1 Score

| Target | V3 | CALLM | AR | MiniLM | TF-IDF | BoW | V1 | V2 |
|--------|-----|-------|-----|--------|--------|-----|----|----|
| PA_State | .359 | .471 | **.657** | **.709** | .683 | .663 | .228 | .200 |
| NA_State | .348 | .367 | .497 | **.580** | .548 | .542 | .277 | .274 |
| happy | **.750** | .668 | .665 | .693 | .667 | .653 | .295 | .241 |
| sad | .390 | .433 | .525 | **.528** | .499 | .501 | .395 | .323 |
| afraid | .421 | .228 | **.575** | .431 | .425 | .425 | .333 | .330 |
| miserable | .353 | .217 | **.560** | .443 | .421 | .426 | .025 | .000 |
| worried | .593 | .537 | .545 | .576 | .560 | .545 | .365 | .362 |
| cheerful | **.808** | .671 | .661 | .696 | .666 | .657 | .297 | .231 |
| pleased | **.727** | .696 | .656 | .694 | .660 | .647 | .323 | .254 |
| grateful | **.708** | .675 | .698 | .662 | .648 | .637 | .417 | .381 |
| lonely | .000 | .097 | **.623** | .450 | .464 | .456 | .186 | .143 |
| int._quality | **.667** | .530 | **.672** | .642 | .621 | .615 | .288 | .182 |
| pain | .522 | .212 | **.621** | .461 | .467 | .451 | .178 | .144 |
| forecasting | .049 | .243 | **.682** | .637 | .619 | .611 | .263 | .271 |
| ER_desire | **.667** | **.667** | .589 | .581 | .560 | .548 | .418 | .419 |
| INT_avail. | .868 | **.928** | — | .624 | .615 | .599 | .765 | .783 |
| **Mean F1** | .514 | .478 | **.615** | .588 | .570 | .561 | .316 | .284 |

---

## Table 5: Prediction Calibration (Pilot Systems)

How well-calibrated are the predicted distributions vs ground truth?

| System | PANAS_Pos pred μ±σ | PANAS_Pos GT μ±σ | PANAS_Neg pred μ±σ | PANAS_Neg GT μ±σ | ER_desire pred μ±σ | ER_desire GT μ±σ |
|--------|-------------------|-----------------|-------------------|-----------------|-------------------|-----------------|
| **V3** | 15.8±6.6 | 14.8±6.8 | 2.4±3.1 | 1.4±2.7 | 2.1±2.5 | 1.9±2.6 |
| CALLM | 18.5±7.0 | 18.8±8.3 | 2.8±3.8 | 1.7±3.3 | 1.6±2.3 | 1.5±2.5 |
| V1 | 16.2±3.4 | 18.8±8.3 | 11.2±4.5 | 1.7±3.3 | 4.0±1.3 | 1.5±2.5 |
| V2 | 15.0±3.8 | 18.8±8.3 | 10.5±4.2 | 1.7±3.3 | 4.1±1.4 | 1.5±2.5 |

Key: V3 predictions are well-calibrated (σ_pred ≈ σ_GT). V1/V2 show severe mean regression (σ_pred << σ_GT) and systematically overpredict PANAS_Neg.

---

## Table 6: Hypotheses Evaluation

| Hypothesis | Comparison | Result | Evidence |
|------------|-----------|--------|----------|
| **H1**: Sensing data enhances diary-based prediction | V3 vs CALLM | ✅ **Supported** | V3 BA=0.674 > CALLM BA=0.645 |
| **H2**: LLM agents outperform traditional ML baselines | V1/V2 vs ML baselines | ❌ **Not supported** (sensing-only) | V1 BA=0.539, V2 BA=0.531 vs RF BA=0.501 — marginal improvement |
| **H3**: Passive sensing alone enables meaningful prediction | V1/V2 vs random | ⚠️ **Weakly supported** | BA ~0.53 is above 0.50 but far below AR (0.658) |
| **H4**: Agentic > structured | V2 vs V1, V4 vs V3 | ❓ **Pending V4** | V2 BA=0.531 < V1 BA=0.539 (opposite direction) |

---

## Notes

1. **Evaluation set mismatch**: Baselines use 5-fold CV on full dataset; LLM agents use 5-user pilot. Direct comparison requires caution.
2. **V3 model**: Run with claude-haiku-4-5-20251001. All other LLM agents use claude-sonnet-4-6. For fair comparison, V3 may need re-running with Sonnet.
3. **V3 n_entries**: 1306 total EMA entries processed, but only 335 yielded parseable continuous predictions. Binary metrics are computed on the 335 parseable entries.
4. **Ridge excluded**: 3/5 folds diverged catastrophically (MAE ~1e12). Not suitable for this dataset.
5. **MLP 4-fold**: Fold 5 diverged even with gradient clipping. Reported as 4-fold mean.
6. **AR BA identical for both variants**: Binary predictions (carry forward) are the same; only continuous MAE differs.
7. **V4**: Running as of 2026-02-26. Results pending for user310 (81 entries) and user458 (82 entries). User71/119/164 already completed.
