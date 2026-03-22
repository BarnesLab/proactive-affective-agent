# Project Digest for Paper Writing — PULSE

## What the system/method does
PULSE (Proactive Affective Agent) deploys LLM agents to predict cancer survivors' emotional states and intervention opportunities from passive smartphone sensing data and daily diary text. Instead of feeding pre-formatted features to a model, the agents autonomously investigate behavioral data streams via purpose-built tools (MCP sensing query tools), deciding what data to examine, how far back to look, and which cross-user comparisons to make. The system is evaluated through a 2×2 factorial design (structured vs. autonomously-agentic × sensing-only vs. multimodal) on 50 cancer survivors from the BUCS study.

## Key technical contributions
- **Agentic sensing investigation**: LLM agents autonomously query 8 sensing modalities via MCP tools (8 tools: daily summary, behavioral timeline, targeted hourly query, raw events, baseline comparison, receptivity history, similar days, peer cases) — agent decides investigation strategy per-prediction
- **2×2 factorial design** isolating agentic reasoning benefit: Struct-Sense, Auto-Sense, Struct-Multi, Auto-Multi, plus filtered variants (Auto-Sense+, Auto-Multi+) and CALLM baseline = 7 total versions
- **Cross-user RAG for calibration**: retrieves similar cases from population (text-based or sensing-based) to ground predictions in empirical evidence, not just LLM intuition
- **Session memory**: per-user longitudinal memory with self-reflections after each prediction; references receptivity signals only (no ground truth leakage)
- **Cost-free LLM inference** via Claude Max subscription using CLI (`claude -p` for structured, `claude --print` + MCP for agentic)

## Main results and findings (N=50 users, ~3,900 entries per version)

### Aggregate performance (mean BA across 16 binary targets)
| System | Input | Mean BA | Mean F1 |
|--------|-------|---------|---------|
| Auto-Multi+ (v6) | Multimodal filtered | **0.661** | **0.656** |
| Auto-Multi (v4) | Multimodal | **0.660** | **0.657** |
| CALLM | Diary + RAG | 0.611 | 0.594 |
| Struct-Multi (v3) | Multimodal | 0.603 | 0.584 |
| Auto-Sense+ (v5) | Sensing filtered | 0.591 | 0.584 |
| Auto-Sense (v2) | Sensing only | 0.589 | 0.583 |
| Struct-Sense (v1) | Sensing only | 0.516 | 0.428 |

### Focus targets (4 key clinical constructs)
| Target | CALLM | Struct-Sense | Auto-Sense | Struct-Multi | Auto-Multi | Auto-Sense+ | Auto-Multi+ |
|--------|-------|-------------|------------|-------------|------------|-------------|-------------|
| ER_desire | 0.632 | 0.508 | 0.652 | 0.664 | 0.745 | 0.653 | **0.751** |
| INT_avail | 0.542 | 0.533 | 0.706 | 0.551 | **0.716** | 0.685 | 0.707 |
| PA_State | 0.534 | 0.505 | 0.598 | 0.533 | 0.724 | 0.595 | **0.733** |
| NA_State | 0.641 | 0.510 | 0.592 | 0.674 | 0.716 | 0.601 | **0.722** |

### ML baselines (5-fold CV on full 399 users, sensing features)
| Model | Mean BA | Mean F1 |
|-------|---------|---------|
| RF | 0.518 | 0.403 |
| XGBoost | 0.514 | 0.457 |
| Logistic | 0.515 | 0.478 |

### Key findings
1. **Agentic >> Structured**: Auto-Multi (0.660) >> Struct-Multi (0.603); Auto-Sense (0.589) >> Struct-Sense (0.516)
2. **Multimodal >> Sensing-only**: Auto-Multi (0.660) >> Auto-Sense (0.589)
3. **LLM agents >> ML baselines**: Auto-Multi+ (0.661) >> best ML (RF 0.518) on sensing
4. **INT_avail is behavioral**: Auto-Sense (0.706) >> CALLM (0.542) — sensing outperforms diary for availability
5. **Diary paradox**: diary is most informative but absent when needed most; sensing-only viable (0.589)
6. **Filtering marginal**: Auto-Multi+ (0.661) ≈ Auto-Multi (0.660); data quality filtering doesn't add much when agent can already handle noise

### Statistical significance
- Wilcoxon signed-rank tests with bootstrap CIs available in statistical_tests.json
- Per-user BA distributions available for all version pairs

### Representativeness (50 vs 418)
- PA_State, ER_desire, INT_avail base rates: p > 0.05 (no significant difference)
- NA_State: p = 0.028 (small effect size r = 0.19, slightly lower NA in pilot)
- EMA count: significantly higher in pilot (82 vs 34 — by design, selected high-compliance users)
- Platform: slightly more Android in pilot (36% vs 25%, p = 0.027)

## Available evidence
- `outputs/pilot_v2/evaluation.json` — all metrics for 7 versions (50 users)
- `outputs/pilot_v2/statistical_tests.json` — per-user BA, Wilcoxon, bootstrap CIs
- `outputs/pilot_v2/representativeness.json` — 50 vs 418 user comparison
- `outputs/pilot_v2/*_predictions.csv` — raw predictions for all 7 versions
- `outputs/ml_baselines/ml_baseline_metrics.json` — RF, XGBoost, Logistic, Ridge baselines
- `paper_results_table.md` — pre-formatted results tables (older N=18 version, use evaluation.json for current)
- `src/` — full source code (agent architecture, MCP tools, prompts, evaluation)

## Target venue: IMWUT
- Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies
- Journal format (not conference): Accept / Minor Revision / Major Revision / Reject
- Typical paper: 20-30 pages, strong system contribution + evaluation
- Acceptance rate ~20-25%

## Existing draft status: none (blind write)
No existing draft provided. Writing from scratch based on project materials.

## User-provided notes
- ML baseline numbers are from 5-fold CV on all 399 users (not the same 50). Treat as placeholder — user will update values later. Write confidently with current numbers.
- The system name is PULSE (short) / Proactive Affective Agent (full).
- Dataset: BUCS (Building Up Cancer Survivors), 418 participants, ~5-week study.
- 8 passive sensing modalities: motion, GPS, screen, keyboard, app usage (Android), light (Android), music, sleep.
- Focus targets: ER_desire (emotion regulation desire), INT_availability (intervention availability), PA_State (positive affect state), NA_State (negative affect state).
