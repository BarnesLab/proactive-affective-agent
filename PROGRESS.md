# Project Progress — Proactive Affective Agent (BUCS Pilot)

**Last updated:** 2026-03-12
**Status:** 11/14 users fully complete across 7 versions; 11 tasks remaining for 3 users. V3 model contamination in pilot/ (mixed sonnet+haiku) needs resolution.

---

## New Session Onboarding

If starting fresh, read in this order:
1. `CLAUDE.md` (repo root) — project brief + conventions
2. This file (`PROGRESS.md`) — exact current state + pending tasks
3. `~/.claude/projects/-Users-zwang/memory/MEMORY.md` — cross-session context

Key commands:
```bash
# What's running?
ps aux | grep python | grep -v grep

# Checkpoint status (quick)
ls outputs/pilot_v2/checkpoints/ | wc -l
ls outputs/pilot/checkpoints/ | wc -l

# Run evaluation
PYTHONPATH=. python3 scripts/evaluate_pilot.py

# Resume incomplete tasks (5 workers)
python3 scripts/queue_runner.py --dry-run   # preview
python3 scripts/queue_runner.py             # execute
```

---

## Architecture: 7 Versions (2x2 + 2 + 1)

| Version | Data Sources | Strategy | Description |
|---------|-------------|----------|-------------|
| **CALLM** | Diary + TF-IDF RAG | Structured | Baseline, no sensing |
| **V1** | Sensing only | Structured | Fixed 5-step pipeline |
| **V2** | Sensing only | Agentic | Autonomous tool-use |
| **V3** | Diary + sensing + RAG | Structured | Full multimodal, fixed pipeline |
| **V4** | Diary + sensing + RAG | Agentic | Full multimodal, autonomous |
| **V5** | Sensing only (filtered) | Agentic | V2 + data quality filtering |
| **V6** | Diary + sensing + RAG (filtered) | Agentic | V4 + data quality filtering |

All versions run via `scripts/run_pilot.py` with `--model sonnet`. Queue runner at `scripts/queue_runner.py` manages parallel execution (5 workers).

---

## Two Output Directories

| Directory | Users | Model | Notes |
|-----------|-------|-------|-------|
| `outputs/pilot_v2/` | 43, 71, 258, 275, 338, 362, 399, 403, 437, 513 | **ALL Sonnet** | Current primary. queue_runner hardcodes `--model sonnet` |
| `outputs/pilot/` | 71, 119, 164, 310, 458 (original 5) | **V3 mixed, rest sonnet** | Legacy. Only V3 has mixed sonnet+haiku. See below |

The evaluation script (`scripts/evaluate_pilot.py`) checks **pilot_v2 first**, then pilot. So pilot_v2 checkpoints always take precedence when both exist for the same user+version.

### Model Attribution in `outputs/pilot/` (from records.jsonl)

For the 4 pilot-only users (**119, 164, 310, 458**) — user 71 has pilot_v2 checkpoints that take priority.

**Authoritative source**: `records.jsonl` files contain per-entry `model` field. Log files may be misleading (haiku runs happened but didn't always overwrite sonnet checkpoints — runs RESUME from existing checkpoints).

| Version | Model in pilot/ | Evidence (records.jsonl) | Action Needed |
|---------|----------------|--------------------------|---------------|
| CALLM | **Sonnet** | All entries: `model=sonnet` | None |
| V1 | **Sonnet** | All entries: `model=sonnet` | None |
| V2 | **Sonnet** | All entries: `model=sonnet` | None |
| **V3** | **Mixed sonnet+haiku** | First ~50 entries: `model=sonnet`; remaining entries: `model=""` (haiku resume). Haiku run resumed from sonnet checkpoint, so predictions are from both models | **Rerun with sonnet** |
| V4 | **Sonnet** | All entries: `model=sonnet` | None |
| V5 | **Sonnet** | No records.jsonl (ran via queue_runner which uses sonnet) | None |
| V6 | **Sonnet** | No records.jsonl (ran via queue_runner which uses sonnet) | None |

**V3 contamination detail** (per user):
- user71: 52 sonnet + 41 haiku entries
- user119: 51 sonnet + 33 haiku entries
- user164: 50 sonnet + 37 haiku entries
- user310: 52 sonnet + 29 haiku entries
- user458: 49 sonnet + 33 haiku entries

**Impact**: Only V3 has model contamination. For 4 pilot-only users, V3 checkpoints contain ~60% sonnet + ~40% haiku predictions. All other versions are pure sonnet.

### Corrupted V1 Checkpoints in pilot/ (non-blocking)

These pilot/ V1 files have wildly inflated n_entries (bug from an earlier run):
- `v1_user275`: 880 (should be 89)
- `v1_user362`: 968 (should be 88)
- `v1_user437`: 1056 (should be 88)
- `v1_user513`: 791 (should be 90)

**Not a problem**: pilot_v2 has correct V1 checkpoints for all these users, and the evaluate script checks pilot_v2 first.

---

## Completion Status (per user × version)

### Fully Complete Users (11/14)

| User | Total | Source | All 7 versions done? |
|------|-------|--------|---------------------|
| 43 | 93 | pilot_v2 | Yes (sonnet) |
| 71 | 93 | pilot_v2 | Yes (sonnet) |
| 119 | 84 | pilot | Yes — V3 mixed sonnet+haiku, rest sonnet |
| 164 | 87 | pilot | Yes — V3 mixed sonnet+haiku, rest sonnet |
| 258 | 94 | pilot_v2 | Yes (sonnet) |
| 275 | 89 | pilot_v2 | Yes (sonnet) |
| 310 | 81 | pilot | Yes — V3 mixed sonnet+haiku, rest sonnet |
| 338 | 81 | pilot_v2 | Yes (sonnet) |
| 403 | 82 | pilot_v2 | Yes (sonnet) |
| 458 | 82 | pilot | Yes — V3 mixed sonnet+haiku, rest sonnet |
| 513 | 90 | pilot_v2 | Yes (sonnet) |

### Incomplete Users (3/14)

| User | Total | Incomplete Versions | Details |
|------|-------|-------------------|---------|
| **362** | 88 | v2 (59/88), v4 (11/88), v5 (0), v6 (0) | 4 tasks remaining |
| **399** | 96 | callm (21/96), v1 (18/96), v3 (48/96) | 3 tasks remaining |
| **437** | 88 | v2 (32/88), v4 (0), v5 (0), v6 (0) | 4 tasks remaining |

**Total remaining: 11 tasks** — all in pilot_v2 (sonnet).

---

## Current Evaluation Results (2026-03-12)

**WARNING**: V3 results include 4 users (119, 164, 310, 458) with mixed sonnet+haiku checkpoints from pilot/. All other versions are pure sonnet.

Evaluated across 11 complete users, 956 entries per version.

| Version | Mean MAE ↓ | Mean BA ↑ | Mean F1 ↑ | Notes |
|---------|-----------|----------|----------|-------|
| CALLM | 3.661 | 0.624 | 0.611 | Diary + TF-IDF RAG |
| V1 | 5.502 | 0.523 | 0.465 | Sensing structured |
| V2 | 4.881 | 0.601 | 0.594 | Sensing agentic |
| V3 | 3.947 | 0.638 | 0.632 | Multimodal structured |
| **V4** | **4.353** | **0.673** | **0.667** | Multimodal agentic |
| V5 | 4.878 | 0.602 | 0.596 | Sensing agentic (filtered) |
| **V6** | **4.220** | **0.675** | **0.669** | Multimodal agentic (filtered) — **BEST BA** |

AR baseline: Mean BA = 0.658 (autocorrelation ceiling)

**Key insights (preliminary):**
- V6 (0.675) and V4 (0.673) beat the AR baseline (0.658) — multimodal agentic works
- V3 (0.638) is close but doesn't beat AR in this larger evaluation
- Sensing-only versions (V1/V2/V5) underperform: BA 0.52-0.60
- CALLM (diary-only baseline) at 0.624 is competitive
- Filtering (V5 vs V2, V6 vs V4) makes marginal difference

Full results saved to `outputs/pilot_v2/evaluation.json`.

---

## PENDING ACTION — Priority Order

### P0: Complete 11 Remaining Tasks (3 users)

These are all in pilot_v2 using sonnet. Resume with queue_runner:

```bash
cd /Users/zwang/Documents/proactive-affective-agent
python3 scripts/queue_runner.py --dry-run   # verify 11 tasks
python3 scripts/queue_runner.py --workers 5  # execute
```

**Note**: Sonnet weekly quota may need to reset first. Check quota before running.

### P1: Resolve V3 Model Contamination (4 users × 1 version = 4 tasks)

Users 119, 164, 310, 458 have V3 checkpoints with mixed sonnet+haiku predictions. Options:

**Option A (recommended): Rerun V3 for these 4 users in pilot_v2/**
```bash
python3 scripts/run_pilot.py \
    --version v3 --users 119 164 310 458 --output-dir outputs/pilot_v2 --model sonnet
```
This creates fresh sonnet-only V3 checkpoints in pilot_v2/ which take evaluation priority.

**Option B: Accept mixed-model V3**
Document in paper that 4/11 users have mixed sonnet+haiku for V3 only. Minimal impact since V3 is not a top performer.

**Option C: Exclude these 4 users from V3 evaluation**
Evaluate V3 with only the 7 pure-sonnet users. Reduces V3 sample size but ensures model consistency.

### P2: Re-run Evaluation After P0/P1

```bash
PYTHONPATH=. python3 scripts/evaluate_pilot.py
```

### P3: Paper-Ready Results Table

Once all users are done with consistent model:
- Final comparison table across all systems + baselines
- Per-target breakdown
- Statistical significance tests

---

## Baseline Results (Complete)

### AR Autocorrelation (ceiling for autocorrelation-only prediction)
| Variant | Mean MAE | Mean BA | Mean F1 |
|---------|----------|---------|---------|
| last_value (AR1) | 2.758 | 0.658 | 0.617 |
| rolling_mean_w3 | 2.552 | 0.658 | 0.617 |

### Text Baselines (diary only)
| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| TF-IDF | 3.999 | 0.613 | 0.570 |
| BoW | 4.043 | 0.607 | 0.561 |
| MiniLM | 3.898 | 0.629 | 0.588 |

### Traditional ML (sensing features, 5-fold CV)
| Model | Mean MAE | Mean BA | Mean F1 | Notes |
|-------|----------|---------|---------|-------|
| RF | 5.923 | 0.501 | 0.365 | |
| XGBoost | 9.374 | 0.502 | 0.391 | |
| Logistic | — | 0.500 | 0.302 | Classification only |
| Ridge | DIVERGED | — | — | Exclude from paper |

### DL MLP (sensing features)
4-fold mean (fold 5 diverged): MAE=4.699, BA=0.507, F1=0.440

### Combined (sensor + diary embeddings)
| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| RF | 3.935 | 0.620 | 0.568 |
| Logistic | — | 0.615 | 0.575 |

---

## Directory Structure

### `outputs/` (clean — archived dirs in `outputs/_archive/`)

| Directory | Purpose |
|-----------|---------|
| `pilot/` | Original 5-user run (71, 119, 164, 310, 458) — legacy |
| `pilot_v2/` | Primary 10-user run — all sonnet, queue_runner output |
| `ar_baseline/` | Autocorrelation baseline results |
| `ml_baselines/` | Traditional ML + text + DL baseline results |
| `advanced_baselines/` | Combined sensor+diary ML baselines |
| `_archive/` | Old experiments: GPT smoke tests, codexmini, haiku, rerun artifacts |

### `scripts/` (core — archived scripts in `scripts/archive/`)

| Script | Purpose |
|--------|---------|
| `run_pilot.py` | Run any version for any users |
| `queue_runner.py` | Parallel queue (5 workers, auto-resume) |
| `clean_checkpoints.py` | Remove fallback entries from agentic checkpoints |
| `evaluate_pilot.py` | Compute MAE/BA/F1, outputs JSON + table |
| `run_ml_baselines_new.py` | Comprehensive ML/DL/text baselines |
| `run_ar_baseline.py` | AR autocorrelation baseline |
| `run_dl_baselines.py` | Deep learning baselines |
| `run_titan_baselines.py` | Titan server baselines |
| `build_filtered_data.py` | Build filtered datasets for V5/V6 |
| `select_pilot_users.py` | User selection logic |
| `generate_dashboard.py` | Progress dashboard |
| `integration_test.py` | End-to-end LLM integration tests |
| `sync_overleaf.py` | Overleaf paper sync |
| `offline/` | Data processing pipeline (sensor → Parquet) |

### Queue Runner Usage
```bash
# Preview what needs running
python3 scripts/queue_runner.py --dry-run

# Run with 5 workers (default)
python3 scripts/queue_runner.py

# Run with fewer workers (if rate limited)
python3 scripts/queue_runner.py --workers 3

# Clean fallbacks first, then run
python3 scripts/queue_runner.py --clean
```

**Important**: Queue runner sends Telegram notifications on start/finish. It auto-requeues failed tasks (except keyboard interrupts).

### queue_runner.py Configuration (current)
- `ALL_VERSIONS = ["CALLM", "v1", "v2", "v3", "v4", "v5", "v6"]`
- `USERS = [43, 258, 338, 399, 403, 275, 513, 362, 71, 437]`
- Model: hardcoded `--model sonnet` (line 128)
- Output: `outputs/pilot_v2/`

**To add users 119, 164, 310, 458** for V3 rerun, either:
- Edit `USERS` list in queue_runner.py and run, OR
- Run directly: `python3 scripts/run_pilot.py --version v3 --users 119 164 310 458 --output-dir outputs/pilot_v2 --model sonnet`

---

## Bug Fixes Applied (Sessions 3-7)

1. V1 prompt missing OUTPUT_FORMAT (session 3)
2. TFIDFRetriever min_df=2 crash on small corpus (session 3)
3. SelectKBest k > n_features (session 3)
4. MLP gradient explosion (session 4)
5. Sigmoid overflow (session 4)
6. Ridge regression divergence (session 4)
7. V2/V4 parallel tool-use bug — process ALL tool_use blocks (session 4)
8. V2/V4 duplicate assistant message (session 4)
9. evaluate_pilot.py extended to multi-directory + 11 users (session 7)

---

## Design Decisions (Frozen)

1. **ER_desire binary threshold**: >= 5 (scale midpoint)
2. **No oracle mode**: Historical PA/NA = time series forecasting, not sensing prediction
3. **Session memory**: Only receptivity signals (diary, ER_desire, INT_availability). Never targets.
4. **5-fold CV**: Across-subject. All observations from a user in one fold.
5. **AR baseline role**: Empirical ceiling for autocorrelation-only prediction
6. **Model**: claude-sonnet for all experiments (Haiku only during testing)
7. **V5/V6 added**: Filtered variants of V2/V4 with data quality checking

---

## Test Suite: 171/171 Passing

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
