# Project Progress — Proactive Affective Agent (BUCS Pilot)

**Last updated:** 2026-03-17
**Status:** 50 users have checkpoints; 18 users form the primary evaluation set (V2∩V4∩V5∩V6 clean complete); 9 users fully complete across all 7 versions.

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

### Corrupted V1 Checkpoints (separate issue — exclude from evaluation)

Three V1 checkpoints have corrupted entry counts and must be excluded:
- `v1_user61`: 1651 entries (corrupted)
- `v1_user86`: 1317 entries (corrupted)
- `v1_user99`: 1569 entries (corrupted)

---

## Completion Status

### Primary Evaluation Set (18 users — complete for V2/V4/V5/V6)

Users: **24, 43, 71, 119, 164, 232, 242, 258, 275, 310, 338, 362, 399, 403, 437, 458, 505, 513**

These 18 users have clean, complete checkpoints for V2, V4, V5, and V6. Other versions have partial coverage (see per-version counts below).

### Per-Version Clean Complete User Counts

| Version | Clean Complete Users | Notes |
|---------|---------------------|-------|
| CALLM | 23 | |
| V1 | 22 | 3 corrupted excluded: user61 (1651), user86 (1317), user99 (1569) |
| V2 | 22 | |
| **V3** | **17** | **Bottleneck** — fewest users |
| V4 | 24 | |
| V5 | 25 | |
| V6 | 24 | |

### Consistent Set — All 7 Versions Complete (9 users)

**[24, 43, 71, 258, 275, 310, 338, 403, 458]**

These 9 users have clean, complete checkpoints across all 7 versions. Useful for fully within-subject analyses.

---

## Current Evaluation Results (2026-03-17, 18-user primary set)

Evaluated across 18 primary users where possible (per-version user counts vary).

| Version | Users | Entries | Mean BA ↑ | Mean F1 ↑ | Notes |
|---------|-------|---------|----------|----------|-------|
| CALLM | 13/18 | 1137 | 0.626 | 0.618 | Diary + TF-IDF RAG |
| V1 | 13/18 | 1131 | 0.521 | 0.453 | Sensing structured |
| V2 | 18/18 | 1567 | 0.598 | 0.591 | Sensing agentic |
| V3 | 10/18 | 862 | 0.607 | 0.590 | Multimodal structured |
| V4 | 18/18 | 1567 | 0.666 | 0.661 | Multimodal agentic |
| V5 | 18/18 | 1567 | 0.601 | 0.596 | Sensing agentic (filtered) |
| **V6** | **18/18** | **1567** | **0.669** | **0.664** | Multimodal agentic (filtered) — **BEST BA** |

**Key insights:**
- V6 (0.669) and V4 (0.666) are best — multimodal agentic works
- Agentic > Structured: V4 > V3, V2 > V1
- Multimodal > Sensing-only across the board
- V3 only has 10/18 users — bottleneck, needs more runs
- CALLM and V1 also short (13/18 each)

Full results saved to `outputs/pilot_v2/evaluation.json`.

---

## PENDING ACTION — Priority Order

### P0: Complete Remaining Tasks for 18 Primary Users

Fill in missing version coverage for the 18-user primary set:
- **CALLM**: 5 more users needed (13 → 18)
- **V1**: 5 more users needed (13 → 18)
- **V3**: 8 more users needed (10 → 18) — biggest gap

```bash
cd /Users/zwang/Documents/proactive-affective-agent
python3 scripts/queue_runner.py --dry-run   # verify pending tasks
python3 scripts/queue_runner.py --workers 5  # execute
```

**Note**: Sonnet weekly quota may need to reset first. Check quota before running.

### P1: Fix Corrupted V1 Checkpoints (3 users)

Users 61, 86, 99 have corrupted V1 checkpoints (inflated entry counts: 1651, 1317, 1569). These need to be deleted and rerun.

### P2: Expand queue_runner USERS List to All 50 Users

The current `USERS` list in `queue_runner.py` only has 15 users. Update it to include all 50 users with checkpoints.

### P3: Re-evaluate After Completion

```bash
PYTHONPATH=. python3 scripts/evaluate_pilot.py
```

### P4: Paper-Ready Results Table

Once all users are done with consistent model:
- Final comparison table across all systems + baselines
- Per-target breakdown
- Statistical significance tests

---

## Baseline Results (Complete)

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
- `USERS = [43, 258, 338, 399, 403, 275, 513, 362, 71, 437]` — **only 15 users; needs expansion to all 50**
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
