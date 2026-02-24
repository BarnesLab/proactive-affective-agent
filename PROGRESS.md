# Project Progress — Proactive Affective Agent (BUCS Pilot)

**Last updated:** 2026-02-24 (session 3 end — hand-off to session 4)
**Status:** V3 pilot running locally (~10:30 PM done), Titan jobs running, next agent picks up

---

## ⚠️ New Session Onboarding

If starting fresh, read in this order:
1. `CLAUDE.md` (repo root) — project brief
2. This file (`PROGRESS.md`) — exact current state + pending tasks
3. `~/.claude/projects/-Users-zwang/memory/MEMORY.md` — cross-session context

Key commands to verify state:
```bash
# What's running locally?
ps aux | grep python | grep -v grep

# V3 pilot log tail
tail -5 outputs/pilot/logs/v3_all_users.log

# V4 chain watcher alive?
ps aux | grep run_v4_after_v3 | grep -v grep

# Checkpoint files
ls -la outputs/pilot/checkpoints/

# Titan status
ssh zhiyuan@172.29.39.82 'ps aux | grep python | grep -v grep | wc -l'
```

---

## Dataset

| Item | Value |
|------|-------|
| Study | BUCS (cancer survivorship EMA + passive sensing) |
| Participants | 399 users |
| EMA entries | ~15,984 total (5-fold test sets combined) |
| CV strategy | 5-fold **across-subject** (participant-level, zero overlap) |
| Splits location | `data/processed/splits/group_{1-5}_{train,test}.csv` |
| Hourly Parquet | `data/processed/hourly/{screen,motion,keyinput,light,mus}/` |

**Sensing data coverage:**
- `screen`: 407 users
- `motion`: 371 users
- `keyinput`: 280 users
- `light`: 111 users
- `mus` (music): 91 users
- `accelerometer`: 11,134 rows (`val_sleep_duration_min`)
- `gps`: 12,926 rows (`travel_km`, `home_minutes`, `location_variance`, etc.)

**Prediction targets:**
- Continuous (3): `PANAS_Pos` (0–30), `PANAS_Neg` (0–30), `ER_desire` (0–10)
- Binary (15): `Individual_level_{PA,NA,happy,sad,afraid,miserable,worried,cheerful,pleased,grateful,lonely,interactions_quality,pain,forecasting,ER_desire}_State`
- `INT_availability` (yes/no)

---

## System Architecture

```
Pilot versions:
  CALLM  — Diary + TF-IDF RAG, structured output
  V1     — Sensing only, structured 5-step prompt
  V2     — Sensing only, autonomous (no fixed steps)
  V3     — Diary + sensing + multimodal RAG, structured output  ← RUNNING NOW
  V4     — Diary + sensing + multimodal RAG, autonomous         ← AUTO-STARTS AFTER V3

Scripts:
  scripts/run_pilot.py              — Run CALLM/V1/V2/V3/V4 pilot simulation
  scripts/evaluate_pilot.py         — Compute MAE/BA/F1 from checkpoint files
  scripts/run_ml_baselines.py       — Traditional ML (RF/XGBoost/Ridge/Logistic)
  scripts/run_dl_baselines.py       — DL (MLP, Transformer, Combined, Text)
  scripts/run_ar_baseline.py        — AR autocorrelation baseline
  scripts/merge_baseline_results.py — Merge fold-level results into one JSON
```

---

## 🔴 CURRENTLY RUNNING (2026-02-24 ~16:10)

### Local Machine

| Process | PID | Status | ETA |
|---------|-----|--------|-----|
| V3 pilot (all 5 users) | 71841 | Entry 17/93 user71 (~16s/entry after API) | ~22:30 tonight |
| V4 chain watcher | 72549 | Sleeping, polls every 30s for V3 to finish | After V3 |

**V3 checkpoint files:** `outputs/pilot/checkpoints/v3_user{71,119,164,310,458}_checkpoint.json`

**V3 command that was used:**
```bash
PYTHONPATH=. python3 scripts/run_pilot.py \
    --version v3 \
    --users 71,119,164,310,458 \
    --model claude-haiku-4-5-20251001 \
    --delay 1.0 \
    --verbose \
    > outputs/pilot/logs/v3_all_users.log 2>&1 &
```

**V4 chain watcher:** `/tmp/run_v4_after_v3.sh` — polls PID 71841, then runs V4 with same args.
If watcher died: start V4 manually after V3 checkpoint files are complete (all 5 users).

**V4 manual start command:**
```bash
PYTHONPATH=. python3 scripts/run_pilot.py \
    --version v4 \
    --users 71,119,164,310,458 \
    --model claude-haiku-4-5-20251001 \
    --delay 1.0 \
    --verbose \
    > outputs/pilot/logs/v4_all_users.log 2>&1 &
```

### Titan Server (zhiyuan@172.29.39.82)

| Process | PID | Status | Output |
|---------|-----|--------|--------|
| ML Ridge re-run (5 folds parallel) | ~1341616+ | Running, all 5 folds started at 16:01 | `outputs/ml_baselines_ridge_v2/fold_N/` |
| Combined baseline (all 5 folds) | 1212188 | Fold 3/5 started at ~16:07 | `outputs/advanced_baselines/combined/` |
| DL MLP fold 5 re-run | (bg) | Started 16:05, fixing gradient explosion | `outputs/advanced_baselines_dl_fold5_rerun/` |

**Check commands:**
```bash
# Ridge folds done?
ssh zhiyuan@172.29.39.82 'for f in 1 2 3 4 5; do echo -n "Ridge fold_$f: "; cat ~/proactive-affective-agent/outputs/ml_baselines_ridge_v2/fold_$f/ml_baseline_summary.md 2>/dev/null | grep "Mean MAE" || echo "running"; done'

# Combined done?
ssh zhiyuan@172.29.39.82 'cat ~/proactive-affective-agent/outputs/advanced_baselines/combined/combined_baseline_summary.md 2>/dev/null'

# DL fold 5 rerun done?
ssh zhiyuan@172.29.39.82 'cat ~/proactive-affective-agent/outputs/advanced_baselines_dl_fold5_rerun/dl/fold_5/dl_baseline_summary.md 2>/dev/null'
```

---

## ✅ Completed Results

### Baseline: AR Autocorrelation

| Variant | Mean MAE | Mean BA | Mean F1 |
|---------|----------|---------|---------|
| last_value (AR1) | 2.758 | 0.658 | 0.617 |
| rolling_mean_w3 | 2.552 | 0.658 | 0.617 |

**Role:** Empirical ceiling for autocorrelation-only prediction. Sensing agents must approach BA≈0.658.

### Baseline: Text (TF-IDF / BoW on diary)

| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| TF-IDF | 3.999 | 0.613 | 0.570 |
| BoW | 4.043 | 0.607 | 0.561 |

### Baseline: Transformer (MiniLM on diary)

| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| MiniLM | 3.898 | 0.629 | 0.588 |

### Baseline: Traditional ML (Sensing features, 5-fold CV) — RF/XGBoost/Logistic DONE

All 5 folds completed. **Ridge diverged → being re-run with RidgeCV fix on Titan.**

| Model | Mean MAE | Mean BA | Mean F1 | Notes |
|-------|----------|---------|---------|-------|
| RF | 5.923 | 0.501 | 0.365 | 5-fold avg |
| XGBoost | 9.374 | 0.502 | 0.391 | 5-fold avg |
| Logistic | — | 0.500 | 0.302 | Classification only |
| Ridge | DIVERGED | — | — | Re-running with RidgeCV |

Per-fold RF results:

| Fold | MAE | BA | F1 |
|------|-----|----|----|
| 1 | 5.689 | 0.502 | 0.340 |
| 2 | 5.616 | 0.501 | 0.380 |
| 3 | 5.894 | 0.500 | 0.379 |
| 4 | 5.830 | 0.499 | 0.358 |
| 5 | 6.588 | 0.501 | 0.369 |

**Key insight: BA ≈ 0.50 = essentially random. Sensing features alone are not predictive.**

### Baseline: DL MLP (Sensing features) — Folds 1-4 done, Fold 5 being re-run

| Fold | MAE | BA | F1 | Notes |
|------|-----|----|----|-------|
| 1 | 5.082 | 0.506 | 0.437 | ✅ |
| 2 | 4.695 | 0.509 | 0.448 | ✅ |
| 3 | 4.511 | 0.510 | 0.440 | ✅ |
| 4 | 4.506 | 0.501 | 0.436 | ✅ |
| 5 | ~1e12 | 0.448 | — | ❌ gradient explosion → rerunning |

4-fold mean (excluding fold 5): MAE=4.699, BA=0.507, F1=0.440

### Pilot: CALLM, V1, V2 — All 5 users complete (427 entries each)

| Version | PANAS_Pos MAE | happy_State BA | Mean BA | Notes |
|---------|--------------|----------------|---------|-------|
| CALLM | 1.850 | 0.709 | 0.645 | Diary + TF-IDF RAG |
| V1 | 8.016 | 0.547 | 0.539 | Sensing structured — mean regression |
| V2 | 8.834 | 0.551 | 0.531 | Sensing autonomous — mean regression |

**Key insight:** V1/V2 (sensing-only) produce mean-regressing predictions (pred std≈2.5 vs GT std≈8.0).
CALLM diary→RAG is dramatically better. V3/V4 (diary+sensing) should bridge this gap.

---

## ❌ Pending Tasks (for Next Session)

### HIGHEST PRIORITY — After jobs complete tonight

**1. Evaluate V3 + V4 pilot results**
```bash
cd /Users/zwang/Documents/proactive-affective-agent
PYTHONPATH=. python3 scripts/evaluate_pilot.py
```
This reads `outputs/pilot/checkpoints/{callm,v1,v2,v3,v4}_user{71,...}_checkpoint.json` and prints comparison table. V3 finishes ~22:30, V4 after that.

**2. Merge ML Ridge v2 results (after all 5 folds done)**
```bash
# Check if done:
ssh zhiyuan@172.29.39.82 'ls ~/proactive-affective-agent/outputs/ml_baselines_ridge_v2/fold_{1,2,3,4,5}/*.json 2>/dev/null | wc -l'

# Merge (if merge script supports per-model output dir):
ssh zhiyuan@172.29.39.82 'cd ~/proactive-affective-agent && PYTHONPATH=. python scripts/merge_baseline_results.py --output outputs/ml_baselines_ridge_v2 --type ml'
```

**3. Get Combined baseline results (after done)**
```bash
ssh zhiyuan@172.29.39.82 'cat ~/proactive-affective-agent/outputs/advanced_baselines/combined/combined_baseline_summary.md'
```

**4. Get DL fold 5 result (after rerun done)**
```bash
ssh zhiyuan@172.29.39.82 'cat ~/proactive-affective-agent/outputs/advanced_baselines_dl_fold5_rerun/dl/fold_5/dl_baseline_summary.md'
```

**5. Update PROGRESS.md results table with final numbers**

### MEDIUM PRIORITY

**6. Run full evaluation: CALLM vs V1-V4 comparison table**
Once V3+V4 complete, run `scripts/evaluate_pilot.py` and update the results summary table.

**7. Investigate V3/V4 behavior**
- Does V3 (structured diary+sensing) match CALLM performance?
- Does V4 (autonomous) equal or beat V3?
- Key metric: PANAS_Pos MAE and mean BA vs AR baseline (0.658)

**8. Paper-ready results table**
Once all baselines + V3/V4 pilot are done, compile final comparison:
- AR baseline (ceiling)
- Text / Transformer (diary text only)
- ML Sensing (RF, Ridge, XGB, Logistic)
- DL Sensing (MLP)
- Combined (sensor + diary)
- CALLM / V1 / V2 / V3 / V4

---

## Results Summary Table (Current State — Incomplete)

| System | Method | PANAS_Pos MAE↓ | happy_State BA↑ | Mean BA | Status |
|--------|---------|----------------|-----------------|---------|--------|
| AR last_value | Autocorrelation | 2.758 | 0.658 | 0.658 | ✅ Done |
| AR rolling_mean | Autocorrelation | 2.552 | 0.658 | 0.658 | ✅ Done |
| Text TF-IDF | Diary text only | 3.999 | 0.613 | 0.613 | ✅ Done |
| Text BoW | Diary text only | 4.043 | 0.607 | 0.607 | ✅ Done |
| Transformer MiniLM | Diary text only | 3.898 | 0.629 | 0.629 | ✅ Done |
| ML RF | Sensing features | 5.923 | ~0.501 | 0.501 | ✅ Done |
| ML XGBoost | Sensing features | 9.374 | ~0.502 | 0.502 | ✅ Done |
| ML Ridge | Sensing features | DIVERGED | — | — | ⏳ RidgeCV fix running |
| ML Logistic | Sensing features | — | — | 0.500 | ✅ Done |
| DL MLP | Sensing features | ~4.7* | ~0.507* | 0.507* | ⏳ Fold 5 rerunning |
| Combined | Sensor + diary | pending | pending | pending | ⏳ Running |
| **CALLM** | Diary + TF-IDF RAG | **1.850** | **0.709** | **0.645** | ✅ Done (427 entries) |
| **V1** | Sensing structured | 8.016 | 0.547 | 0.539 | ✅ Done (427 entries) |
| **V2** | Sensing autonomous | 8.834 | 0.551 | 0.531 | ✅ Done (427 entries) |
| **V3** | Diary+sensing structured | running | running | running | ⏳ ~22:30 tonight |
| **V4** | Diary+sensing autonomous | pending | pending | pending | ⏳ After V3 |

\* = 4-fold mean only, fold 5 diverged

---

## Bug Fixes Applied (This Session)

1. **V1 prompt missing OUTPUT_FORMAT** (`src/think/prompts.py`): Added `{OUTPUT_FORMAT}` schema at end of `v1_prompt()`. Existing V1 results were with old broken prompt.

2. **TFIDFRetriever min_df=2 crash on small corpus** (`src/remember/retriever.py`): Added try/except fallback to min_df=1 for corpora with < 2 docs.

3. **SelectKBest k > n_features** (`src/baselines/ml_pipeline.py`): Changed fixed k_vals to use `"all"` string for max K, avoiding post-VarianceThreshold dimensionality mismatch.

4. **MLP gradient explosion** (`src/baselines/deep_learning_baselines.py`): Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

5. **Sigmoid overflow** (`src/baselines/deep_learning_baselines.py`): Added `logits_clipped = np.clip(logits, -500.0, 500.0)` before sigmoid.

6. **Ridge regression divergence** (`src/baselines/ml_pipeline.py`): Replaced `Ridge(alpha=1.0)` with `RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])`.

7. **GPS/accelerometer data confirmed**: Both have data (gps: 12,926 rows, accel: 11,134 rows). PROGRESS.md corrected.

---

## Test Suite: 171/171 Passing

```bash
# Run all tests
PYTHONPATH=. python3 -m pytest tests/ -v

# Quick smoke test
PYTHONPATH=. python3 -m pytest tests/ -x -q
```

All tests use dry-run mode for LLM calls — no API tokens consumed.

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/sense/query_tools.py` | SensingQueryEngine — core data access |
| `src/data/hourly_features.py` | HourlyFeatureLoader — Parquet features for ML/DL |
| `src/baselines/ml_pipeline.py` | ML baseline (RF/XGBoost/Ridge/Logistic + SelectKBest) |
| `src/baselines/deep_learning_baselines.py` | MLP + Transformer + Combined + Text |
| `src/agent/structured_full.py` | V3 agent implementation |
| `src/agent/autonomous_full.py` | V4 agent implementation |
| `src/think/prompts.py` | All LLM prompts for CALLM/V1/V2/V3/V4 |
| `scripts/run_pilot.py` | Main pilot runner |
| `scripts/evaluate_pilot.py` | Compute MAE/BA/F1 from checkpoint files |
| `scripts/run_ml_baselines.py` | ML baseline runner (per-fold) |
| `scripts/run_dl_baselines.py` | DL/Text/Transformer/Combined runner |
| `data/processed/hourly/` | Hourly Parquet files |
| `data/processed/splits/` | 5-fold CV splits |
| `outputs/pilot/checkpoints/` | Per-user prediction checkpoints |

---

## Key Design Decisions (Frozen)

1. **ER_desire binary threshold:** `>= 5` (scale midpoint), NOT person mean
2. **Oracle mode:** Removed — giving agent historical PA/NA = time series forecasting, not sensing-based prediction
3. **Session memory content:** Only receptivity signals (diary, raw ER_desire, INT_availability). Never prediction targets.
4. **5-fold CV:** Across-subject (participant-level). All observations from a user in exactly one fold.
5. **AR baseline role:** Empirical ceiling for autocorrelation-only prediction.
6. **Feature selection:** SelectKBest with K as hyperparameter (25/50/75/100% fractions), 3-fold inner CV.
7. **V3/V4 model:** `claude-haiku-4-5-20251001` (cost-efficient, ~10s/call)

---

## Server Setup: Titan (zhiyuan@172.29.39.82)

```bash
# Code sync (local → server)
rsync -avz \
    --exclude="data/bucs-data" \
    --exclude="__pycache__" --exclude="*.pyc" \
    --exclude=".git" --exclude="outputs" --exclude="*.egg-info" \
    /Users/zwang/Documents/proactive-affective-agent/ \
    zhiyuan@172.29.39.82:~/proactive-affective-agent/

# Env
ssh zhiyuan@172.29.39.82
source ~/anaconda3/etc/profile.d/conda.sh && conda activate efficient-ser

# GPU: use GPU1 (RTX A6000, 49GB) for new jobs
export CUDA_VISIBLE_DEVICES=1
```
