# Project Progress — Proactive Affective Agent (BUCS Pilot)

**Last updated:** 2026-02-25 (session 5)
**Status:** V3 done (best so far), V4 checkpoints empty (needs re-run after bug fix), all Titan baselines complete

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

## 🔴 PENDING ACTION (2026-02-25)

### V4 Pilot Re-run Required

V4 checkpoint files exist but all predictions are **empty `{}`** — caused by the parallel tool-use bug that was fixed in session 4. Must delete old checkpoints and re-run.

**Decision needed:** V3 was run with `claude-haiku-4-5-20251001`. User wants Sonnet for all future experiments. Options:
1. Re-run V4 only with Sonnet (model mismatch with V3)
2. Re-run both V3 + V4 with Sonnet (fair comparison, but V3 takes ~6h)

**Command to re-run V4:**
```bash
# First, delete empty V4 checkpoints
rm outputs/pilot/checkpoints/v4_user*_checkpoint.json

# Then re-run V4 with Sonnet
PYTHONPATH=. python3 scripts/run_pilot.py \
    --version v4 \
    --users 71,119,164,310,458 \
    --model sonnet \
    --delay 1.0 \
    --verbose \
    > outputs/pilot/logs/v4_sonnet_rerun.log 2>&1 &
```

### All Titan Baselines Complete (verified 2026-02-25)

Server is healthy: load avg 1.68, RAM 16GB/125GB used.

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

**Ridge note:** Even with RidgeCV (alphas=[0.1, 1, 10, 100, 1000]) inside StandardScaler pipeline, 3/5 folds still diverge catastrophically (MAE ~1e12–1e14). Only folds 2 (MAE=9.5) and 3 (MAE=10.8) are reasonable. Root cause: extreme feature collinearity + outlier targets in some folds. Ridge is not reliable for this dataset — **exclude Ridge from final paper results.**

### Baseline: DL MLP (Sensing features) — Folds 1-4 done, Fold 5 diverged

| Fold | MAE | BA | F1 | Notes |
|------|-----|----|----|-------|
| 1 | 5.082 | 0.506 | 0.437 | ✅ |
| 2 | 4.695 | 0.509 | 0.448 | ✅ |
| 3 | 4.511 | 0.510 | 0.440 | ✅ |
| 4 | 4.506 | 0.501 | 0.436 | ✅ |
| 5 | ~1e12 | 0.511 | 0.459 | ❌ diverged even after rerun with gradient clipping |

**4-fold mean (folds 1-4 only):** MAE=4.699, BA=0.507, F1=0.440

### Baseline: Combined (Sensor + Diary embeddings, 5-fold CV) — DONE

| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| RF | 3.935 | 0.620 | 0.568 |
| Logistic | — | 0.615 | 0.575 |
| Ridge | 87.5B (diverged) | — | — |

**Key: Combined RF (sensing + diary embeddings) gets BA=0.620 — best traditional ML result.**

### Pilot: CALLM, V1, V2, V3 — All complete (5 users)

| Version | Mean MAE | Mean BA | Mean F1 | N entries | Notes |
|---------|----------|---------|---------|-----------|-------|
| CALLM | 1.167 | 0.645 | 0.478 | 427 | Diary + TF-IDF RAG |
| V1 | 6.977 | 0.539 | 0.316 | 427 | Sensing structured — mean regression |
| V2 | 7.062 | 0.531 | 0.284 | 427 | Sensing autonomous — mean regression |
| **V3** | **0.866** | **0.674** | **0.514** | **1306** | **Diary+sensing structured — BEST** |
| V4 | — | — | — | 0 (empty) | Bug caused empty predictions; needs re-run |

**Key insights:**
- V3 (structured multimodal) is the best system so far: MAE=0.866, BA=0.674 (beats AR baseline 0.658!)
- V1/V2 (sensing-only) produce mean-regressing predictions (pred std≈2.5 vs GT std≈8.0)
- V3 has well-calibrated predictions: pred μ=15.8±6.6 vs GT μ=14.8±6.8
- V4 needs re-run with fixed parallel tool-use bug

---

## ❌ Pending Tasks

### HIGHEST PRIORITY

**1. Re-run V4 pilot with fixed code**
V4 checkpoints are empty due to parallel tool-use bug (now fixed). Must delete + re-run.
Decision: use Sonnet (user mandate) or Haiku (match V3)? See "PENDING ACTION" section above.

**2. Consider re-running V3 with Sonnet**
V3 was run with Haiku. For fair V3 vs V4 comparison, both should use same model.
If V3 re-run is too costly, document the model difference in the paper.

### MEDIUM PRIORITY

**3. Paper-ready results table**
Once V4 is done, compile final comparison across all systems.

**4. Investigate V3 behavior in detail**
- V3 beats AR baseline (BA=0.674 vs 0.658) — first system to do so!
- V3 predictions are well-calibrated (pred std matches GT std)
- What specific sensing+diary patterns drive V3's accuracy?

**5. Integration test: passed 50/50**
10 EMA entries × 5 versions, all PASSED. Logs in `test_logs/run_20260225_120453/`.

---

## Results Summary Table (2026-02-25)

| System | Method | Mean MAE↓ | Mean BA↑ | Mean F1↑ | Status |
|--------|---------|-----------|----------|----------|--------|
| AR last_value | Autocorrelation | 2.758 | 0.658 | 0.617 | ✅ Done |
| AR rolling_mean | Autocorrelation | 2.552 | 0.658 | 0.617 | ✅ Done |
| Text TF-IDF | Diary text only | 3.999 | 0.613 | 0.570 | ✅ Done |
| Text BoW | Diary text only | 4.043 | 0.607 | 0.561 | ✅ Done |
| Transformer MiniLM | Diary text only | 3.898 | 0.629 | 0.588 | ✅ Done |
| ML RF | Sensing features | 5.923 | 0.501 | 0.365 | ✅ Done |
| ML XGBoost | Sensing features | 9.374 | 0.502 | 0.391 | ✅ Done |
| ML Ridge | Sensing features | DIVERGED | — | — | ❌ Exclude |
| ML Logistic | Sensing features | — | 0.500 | 0.302 | ✅ Done |
| DL MLP | Sensing features | 4.699* | 0.507* | 0.440* | ✅ Done (4-fold) |
| Combined RF | Sensor + diary | 3.935 | 0.620 | 0.568 | ✅ Done |
| Combined Logistic | Sensor + diary | — | 0.615 | 0.575 | ✅ Done |
| **CALLM** | Diary + TF-IDF RAG | 1.167 | 0.645 | 0.478 | ✅ Done (427 entries) |
| **V1** | Sensing structured | 6.977 | 0.539 | 0.316 | ✅ Done (427 entries) |
| **V2** | Sensing autonomous | 7.062 | 0.531 | 0.284 | ✅ Done (427 entries) |
| **V3** | Diary+sensing structured | **0.866** | **0.674** | **0.514** | ✅ Done (1306 entries, Haiku) |
| **V4** | Diary+sensing autonomous | — | — | — | ❌ Re-run needed (bug fix applied) |

\* = 4-fold mean only, fold 5 diverged

**Ranking by Mean BA:** V3 (0.674) > AR (0.658) > CALLM (0.645) > Combined RF (0.620) > MiniLM (0.629) > Text TF-IDF (0.613) > V1 (0.539) > V2 (0.531) > ML/DL (~0.50)

---

## Bug Fixes Applied (Sessions 3-5)

1. **V1 prompt missing OUTPUT_FORMAT** (`src/think/prompts.py`): Added `{OUTPUT_FORMAT}` schema at end of `v1_prompt()`. Existing V1 results were with old broken prompt.

2. **TFIDFRetriever min_df=2 crash on small corpus** (`src/remember/retriever.py`): Added try/except fallback to min_df=1 for corpora with < 2 docs.

3. **SelectKBest k > n_features** (`src/baselines/ml_pipeline.py`): Changed fixed k_vals to use `"all"` string for max K, avoiding post-VarianceThreshold dimensionality mismatch.

4. **MLP gradient explosion** (`src/baselines/deep_learning_baselines.py`): Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

5. **Sigmoid overflow** (`src/baselines/deep_learning_baselines.py`): Added `logits_clipped = np.clip(logits, -500.0, 500.0)` before sigmoid.

6. **Ridge regression divergence** (`src/baselines/ml_pipeline.py`): Replaced `Ridge(alpha=1.0)` with `RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])`.

7. **GPS/accelerometer data confirmed**: Both have data (gps: 12,926 rows, accel: 11,134 rows). PROGRESS.md corrected.

8. **V2/V4 parallel tool-use bug** (session 4): Anthropic API returns multiple `tool_use` blocks in one response. Breaking mid-batch left unmatched `tool_result` blocks → API 400 error. Fix: process ALL tool_use blocks before checking max_tool_calls limit.

9. **V2/V4 duplicate assistant message** (session 4): After tool loop exits via tool_use processing, assistant content was already appended. Fix: only append when `stop_reason == "end_turn"`.

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
7. **Model for future experiments:** `claude-sonnet-4-6` (Sonnet). V3 pilot was run with Haiku; V4+ must use Sonnet.

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
