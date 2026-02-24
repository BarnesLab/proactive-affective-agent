# Project Progress — Proactive Affective Agent (BUCS Pilot)

**Last updated:** 2026-02-24 (session 2)
**Status:** Active development — all baselines running in parallel on Titan, results pending

---

## ⚠️ Token Limit Warning

Claude Max weekly limit may be reached soon. If a new session is started:
1. The project's MEMORY.md at `~/.claude/projects/-Users-zwang/memory/MEMORY.md` has cross-session state
2. Read `CLAUDE.md` (this repo root) to restore context
3. Read this file (`PROGRESS.md`) for exact current state
4. Sessions **cannot** be resumed across account switches — start fresh with context from these files

Key commands to reorient a new session:
```bash
# Check what's running
ps aux | grep python | grep -v grep

# Check recent git commits
git log --oneline -10

# Quick sanity-check sensing pipeline
python3 -c "
import pandas as pd
from src.sense.query_tools import SensingQueryEngine
from pathlib import Path
dfs = [pd.read_csv(f'data/processed/splits/group_{i}_test.csv') for i in range(1,6)]
df = pd.concat(dfs); df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
eng = SensingQueryEngine(processed_dir=Path('data/processed/hourly'), ema_df=df)
print(eng.call_tool('get_daily_summary', {'date': '2023-11-20'}, study_id=71, ema_timestamp='2023-11-20 18:00:00'))
"
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
- `accelerometer`: 11,134 rows (sleep detection via `val_sleep_duration_min`)
- `gps`: 12,926 rows (`travel_km`, `home_minutes`, `location_variance`, etc.)

**Prediction targets:**
- Continuous (3): `PANAS_Pos` (0–30), `PANAS_Neg` (0–30), `ER_desire` (0–10)
- Binary (15): `Individual_level_{PA,NA,happy,sad,afraid,miserable,worried,cheerful,pleased,grateful,lonely,interactions_quality,pain,forecasting,ER_desire}_State`
- `INT_availability` (yes/no)

**Binary threshold for `ER_desire_State`:** `ER_desire >= 5` (scale midpoint), **not** person mean.
Rationale: distribution is bimodal (45.9% are 0, mean=2.37, median=1), midpoint is the natural breakpoint.

---

## System Architecture

```
run_agentic_pilot.py
    ├── CC backend:  ClaudeCodeAgent (src/agent/cc_agent.py)
    │                └── claude --print subprocess + MCP server (src/sense/mcp_server.py)
    └── API backend: AgenticSensingAgent (src/agent/agentic_sensing.py)
                     └── Anthropic SDK tool-use loop + SensingQueryEngine
```

**SensingQueryEngine** (`src/sense/query_tools.py`):
- Reads hourly Parquet files from `data/processed/hourly/`
- Tools: `get_daily_summary`, `query_sensing`, `query_raw_events`, `compare_to_baseline`, `get_receptivity_history`, `find_similar_days`
- Column name fix applied (2026-02-24): `hour_local` → `hour_start`, `keyboard/music` → `keyinput/mus`

**Session memory** (per-user, longitudinal): `outputs/agentic_pilot/memory/`
- Accumulates receptivity feedback (diary, raw ER_desire, INT_availability) across EMA entries
- **No prediction targets** in memory (receptivity leakage fixed 2026-02-23)

---

## Completed Work

### ✅ Data Infrastructure
- [x] 5-fold across-subject CV splits generated (399 users)
- [x] Phase 1 offline processing: hourly Parquet files generated for screen, motion, keyinput, light, mus
- [x] Home location and participant platform metadata
- [x] `SensingQueryEngine` — Parquet-backed tool layer for agent + ML pipeline
- [x] **CRITICAL BUG FIXED (2026-02-24)**: `hour_local` column not being recognized → all queries returned "no data". Fix: added to `_normalize_hour_start()` alternates. Also fixed `keyboard→keyinput`, `music→mus` directory aliases in both `query_tools.py` and `hourly_features.py`.

### ✅ Baseline: AR Autocorrelation
- Script: `scripts/run_ar_baseline.py`
- Output: `outputs/ar_baseline/ar_results.json`
- **Results (n=15,585 predictions, 399 users):**

| Variant | Mean MAE | Mean BA | Mean F1 |
|---------|----------|---------|---------|
| `last_value` (AR1) | 2.758 | 0.658 | 0.617 |
| `rolling_mean_w3` | 2.552 | 0.658 | 0.617 |

These are the **empirical ceilings** for autocorrelation-based prediction (no sensing needed). The V4 agent must approach or beat BA=0.658 to demonstrate sensing adds value.

### ✅ Baseline: Text (TF-IDF / BoW on diary)
- Script: `scripts/run_dl_baselines.py --pipelines text`
- Output: `outputs/advanced_baselines/text/`
- **Results (diary-present rows only, 5-fold CV):**

| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| TF-IDF | 3.999 | 0.613 | 0.570 |
| BoW | 4.043 | 0.607 | 0.561 |

### ✅ Baseline: Traditional ML (Sensing features)
- Script: `scripts/run_ml_baselines.py --features parquet`
- Output: `outputs/ml_baselines/`
- **Old results (WRONG — sensing data bug not yet fixed when run):**

| Model | Mean MAE | Mean BA | Mean F1 |
|-------|----------|---------|---------|
| RF | 4.373 | 0.500 | 0.298 |
| XGBoost | 4.373 | 0.500 | 0.360 |
| Ridge | 4.373 | — | — |
| Logistic | — | 0.500 | 0.000 |

BA = 0.500 for all = **features were all zeros** (sensing data path bug). Must **rerun** after fixing the `hour_local` bug.

### ✅ Baseline: Deep Learning (MLP)
- Script: `scripts/run_dl_baselines.py --pipelines dl`
- **Status: Never successfully completed** — was running locally with OOM risk when user stopped it
- Same sensing data bug as ML: results would be invalid even if it completed
- Must **rerun on Titan server** with GPU

### ✅ Code: V4 Agentic Agent
- CC backend: `src/agent/cc_agent.py` — invokes `claude --print` subprocess with MCP server
- API backend: `src/agent/agentic_sensing.py` — Anthropic SDK direct tool-use loop
- MCP server: `src/sense/mcp_server.py` — serves sensing tools with EMA timestamp cutoff
- Session memory: per-user longitudinal accumulation of receptivity signals only
- **Status: Only dry-run tested** (pilot results show `[DRY RUN] Placeholder` predictions)
- Sensing tools now confirmed working after `hour_local` bug fix

### ✅ Evaluation Framework
- `src/evaluation/` — metrics computation (MAE, BA, F1)
- AR baseline as empirical upper bound for autocorrelation-only prediction
- Oracle mode **removed** (would turn prediction into time series forecasting)

---

## 🔄 Running on Titan (zhiyuan@172.29.39.82)

### ML + DL Baselines — CURRENTLY RUNNING IN PARALLEL (2026-02-24 ~15:17)

All baselines were relaunched with fold-level parallelism after identifying resource underutilization.

**Hardware allocation:**
- **CPU (40 cores)**: 5 ML fold processes × 8 cores each = all 40 cores utilized
- **GPU1 RTX A6000 (49GB)**: 5 DL MLP fold processes in parallel
- **GPU0 RTX A5000 (24GB)**: Transformer + Combined (sentence-transformers models)

**Processes running:**

| Process | PIDs | GPU | Output |
|---------|------|-----|--------|
| ML (RF/XGB/Logistic/Ridge) fold 1-5 | 1203761-1203765 | CPU 8cores/fold | `outputs/ml_baselines_v2/fold_N/` |
| DL MLP fold 1-5 | 1203766-1203770 | GPU1 | `outputs/advanced_baselines/dl/fold_N/` |
| Transformer (MiniLM) all folds | 1212187 | GPU0 (cuda:0) | `outputs/advanced_baselines/transformer/` |
| Combined (sensor+text) all folds | 1212188 | GPU0 (cuda:0) | `outputs/advanced_baselines/combined/` |
| Text (TF-IDF/BoW) all folds | 1212186 | CPU | `outputs/advanced_baselines/text/` |

**After all complete, merge fold results:**
```bash
# On Titan:
cd ~/proactive-affective-agent

# Merge ML results
PYTHONPATH=. python scripts/merge_baseline_results.py \
    --output outputs/ml_baselines_v2 --type ml

# Merge DL MLP results
PYTHONPATH=. python scripts/merge_baseline_results.py \
    --output outputs/advanced_baselines --type dl --pipeline dl
```

**Check progress:**
```bash
ssh zhiyuan@172.29.39.82 "
  ps aux | grep 'run_ml\|run_dl' | grep -v grep | wc -l
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  tail -3 ~/proactive-affective-agent/outputs/ml_baselines_v2/fold_1.log
"
```

**Feature selection:** `MLBaseline` uses `SelectKBest` with K tuned via 3-fold inner CV (grid: 25/50/75/100% of features). `n_jobs=8` per fold.

**DL CUDA:** `deep_learning_baselines.py` uses `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — automatically uses GPU1 for MLP.

---

## ❌ Pending

### 1. Merge ML/DL fold results (Priority: HIGH — after jobs complete)
Run `scripts/merge_baseline_results.py` to combine per-fold JSON outputs.

### 2. V4 Agent — Real End-to-End Test (Priority: HIGH)
**Status:** Only dry-run completed. No real `claude --print` subprocess calls made.

```bash
# Run real V4 CC on a few users
python3 scripts/run_agentic_pilot.py \
    --users 71,72,73 \
    --backend cc \
    --max-turns 16 \
    --output outputs/agentic_pilot/v4_real_test
```

**Prerequisites:**
- Sensing data confirmed working (done — `hour_local` fix)
- V4 CC backend reads `claude` CLI from PATH (must be available on server too)

For API backend test (doesn't need claude CLI, just ANTHROPIC_API_KEY):
```bash
python3 scripts/run_agentic_pilot.py \
    --users 71,72,73 \
    --backend api \
    --output outputs/agentic_pilot/v4_api_test
```

---

## Server Setup: Titan (zhiyuan@172.29.39.82)

**Status:** ✅ Code + data synced (2026-02-24). Both GPUs in use.

```bash
# 1. Sync code to server (exclude large raw data, keep processed Parquets)
rsync -avz \
    --exclude="data/bucs-data" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --exclude=".git" \
    --exclude="outputs" \
    --exclude="*.egg-info" \
    /Users/zwang/Documents/proactive-affective-agent/ \
    zhiyuan@172.29.39.82:~/proactive-affective-agent/

# 2. Sync processed Parquet data (hourly features, splits)
rsync -avz \
    data/processed/ \
    zhiyuan@172.29.39.82:~/proactive-affective-agent/data/processed/

# 3. On server: install packages in efficient-ser env
ssh zhiyuan@172.29.39.82
source ~/anaconda3/etc/profile.d/conda.sh && conda activate efficient-ser
pip install xgboost pyarrow scikit-learn sentence-transformers

# 4. GPU to use: GPU1 (RTX A6000, 49GB, nearly free)
export CUDA_VISIBLE_DEVICES=1
```

**Available GPU:**
- GPU0: RTX A5000 (24GB) — has alphapose using ~480MB
- GPU1: RTX A6000 (49GB) — basically free (22MB used)

---

## Results Summary Table (Current State)

| System | Method | Mean MAE↓ | Mean BA↑ | Mean F1↑ | Notes |
|--------|---------|-----------|----------|----------|-------|
| AR baseline | Last value (AR1) | 2.758 | 0.658 | 0.617 | Autocorrelation ceiling |
| AR baseline | Rolling mean (w=3) | 2.552 | 0.658 | 0.617 | Autocorrelation ceiling |
| Text | TF-IDF on diary | 3.999 | 0.613 | 0.570 | Diary-present rows only |
| Text | BoW on diary | 4.043 | 0.607 | 0.561 | Diary-present rows only |
| ML sensing | RF (parquet) | ~~4.373~~ | ~~0.500~~ | ~~0.298~~ | **INVALID** — features were all zeros |
| ML sensing | XGBoost (parquet) | ~~4.373~~ | ~~0.500~~ | ~~0.360~~ | **INVALID** — features were all zeros |
| DL sensing | MLP (parquet) | — | — | — | Not run yet |
| Transformer | MiniLM on diary | — | — | — | Not run yet |
| Combined | Sensor + text | — | — | — | Not run yet |
| **V4 agent** | CC backend | — | — | — | **Not real-run yet** |
| **V4 agent** | API backend | — | — | — | **Not real-run yet** |

---

## Key Design Decisions (Frozen)

1. **ER_desire binary threshold:** `>= 5` (scale midpoint), NOT person mean
2. **Oracle mode:** Removed — giving agent historical PA/NA = time series forecasting, not sensing-based prediction
3. **Session memory content:** Only receptivity signals (diary, raw ER_desire, INT_availability). Never prediction targets.
4. **5-fold CV:** Across-subject (participant-level). All observations from a user in exactly one fold.
5. **AR baseline role:** Empirical ceiling for autocorrelation-only prediction. V4 must approach BA≈0.658 to show sensing adds value.
6. **Feature selection:** SelectKBest with K tuned as hyperparameter (25/50/75/100% fractions), 3-fold inner CV.

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/sense/query_tools.py` | SensingQueryEngine — core data access for agent + baselines |
| `src/data/hourly_features.py` | HourlyFeatureLoader — Parquet features for ML/DL |
| `src/agent/cc_agent.py` | V4 CC backend (claude --print subprocess) |
| `src/agent/agentic_sensing.py` | V4 API backend (Anthropic SDK) |
| `src/sense/mcp_server.py` | MCP server for CC backend |
| `src/baselines/ml_pipeline.py` | ML baseline with SelectKBest feature selection |
| `scripts/run_agentic_pilot.py` | Main V4 evaluation runner |
| `scripts/run_ml_baselines.py` | Traditional ML baselines |
| `scripts/run_dl_baselines.py` | DL + Text + Transformer + Combined baselines |
| `scripts/run_ar_baseline.py` | AR autocorrelation baseline |
| `data/processed/hourly/` | Hourly Parquet files (screen/motion/keyinput/light/mus) |
| `data/processed/splits/` | 5-fold CV splits |

---

## Recent Bug Fixes (2026-02-24)

1. **`hour_local` column not recognized** in `_normalize_hour_start()`:
   - Affected: `SensingQueryEngine` (`query_tools.py`) and `HourlyFeatureLoader` (`hourly_features.py`)
   - Root cause: Parquet files use `hour_local` column; code only checked `timestamp/ts/datetime/hour`
   - Fix: Added `hour_local`, `hour_utc` to the alternates list in both files
   - Impact: ALL sensing queries were returning "no data" → ML BA=0.5 → invalid results

2. **Keyboard/music directory name mismatch**:
   - Disk uses: `keyinput/`, `mus/` directories with `{pid}_keyinput_hourly.parquet` filenames
   - Code expected: `keyboard/`, `music/`
   - Fix: Added `_MODALITY_DIR_ALIAS` in `SensingQueryEngine`; updated `MODALITY_DIRS` in `HourlyFeatureLoader`

3. **Oracle mode removed**: `--feedback-mode oracle` in `run_agentic_pilot.py` was passing historical PANAS/NA to the agent = prediction target leakage.

4. **Receptivity leakage in tools**: `get_receptivity_history` and `find_similar_days` were returning prediction targets (PANAS_Pos/Neg, Individual_level_*) as historical context. Fixed to return only ER_desire + INT_availability.
