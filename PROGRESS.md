# BUCS Pilot Study — Progress & Next Steps

Last updated: 2026-02-24

---

## Project Overview

Goal: Predict emotional state (PANAS Pos/Neg, ER desire, 15 individual-level binary states) of cancer survivors at EMA time points using passive smartphone sensing data. Compare an agentic LLM-based approach (V4) against traditional ML/DL baselines.

**Study cohort:** 400 participants, 5-fold across-subject CV (participant-level, zero leakage), ~15,585 test EMA entries per fold.

---

## Architecture Overview

```
data/
  raw/                        # Original CSVs (cancer_survival/Passive Sensing Data/)
  processed/
    splits/                   # 5-fold CV splits (group_{1-5}_{train,test}.csv)
    hourly/                   # Per-participant per-modality hourly Parquet files
      motion/   001_motion_hourly.parquet ...
      screen/   001_screen_hourly.parquet ...
      keyinput/ 001_keyinput_hourly.parquet ...  (logical name: "keyboard")
      mus/      001_mus_hourly.parquet ...        (logical name: "music")
      light/    001_light_hourly.parquet ...
      home_locations.parquet
      participant_platform.parquet

src/
  sense/
    query_tools.py         # SensingQueryEngine — Parquet-backed tool dispatch
    mcp_server.py          # FastMCP server for claude --print subprocess (V4-CC)
  agent/
    agentic_sensing.py     # V4 API agent (Anthropic SDK tool-use loop)
    cc_agent.py            # V4-CC agent (claude --print subprocess)
  baselines/
    ml_pipeline.py         # RF / XGBoost / LR with SelectKBest + GridSearchCV
    deep_learning_baselines.py  # MLP on Parquet features (PyTorch)
    text_baselines.py      # TF-IDF / BoW on diary text
    transformer_baselines.py    # Sentence-BERT embeddings
    combined_baselines.py       # Late fusion (sensing + text)
    feature_builder.py     # build_parquet_features() → 134-dim feature vectors
  data/
    hourly_features.py     # HourlyFeatureLoader (MODALITY_DIRS, _normalize_hour_start)
    loader.py              # DataLoader (EMA splits, sensing CSVs)
  think/
    parser.py              # Parse JSON prediction from LLM output

scripts/
  run_agentic_pilot.py     # Main V4 (API) + V4-CC pilot runner
  run_ml_baselines.py      # Traditional ML runner
  run_dl_baselines.py      # DL + text + transformer + combined runner
  run_ar_baseline.py       # AR autocorrelation baseline
```

---

## What's Done

### Data & Infrastructure

- [x] **5-fold across-subject CV splits** — `data/processed/splits/group_{1-5}_{train,test}.csv`
  - Participant-level: all EMA from a participant in exactly one fold
  - 399 participants, ~16k EMA entries total
- [x] **Hourly Parquet data** — generated and present in `data/processed/hourly/`
  - Modalities: motion, screen, keyboard (dir: keyinput), music (dir: mus), light
  - No GPS or accelerometer available for this cohort
  - 134-dimensional feature vector per EMA (24h lookback)

### Critical Fixes Applied (This Session)

- [x] **`_normalize_hour_start` in query_tools.py & hourly_features.py** — Parquets use `hour_local` column but code was looking for `hour_start`/`timestamp`/etc. → added `hour_local` to alt-column list. **Root cause of BA=0.5** (all features were zero before this fix).
- [x] **`_MODALITY_DIR_ALIAS` in query_tools.py** — `keyboard` dir on disk = `keyinput`, `music` dir = `mus`. Added alias map in `_parquet_path()`.
- [x] **`MODALITY_DIRS` in hourly_features.py** — Same fix: keyboard→keyinput, music→mus. Also fixed filename to use `disk_name` not logical name.
- [x] **`cc_agent.py` — prompt arg ordering** — `--disallowed-tools` was consuming prompt. Fixed: prompt now placed before `--disallowed-tools` in cmd list.
- [x] **`cc_agent.py` — `CLAUDECODE=1` inherited by subprocess** — Stripped `CLAUDECODE`, `CLAUDE_CODE`, `CLAUDE_CODE_SESSION_ID` from subprocess env.
- [x] **`mcp_server.py` — `SensingQueryEngine` missing `ema_df`** — Added `DataLoader` to load EMA data before constructing engine.
- [x] **`cc_agent.py` — `max_turns=8` insufficient** — Increased to 16 (test used 15 turns).
- [x] **`deep_learning_baselines.py` — SIGSEGV on macOS** — PyTorch OpenMP crash on Apple Silicon/Xcode Python 3.9. Fixed with `torch.set_num_threads(1)` + `OMP_NUM_THREADS=1`.
- [x] **ML feature selection** — Added `SelectKBest` + `VarianceThreshold` in `MLBaseline.fit()` with K tuned via 3-fold inner GridSearchCV (K ∈ {25%, 50%, 75%, 100%} of n_features). Previously models had no feature selection → overfitting / BA=0.5.
- [x] **Oracle mode removal** — Removed `--feedback-mode oracle` from `run_agentic_pilot.py`. Oracle (giving agent historical PANAS/NA) = time-series forecasting, not sensing prediction.
- [x] **ER_desire binary threshold** — Changed from person-mean to scale midpoint ≥5 (0-10 scale). Distribution: mean=2.37, median=1.0, bimodal, 45.9% zeros. Midpoint ≥5 gives 24.9% positive rate.
- [x] **`run_ar_baseline.py`** — New script, AR autocorrelation baseline (last_value + rolling_mean_w3).
- [x] **`_build_prompt()` in cc_agent.py** — Added `session_memory` parameter (was silently dropped).

### Baselines Implemented & Run

#### AR Autocorrelation Baseline ✅ (output: `outputs/ar_baseline/ar_results.json`)
| Variant | MAE (cont.) | BA (binary) | F1 (binary) |
|---------|------------|------------|------------|
| last_value (AR1) | 2.758 | 0.658 | 0.617 |
| rolling_mean_w3 | 2.552 | 0.658 | 0.617 |

- n=15,585 predictions, 399 users, 5-fold test sets combined
- **This is the sensing-free ceiling** — any sensing-based model must beat this to prove sensing adds value

#### Traditional ML Baselines (parquet features, 134-dim, 24h lookback)
- Status: **Was BA=0.5 (zero features bug)**. Bug fixed (hour_local column name). Need rerun on Titan.
- Models: RF, XGBoost, LogisticRegression, Ridge
- With feature selection: SelectKBest (K=25%/50%/75%/100% tuned by 3-fold inner CV)
- Scoring: balanced_accuracy (binary), neg_MAE (continuous)

#### Text Baselines ✅ (run previously)
- TF-IDF + BoW on diary text (emotion_driver column)
- Status: Run complete, results in `outputs/advanced_baselines/text/`

#### DL Baselines (MLP on Parquet features)
- Status: **Was crashing** (PyTorch SIGSEGV on macOS, Xcode Python 3.9 + OpenMP). Fix applied.
- Recommend running on Titan server (2x GPU, CUDA 12.1, stable PyTorch 2.3.1)

#### Transformer Baselines
- Status: Not run yet.
- Sentence-BERT embeddings on diary text + Ridge/LR classifier
- Requires `sentence-transformers` package

#### Combined (Late Fusion) Baselines
- Status: Not run yet.
- Parquet features + sentence-BERT embeddings

### V4 Agentic Agent ✅

#### V4-CC (claude --print subprocess) — **Tested & Working**
- Full end-to-end test passed (user 71, 2023-11-20)
- 9 sensing tool calls: `get_daily_summary`, `query_sensing` ×2, `query_raw_events` ×3, `get_receptivity_history`, `find_similar_days`
- Returned valid JSON prediction (`_parse_error: False`)
- MCP server initializes cleanly with correct `ema_df`
- 15 turns consumed → max_turns now set to 16

#### V4 API (Anthropic SDK) — Not yet tested with real data
- Code structure correct (same as V4-CC but via SDK tool-use loop)
- Session memory implemented (receptivity feedback only, no oracle leakage)
- Needs real API call test

---

## What's NOT Done / Needs Work

### Critical

1. **Run ML baselines on Titan server** (Titan: zhiyuan@172.29.39.82)
   - Local machine: memory issues, Python 3.9 from Xcode
   - Titan: 2x RTX GPU (A5000 24GB + A6000 49GB), CUDA 12.1, PyTorch 2.3.1
   - Need to sync code + processed data, set up conda env (`efficient-ser` or new), run
   - Command: `python scripts/run_ml_baselines.py --features parquet`

2. **Run DL baselines on Titan**
   - MLP needs GPU for speed (local is too slow / crashes)
   - Transformer + combined also benefit from GPU

3. **Run V4 pilot on test users** (5-10 users, ~50 EMA entries)
   - Script ready: `scripts/run_agentic_pilot.py`
   - Need Claude Max subscription context / API key
   - Expected runtime: ~5-10 min per user (16 turns × ~20s each)

4. **Run V4 API agent test** (agentic_sensing.py)
   - Verify real Anthropic SDK tool-use loop works end-to-end

### Important

5. **Feature coverage analysis** — confirm non-zero features for a sample of users/dates
   - Before fix: 0 features (all NaN/zero)
   - After fix: 44-68 non-zero features for user 71 (verified)
   - Need systematic coverage check across all 400 users

6. **Transformer + combined baselines** — not yet run, need Titan

7. **Results comparison table** — once all baselines complete, compare:
   - AR baseline (ceiling)
   - Traditional ML (RF, XGB, LR)
   - DL (MLP)
   - Text (TF-IDF/BoW)
   - Transformer (Sentence-BERT)
   - Combined (late fusion)
   - V4 Agentic (API and/or CC)

### Lower Priority

8. **Write final Methods section** for CHI paper — ML/DL methods description drafted inline, not yet in paper
9. **Ablation: sensing modalities** — which modalities most predictive? (motion vs screen vs keyboard)
10. **Error analysis** — which EMA entries does V4 get wrong? Is it correlated with missing data?

---

## Known Issues / Gotchas

| Issue | Status |
|-------|--------|
| Xcode Python 3.9 → PyTorch SIGSEGV | Fixed (torch.set_num_threads(1)). Prefer Titan for DL. |
| `hour_local` column name mismatch | **Fixed** in query_tools.py + hourly_features.py |
| keyboard dir = keyinput, music = mus | **Fixed** in both files |
| ML BA=0.5 | Fixed (was zero features) — needs rerun to confirm |
| V4-CC CLAUDECODE env pollution | **Fixed** |
| V4-CC max_turns too low | **Fixed** (now 16) |
| Session memory: only receptivity, not PANAS | By design (oracle mode removed) |
| ER_desire binary threshold | **Changed** to midpoint ≥5 (was person-mean) |

---

## Baseline Results Summary (Updated as runs complete)

| Method | n | Mean MAE | Mean BA | Mean F1 | Notes |
|--------|---|---------|---------|---------|-------|
| AR (last_value) | 15,585 | 2.758 | 0.658 | 0.617 | Sensing-free ceiling |
| AR (rolling_mean_w3) | 15,585 | 2.552 | 0.658 | 0.617 | Sensing-free ceiling |
| ML (RF) | pending | — | — | — | Need rerun (feature bug fixed) |
| ML (XGB) | pending | — | — | — | Need rerun |
| ML (LR) | pending | — | — | — | Need rerun |
| Text (TF-IDF) | done | — | — | — | See outputs/advanced_baselines/text/ |
| DL (MLP) | pending | — | — | — | Need Titan |
| Transformer | pending | — | — | — | Need Titan |
| Combined | pending | — | — | — | Need Titan |
| V4-CC (agentic) | 1 test | — | — | — | E2E verified, full pilot pending |

---

## Titan Server Setup Notes

- Host: `zhiyuan@172.29.39.82`
- GPUs: RTX A5000 (24GB, GPU0, partially used) + RTX A6000 (49GB, GPU1, free)
- CUDA: 12.3, Driver: 545.23.08
- Conda env `efficient-ser`: has torch 2.3.1+cu121, needs: `pip install xgboost pyarrow scikit-learn`
- Project not yet synced — needs `rsync` of code + `data/processed/`

To set up and run:
```bash
# 1. Sync code
rsync -avz --exclude="__pycache__" --exclude="*.pyc" --exclude=".git" \
  /Users/zwang/Documents/proactive-affective-agent/ \
  zhiyuan@172.29.39.82:~/proactive-affective-agent/

# 2. Sync processed data
rsync -avz data/processed/ zhiyuan@172.29.39.82:~/proactive-affective-agent/data/processed/

# 3. Run on server
ssh zhiyuan@172.29.39.82
conda activate efficient-ser
pip install xgboost pyarrow scikit-learn sentence-transformers --quiet
cd ~/proactive-affective-agent
PYTHONPATH=. python scripts/run_ml_baselines.py --features parquet
PYTHONPATH=. python scripts/run_dl_baselines.py --pipelines dl,transformer,combined
```
