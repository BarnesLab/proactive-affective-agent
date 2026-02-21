# Next Steps: Waiting for Raw Minute-Level Data

## Status (2026-02-21)

Phase 1 pilot complete: CALLM >> V1/V2 (diary+RAG crushes daily aggregate sensing).
V3/V4 code written and pushed. ML baseline code ready. Waiting for colleague's raw data.

## Phase 1 Results Summary

| Metric | CALLM (diary+RAG) | V1 (sensing, structured) | V2 (sensing, autonomous) |
|--------|-------------------|--------------------------|--------------------------|
| Mean MAE | **~1.16** | ~high | 7.06 |
| Mean BA | **~0.632** | ~0.52 | 0.521 |
| Mean F1 | **~0.440** | ~low | 0.190 |
| PT BA | **~0.716** | — | 0.543 |

## What We're Waiting For

From colleague:
1. **Raw minute-level sensing data** (accelerometer, GPS, screen, etc.)
2. **Hourly feature extraction code** (their pipeline)

## When Data Arrives: Action Plan

### Step 1: Integrate Hourly Features
- Implement `src/data/hourly_features.py` (currently placeholder)
  - `HourlyFeatureLoader.get_features_for_ema()`: raw → hourly windows aligned to EMA
  - `format_hourly_sensing_text()`: natural language for LLM prompts
  - `get_feature_matrix()`: numeric matrix for ML baselines
- Update `src/data/schema.py` if new fields are needed

### Step 2: Update V1-V4 Sensing Summaries
- `src/think/prompts.py` → `format_sensing_summary()` needs to support hourly data
  - Show temporal trends (e.g., "sleep disrupted 2-4am, high mobility 10am-2pm")
  - Currently only shows daily aggregates
- V1/V2 prompts may need adjustment for richer sensing context
- V3/V4 prompts should highlight hourly patterns in cross-modal analysis

### Step 3: Update ML Baselines
- `src/baselines/feature_builder.py` → implement `build_hourly_features()`
  - Flatten hourly windows into feature vector (e.g., 24h × features_per_hour)
  - Handle variable lookback windows
- Re-run RF/XGBoost/LogReg with hourly features

### Step 4: Re-run All Experiments
- **V1-V4**: 4 versions × 427 entries = 1708 LLM calls (~2 days on Max Plan)
  - Can reuse CALLM results (diary-only, no sensing change)
  - Run V3/V4 first (most interesting: diary + hourly sensing + RAG)
- **ML baselines**: minutes, no LLM needed
- Use `scripts/run_parallel.sh` (25 processes)

### Step 5: Unified Comparison
- `src/evaluation/unified_comparison.py` → load all results, generate tables
- Compare daily vs hourly features (do hourly features help?)
- Generate LaTeX table for paper

## Known Issues to Fix

1. **Parallel CSV overwrite**: `run_version()` parallel processes overwrite `{version}_predictions.csv`
   - Fix: use per-user CSV filenames, merge at end
   - Or: rely on checkpoint JSONs (all data preserved there)

2. **V1 metrics incomplete**: V1 results from earlier runs may need re-collection
   - Check `outputs/pilot/checkpoints/v1_user*_checkpoint.json`

## File Locations

- V3/V4 code: `src/agent/structured_full.py`, `src/agent/autonomous_full.py`
- ML baselines: `src/baselines/ml_pipeline.py`, `scripts/run_ml_baselines.py`
- Hourly placeholder: `src/data/hourly_features.py`
- Unified eval: `src/evaluation/unified_comparison.py`
- Pilot outputs: `outputs/pilot/` (traces, checkpoints, logs)
