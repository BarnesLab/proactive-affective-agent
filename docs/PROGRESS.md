# Project Progress & Next Steps

> **Auto-updated by agent after major task completions.**
> Last updated: 2026-02-24 11:35

---

## Architecture: 2×2 Design Space

The core novelty is **agentic investigation** applied across two data conditions:

| | **Structured** (fixed pipeline) | **Agentic** (autonomous investigation) |
|---|---|---|
| **Sensing-only** | V2-structured | **V2-agentic** ← agentic loop |
| **Multimodal** (diary + sensing) | V4-structured | **V4-agentic** ← agentic loop |

Alongside: CALLM (diary+RAG baseline, CHI 2025), ML baselines (RF/XGBoost).

**Key insight**: The agentic tool-use loop (V5 in earlier naming, now V2/V4-agentic) uses Anthropic SDK tool calls to query raw hourly Parquet data autonomously, rather than consuming pre-formatted feature summaries.

---

## Data Pipeline Status

### Phase 0 ✅ Complete
- `participant_platform.parquet`: 418 participants (297 iOS, 121 Android)
- `home_locations.parquet`: 359/413 participants, median home_radius_m=24.8m (DBSCAN eps bug fixed)

### Phase 1 — Light Modalities ✅ Complete (~30 min)
| Modality | Participants | Status |
|----------|-------------|--------|
| motion | 371 | ✅ |
| screen/app | 418 | ✅ (Android APPUSAGE duration bug fixed) |
| keyboard | 280 | ✅ |
| music | 91 | ✅ |
| light | 111 (Android) | ✅ |

### Phase 1 — Heavy Modalities ⏳ Not started (run overnight)
| Modality | Data size | Status |
|----------|-----------|--------|
| accelerometer | ~105 GB | ⏳ run `python scripts/offline/process_accel.py` |
| GPS | variable | ⏳ run `python scripts/offline/process_gps.py` |

---

## Evaluation Status

| Method | Implementation | Evaluated on BUCS |
|--------|---------------|-------------------|
| CALLM | ✅ | ✅ Mean MAE~1.16, BA~0.63, F1~0.44 |
| V1 (structured, sensing) | ✅ | ✅ Weak (~BA 0.52) |
| V2 (autonomous, sensing) | ✅ | ✅ Weak (MAE 7.06, BA 0.52) |
| V3 (structured, multimodal) | ✅ | ⏳ Pending BUCS Parquet data |
| V4 (autonomous, multimodal) | ✅ | ⏳ Pending BUCS Parquet data |
| V2-agentic / V4-agentic | ✅ | ⏳ Run `scripts/run_agentic_pilot.py` |
| ML baselines | ✅ | ⏳ Pending |

---

## Completed This Session (2026-02-24)

1. **Phase 0**: Built participant roster (418 users), computed home locations
   - Fixed DBSCAN eps bug: eps in radians = 150m / 6_371_000m ≈ 2.356e-5
2. **Phase 1**: Ran motion, screen/app, keyboard, music, light processors
   - Fixed Android APPUSAGE duration: use `n_foreground_ms` with proportional allocation
   - Fixed `screen_on_min` cap: physical screen ≤ 60 min/hr; `app_total_min` uncapped
3. **Agentic pipeline fixes**:
   - Renamed `get_ema_history` → `get_receptivity_history` (reflects real-world JITAI deployment: only intervention accept/reject is known, not high-freq EMA)
   - Fixed `_execute_tool` in `agentic_sensing.py` to use `engine.call_tool()` (Parquet API)
   - Fixed `_build_initial_context` modality detection (Parquet engine has no `sensing_dfs`)
   - Added token usage tracking per EMA entry (`_input_tokens`, `_output_tokens`, `_total_tokens`)
   - Added diary-present filtering in `run_agentic_pilot.py` (apple-to-apple CALLM comparison)
   - Fixed `SensingQueryEngine` init: use `processed_dir` (Parquet), not `sensing_dfs` (legacy)
4. **Documentation**:
   - README fully rewritten with 2×2 design space
   - `docs/advisor-sync-architecture.html`: SVG diagram + belief-update reasoning trace

---

## Immediate Next Steps

### Must-do before experiments
- [ ] Run Phase 1 heavy: `python scripts/offline/process_accel.py` and `process_gps.py` (overnight)
- [ ] Dry-run V5 agentic: `python scripts/run_agentic_pilot.py --users 71 --dry-run`
- [ ] Verify diary-present filtering works correctly

### Experiments to run
- [ ] V3-structured: `python scripts/run_pilot.py --version v3 --users 71,164,119,458,310`
- [ ] V4-autonomous: `python scripts/run_pilot.py --version v4 --users 71,164,119,458,310`
- [ ] V4-agentic: `python scripts/run_agentic_pilot.py --users 71,164,119 --model claude-opus-4-6`
- [ ] ML baselines: `python scripts/run_ml_baselines.py`
- [ ] Full comparison: `src/evaluation/unified_comparison.py`

### Code improvements
- [ ] Add raw data query support to `query_sensing` (currently returns hourly aggregates; agent could optionally request sub-hourly raw data)
- [ ] GitHub Pages for HTML docs (public links for advisor sync HTML)
- [ ] Fix parallel CSV overwrite in `run_pilot.py` (use per-user CSV filenames)

### Research
- [ ] Token efficiency hypothesis: V4-agentic should use fewer tokens than V3/V4 if it only queries high-entropy signals
- [ ] Decide whether to submit to CHI 2026 or IMWUT

---

## Known Issues

| Issue | Location | Status |
|-------|----------|--------|
| Parallel CSV overwrite | `scripts/run_pilot.py` | ⏳ Minor; checkpoint JSONs preserve all data |
| V1 metrics incomplete | `outputs/pilot/` | ⏳ Re-run if needed |
| Accel/GPS not yet processed | `data/processed/hourly/` | ⏳ Run overnight |
| GitHub Pages not set up | `docs/*.html` | ⏳ Low priority |

---

## File Quick Reference

| Component | Path |
|-----------|------|
| Agentic agent | `src/agent/agentic_sensing.py` |
| Parquet query engine | `src/sense/query_tools.py` |
| Tool schemas (SENSING_TOOLS) | `src/sense/query_tools.py:43` |
| Agentic pilot runner | `scripts/run_agentic_pilot.py` |
| Phase 1 scripts | `scripts/offline/` |
| EMA data | `data/ema/` |
| Processed Parquet | `data/processed/hourly/` |
| Outputs | `outputs/` |
