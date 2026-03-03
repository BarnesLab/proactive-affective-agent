# BUCS Data Filtering Plan

## Motivation

RyanHub/Bobo empirical evidence: LLM agents perform better with processed behavioral summaries than raw data. Raw hourly numbers add noise and waste tokens. Bobo achieves 7.4x compression (3581 → 486 events) with better agent comprehension.

Goal: create a filtered/narrative version of our hourly sensing data that reduces noise and token cost for V2/V4 agentic agents.

## Platform Segmentation

The dataset has two distinct user groups:

| Group | Count | App Data | Light | Keyboard | Motion | Screen |
|-------|-------|----------|-------|----------|--------|--------|
| **Android** | 120 (29.5%) | Yes | Yes (92%) | Yes (60%) | Yes (93%) | Yes (100%) |
| **iOS** | 287 (70.5%) | No | No | Yes (69%) | Yes (87%) | Yes (100%) |

**All 5 pilot users (71, 119, 164, 310, 458) are Android.**

Detection rule: user has app data AND light data → Android; otherwise → iOS.

## Per-Modality Filtering Decisions

### 1. Motion (all users)
- **Keep**: stationary_min, walking_min, automotive_min
- **Drop**: running_min, cycling_min (nearly always 0 for cancer survivors)
- **Drop**: active_min (redundant = walking + running + cycling)
- **Drop hours**: coverage_pct < 50% — unreliable sensor data
- **Collapse**: consecutive stationary-only hours into episode summary
- **Narrative**: "Mostly stationary (4h), walked 15min around 2pm, drove 30min in evening"

### 2. Screen (all users)
- **Keep**: screen_on_min, screen_n_sessions
- **Drop**: mean_session_min, max_session_min (derivable, noisy)
- **Drop hours**: screen_on_min = 0 — no information
- **Narrative**: "Screen · 5 opens · 42m total" (Bobo format)

### 3. App Usage (Android only, 120 users)
- **Keep**: app_total_min, app_social_min, app_comm_min, app_entertainment_min, app_n_apps
- **Drop**: app_game_min (all zeros across pilot users)
- **Drop**: structural_missing_app flag (redundant once we know platform)
- **Hide entirely for iOS users** — don't show "no data"
- **Narrative**: "Apps: 45min total (social 20min, comm 5min), 12 apps"

### 4. Keyboard (all users, sparse)
- **Keep**: key_n_sessions, key_chars_typed
- **Drop**: key_words_typed (derivable from chars), key_session_min
- **Drop**: key_prop_pos, key_prop_neg — sentiment analysis too noisy and sparse (43% null)
- **Only show when**: key_n_sessions > 0
- **Narrative**: "Typed 450 chars across 3 sessions"

### 5. Light (Android only, 110 users)
- **Keep**: light_mean_lux only
- **Drop**: min_lux, max_lux, n_captures — noise
- **Categorize**: dark (<10), indoor (10-500), outdoor (>500)
- **Hide for iOS users**
- **Narrative**: "Environment: indoor" (not "light_mean_lux=127.3")

### 6. Music (sparse, 91 users)
- **Only show when**: mus_is_listening = True AND mus_n_tracks > 0
- **Drop**: mus_n_ads (not behaviorally meaningful)
- **Otherwise**: completely hidden — don't waste tokens on "no music data"

## Output Format

### Per-user filtered database
```
data/processed/filtered/{pid}_daily_filtered.parquet
```

Schema per row (one row per day):
- `date_local`: date
- `platform`: "android" | "ios"
- `modalities_available`: list of available modality names
- Per-hour columns replaced by **daily behavioral summary fields**:
  - `motion_stationary_total_min`, `motion_walking_total_min`, `motion_automotive_total_min`
  - `motion_n_walking_episodes`, `motion_primary_activity`
  - `screen_total_min`, `screen_n_sessions`
  - `app_total_min`, `app_social_min`, `app_comm_min`, `app_entertainment_min`, `app_n_apps_mean` (Android only)
  - `keyboard_total_chars`, `keyboard_n_sessions`
  - `light_category`: "dark" / "indoor" / "outdoor" / null (Android only)
  - `music_listening`: bool
  - `music_n_tracks`: int (only if listening)
- `narrative`: pre-generated human-readable behavioral summary string

### Narrative template
```
[Motion] Mostly stationary (8.2h tracked), walked 25min in 3 episodes, drove 15min.
[Screen] 12 opens, 1h 45m total screen time.
[Apps] 2h 10m total (social 45min, comm 12min), avg 15 apps/hour. (Android only)
[Keyboard] Typed 890 chars in 4 sessions. (only if sessions > 0)
[Environment] Indoor. (Android only, only if light data)
[Music] Listening, 5 tracks. (only if listening)
```

## Implementation

1. `scripts/build_filtered_data.py` — reads hourly parquets, applies filters, outputs daily filtered parquets + narrative
2. Updates to `src/sense/query_tools.py` — add `query_filtered()` tool that returns narrative summaries
3. Future: A/B test raw hourly tools vs filtered narrative tools

## What This Does NOT Change

- Current V2/V4 experiments continue with raw hourly data (running now)
- Evaluation pipeline unchanged
- This creates a parallel "filtered" dataset for next-iteration experiments
