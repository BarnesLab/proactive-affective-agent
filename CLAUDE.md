# Proactive Affective Agent — BUCS Pilot Study

## Project Overview
LLM-based emotional state prediction for cancer survivors using passive smartphone sensing + diary data. Compares 5 prediction approaches in a 2x2 design (structured vs agentic × sensing-only vs multimodal) plus a CALLM baseline.

## Key Architecture

### 2x2 Agent Design
| | Sensing-only | Multimodal (sensing + diary) |
|---|---|---|
| **Structured** (fixed pipeline) | V1 | V3 |
| **Agentic** (autonomous tool-use) | V2 | V4 |

Plus **CALLM** baseline (diary + TF-IDF RAG, structured output).

### Backend Split
- **V1/V3/CALLM**: `claude -p` CLI via `ClaudeCodeClient` (Max subscription, no API cost)
- **V2/V4**: Anthropic Python SDK with tool-use loop (requires `ANTHROPIC_API_KEY`, incurs cost)

### Model
- **All versions must use Sonnet** (`claude-sonnet-4-6` for SDK, `"sonnet"` for CLI)
- Never use Opus for experiments (cost). Never use Haiku (quality).

## Critical Files

| Path | Purpose |
|------|---------|
| `src/agent/personal_agent.py` | Unified agent entry point (routes to V1-V4/CALLM) |
| `src/agent/agentic_sensing.py` | V4 agentic agent (SDK tool-use loop) |
| `src/agent/agentic_sensing_only.py` | V2 agentic agent (no diary) |
| `src/agent/structured.py` | V1 structured sensing-only |
| `src/agent/structured_full.py` | V3 structured multimodal |
| `src/sense/query_tools.py` | SensingQueryEngine — Parquet-backed tool definitions |
| `src/think/llm_client.py` | ClaudeCodeClient wrapping `claude -p` |
| `src/think/prompts.py` | All LLM prompts |
| `src/simulation/simulator.py` | PilotSimulator orchestrator |
| `scripts/run_pilot.py` | Main pilot runner (all versions) |
| `scripts/run_agentic_pilot.py` | Standalone V4 runner with session memory |
| `scripts/integration_test.py` | End-to-end LLM integration tests |
| `scripts/evaluate_pilot.py` | Compute MAE/BA/F1 from checkpoints |

## Data Layout
- EMA splits: `data/processed/splits/group_{1-5}_{train,test}.csv`
- Hourly Parquet: `data/processed/hourly/{screen,motion,keyinput,light,mus}/`
- Pilot checkpoints: `outputs/pilot/checkpoints/{version}_user{id}_checkpoint.json`

## Running Tests
```bash
# Unit tests (dry-run, no LLM calls)
PYTHONPATH=. python3 -m pytest tests/ -v

# Integration tests (real LLM calls, logs to test_logs/)
python3 scripts/integration_test.py --n-entries 10
python3 scripts/integration_test.py --versions v2,v4 --n-entries 5  # agentic only
```

## Known Issues / Gotchas
1. **Parallel tool calls**: Anthropic models may issue multiple `tool_use` blocks in one response. ALL must have matching `tool_result` blocks — cannot break mid-batch. Fixed in session 4.
2. **V4 pilot checkpoints are empty**: Ran before the parallel tool-use bug fix. Must delete and re-run.
3. **Ridge regression diverges**: 3/5 folds produce MAE ~1e12. Exclude from paper.
4. **DL MLP fold 5 diverges**: Extreme outlier targets. Report 4-fold mean.

## Titan Server
- `zhiyuan@172.29.39.82`, SSH key auth
- **Be careful with resources**: check `free -h` and `uptime` before running jobs. Limit to 1-2 parallel processes. Other users share the server.
- Env: `source ~/anaconda3/etc/profile.d/conda.sh && conda activate efficient-ser`

## Conventions
- All code, comments, commit messages in English
- Auto commit + push after changes
- Test with Sonnet, never Opus
