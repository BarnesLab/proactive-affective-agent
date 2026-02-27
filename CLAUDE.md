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

### Backend (All Free — Claude Max Subscription)
- **V1/V3/CALLM**: `claude -p` CLI via `ClaudeCodeClient`
- **V2/V4**: `claude --print` + MCP server via `AgenticCCAgent` (cc_agent.py)
- All inference is free through Max subscription — no paid API keys used anywhere
- Limit concurrent `claude --print` processes to ~5 to avoid Max rate limits

### Model
- During testing phase: use **Haiku** for speed
- For final experiments: use **Sonnet** (`"sonnet"` for CLI)
- Never use Opus for experiments.

## Critical Files

| Path | Purpose |
|------|---------|
| `src/agent/personal_agent.py` | Unified agent entry point (routes to V1-V4/CALLM) |
| `src/agent/cc_agent.py` | V2/V4 agentic agent (claude --print + MCP) |
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
1. **Max rate limits**: Running too many concurrent `claude --print` processes exhausts the Max subscription rate limit. Limit to ~5 concurrent processes.
2. **Ridge regression diverges**: 3/5 folds produce MAE ~1e12. Exclude from paper.
3. **DL MLP fold 5 diverges**: Extreme outlier targets. Report 4-fold mean.

## Titan Server
- `zhiyuan@172.29.39.82`, SSH key auth
- **Be careful with resources**: check `free -h` and `uptime` before running jobs. Limit to 1-2 parallel processes. Other users share the server.
- Env: `source ~/anaconda3/etc/profile.d/conda.sh && conda activate efficient-ser`

## Conventions
- All code, comments, commit messages in English
- Auto commit + push after changes
- Test with Sonnet, never Opus
