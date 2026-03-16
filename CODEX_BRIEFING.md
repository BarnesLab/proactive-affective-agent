# Codex Briefing: Proactive Affective Agent

> ⚠️ **BEFORE running any experiment tasks**, read `PROJECT_CONTROL.md` and check `~/.openclaw/workspace/project-control.json`. If paused (globally, by model, or by project), do NOT start new tasks. Let running tasks finish naturally.

## What This Project Does

Predicts cancer survivors' emotional states (PANAS positive/negative affect, emotion regulation desire, 13 binary emotional indicators) from **passive smartphone sensing data** and **diary text**. This is for an IMWUT paper comparing LLM-based prediction approaches.

## Current Experiment: Claude as Backend

We have 5 agent versions, all using **Claude** (via the `claude` CLI — no API key, free through Max subscription):

| Version | Input | Architecture | Backend |
|---------|-------|-------------|---------|
| CALLM | Diary text + TF-IDF RAG | Single LLM call | `claude -p` (piped prompt) |
| V1 | Sensing data only | Single LLM call | `claude -p` |
| V2 | Sensing data only | Multi-turn tool-use | `claude --print` + MCP server |
| V3 | Diary + Sensing | Single LLM call | `claude -p` |
| V4 | Diary + Sensing | Multi-turn tool-use | `claude --print` + MCP server |

### How We Avoid API Costs (Key Trick)

Claude Max subscription ($100/mo or $200/mo) gives unlimited access to the `claude` CLI tool (Claude Code). We abuse this:

1. **Structured agents (CALLM, V1, V3)**: Shell out to `claude -p "prompt here"` which reads from stdin, returns the LLM response to stdout. See `src/think/llm_client.py` → `ClaudeCodeClient`.

2. **Agentic agents (V2, V4)**: Shell out to `claude --print -m sonnet --output-format json` with an MCP server config that exposes sensing data query tools. Claude autonomously calls tools to investigate behavioral data, then produces a prediction. See `src/agent/cc_agent.py` → `AgenticCCAgent`.

**This is the pattern Codex should replicate** — but using OpenAI models instead of Claude. Options:
- **OpenAI API** (costs money but straightforward) — `openai.ChatCompletion.create()`
- **ChatGPT Plus CLI** if one exists
- **Azure OpenAI** if available
- Any other free/cheap way to call GPT-4o or o1

## What You (Codex) Need to Implement

### Goal
Create parallel versions (e.g., `GPT-CALLM`, `GPT-V1`, `GPT-V3`, optionally `GPT-V2`, `GPT-V4`) that use **OpenAI models** instead of Claude, so we can compare Claude vs GPT in the paper.

### Minimum Viable: Structured Versions (CALLM, V1, V3)

These are easiest — just replace the LLM backend:

1. **Create `src/think/openai_client.py`** — drop-in replacement for `ClaudeCodeClient`:
   ```python
   class OpenAIClient:
       def __init__(self, model="gpt-4o", api_key=None, ...):
           ...
       def generate(self, prompt: str, system_prompt: str = None) -> str:
           # Call OpenAI API, return response text
           ...
   ```

2. **The prompts are identical** — reuse everything in `src/think/prompts.py`. The prompt templates are model-agnostic (they just ask for JSON output).

3. **Wire it up** in `src/agent/personal_agent.py` — add a `backend` parameter or create new version strings like `"gpt-callm"`, `"gpt-v1"`, `"gpt-v3"`.

### Stretch: Agentic Versions (V2, V4)

These are harder because they rely on Claude's MCP (Model Context Protocol) for tool use. For OpenAI:
- Use **OpenAI function calling / tools API** instead
- The tool definitions are in `src/sense/query_tools.py` → `SensingQueryEngine` (5 tools: `query_sensing`, `query_raw_events`, `get_daily_summary`, `compare_to_baseline`, `find_peer_cases`)
- Convert MCP tool schemas to OpenAI function schemas
- Implement the multi-turn tool-use loop yourself (call model → parse tool calls → execute → feed results back → repeat)

## Data Layout

```
data/
  processed/
    splits/
      group_{1-5}_{train,test}.csv    # 5-fold EMA splits
    hourly/
      screen/motion/keyinput/light/mus/  # Hourly Parquet sensing data
    filtered/
      {pid}_daily_filtered.parquet    # Pre-aggregated daily sensing features
```

### EMA Columns (prediction targets)
- `PANAS_Pos` (0-30): positive affect score
- `PANAS_Neg` (0-30): negative affect score
- `ER_desire` (0-10): emotion regulation desire
- `Individual_level_*_State` (bool): 13 binary emotional state indicators
- `INT_availability` ("yes"/"no"): intervention availability
- `emotion_driver` (text): diary entry

### Sensing Data (inputs)
- **Screen**: unlock events, session durations
- **Motion**: accelerometer-derived activity (stationary/walking/driving/running)
- **Keyboard**: typing volume, word sentiment
- **Light**: ambient light levels
- **App usage**: foreground app categories and durations (Android only)

## Users & Progress

### Completed (Pilot V1 — first 5 users, all versions done with Claude Sonnet)
Users: `2, 34, 56, 78, 90` (auto-selected, ~427 entries each)
Results: `outputs/pilot/`

### In Progress (Pilot V2 — 5 new users, Claude Sonnet + cross-user RAG)
Users: **399, 258, 43, 403, 338**
Results: `outputs/pilot_v2/`

Current progress (as of 2026-03-06):
- All 5 versions running in parallel via `scripts/night_scheduler.py`
- User 399 partially done, users 258/43/403/338 not started yet
- Estimated completion: 1-2 days

### What Codex Should Run
**Same users**: 399, 258, 43, 403, 338 (for direct comparison)
Output to: `outputs/pilot_gpt/` (separate directory)

## Key Files to Read

| File | What It Does |
|------|-------------|
| `src/think/llm_client.py` | Claude CLI wrapper — **replace this for OpenAI** |
| `src/think/prompts.py` | All prompt templates — **reuse as-is** |
| `src/think/parser.py` | JSON response parser — **reuse as-is** |
| `src/agent/personal_agent.py` | Agent orchestrator — **extend with GPT versions** |
| `src/agent/structured.py` | V1 pipeline (sensing → prompt → predict) |
| `src/agent/structured_full.py` | V3 pipeline (diary + sensing → prompt → predict) |
| `src/agent/cc_agent.py` | V2/V4 agentic agent — **hardest to port** |
| `src/simulation/simulator.py` | Experiment runner with checkpointing |
| `src/data/schema.py` | Data classes (SensingDay, UserProfile, etc.) |
| `src/sense/query_tools.py` | Sensing data query tools (for agentic versions) |
| `scripts/run_pilot.py` | Main experiment entry point |
| `scripts/evaluate_pilot.py` | Compute metrics from checkpoints |

## Evaluation Metrics

- **MAE** on PANAS_Pos, PANAS_Neg, ER_desire (continuous)
- **Balanced Accuracy** on 13 binary Individual_level_*_State fields
- **Macro F1** on the same binary fields
- Computed by `scripts/evaluate_pilot.py` and `src/evaluation/metrics.py`

## Environment

- Python 3.9+
- Key packages: pandas, numpy, scikit-learn, tiktoken
- OpenAI package: `pip install openai`
- API key: set `OPENAI_API_KEY` environment variable (or put in `.env`)

## Quick Start for Codex

```bash
# 1. Install OpenAI
pip install openai

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Test the structured pipeline with GPT (once implemented)
PYTHONPATH=. python3 scripts/run_pilot.py --version gpt-callm --users 399 --model gpt-4o --output-dir outputs/pilot_gpt

# 4. Evaluate
PYTHONPATH=. python3 scripts/evaluate_pilot.py --checkpoint-dir outputs/pilot_gpt/checkpoints
```

## Cost Estimate

For structured versions (CALLM/V1/V3), ~446 entries × 5 users × 3 versions = ~6,700 API calls.
- GPT-4o: ~$0.01-0.03 per call → **$70-200 total**
- GPT-4o-mini: ~$0.001 per call → **$7-20 total**

For agentic versions (V2/V4), multiply by ~5-10x due to multi-turn tool use.

## Important Notes

1. **No data leakage**: Cross-user RAG uses only training data. Never peek at test labels.
2. **Checkpointing**: The simulator saves after every prediction. If it crashes, it resumes automatically.
3. **Rate limits**: OpenAI has rate limits too. Use exponential backoff (see `src/utils/rate_limit.py` for patterns).
4. **Same prompts**: Use identical prompts for fair comparison. Only the LLM backend changes.
5. **Session memory**: V2/V4 agentic agents accumulate per-user session memory (reflections on past predictions). Port this if implementing agentic versions.
