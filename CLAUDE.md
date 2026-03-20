# PULSE (Proactive Affective Agent) — BUCS Study

> ⚠️ **BEFORE running any experiment tasks**, read `PROJECT_CONTROL.md` and check `~/.openclaw/workspace/project-control.json`. If paused, do NOT start new tasks.

## Project Overview
LLM-based emotional state prediction for cancer survivors using passive smartphone sensing + diary data. System name: **PULSE**. Paper target: IMWUT.

## Naming Convention (Paper ↔ Code)
| Paper Name | Code Name | Description |
|---|---|---|
| **Struct-Sense** | v1 | Structured-agentic, sensing-only |
| **Auto-Sense** | v2 | Autonomously-agentic, sensing-only |
| **Struct-Multi** | v3 | Structured-agentic, multimodal |
| **Auto-Multi** | v4 | Autonomously-agentic, multimodal |
| **Auto-Sense+** | v5 | Auto-Sense + filtered data |
| **Auto-Multi+** | v6 | Auto-Multi + filtered data |
| **CALLM** | callm | Diary-only baseline |

Internal code still uses v1-v6. Paper uses descriptive names. Never use v1-v6 in paper text.

## 2x2 Agent Design
| | Sensing-only | Multimodal |
|---|---|---|
| **Structured-Agentic** | Struct-Sense | Struct-Multi |
| **Autonomously-Agentic** | Auto-Sense / Auto-Sense+ | Auto-Multi / Auto-Multi+ |

## Backend
- **Structured (v1/v3/callm)**: `claude -p` CLI
- **Autonomous (v2/v4/v5/v6)**: `claude --print` + MCP server
- All free via Claude Max subscription. Limit to ~5 concurrent processes.
- Model: **Sonnet** for experiments, Haiku for testing only. Never Opus.

## Overleaf Sync
- **Project ID**: `6999d011b24a9f1d4e6e53e8`
- **Rule**: After every draft edit, compile PDF + sync to Overleaf. Always keep Overleaf up to date.
- **Compile command**: `cd draft && pdflatex -interaction=nonstopmode main.tex && pdflatex main.tex && open main.pdf`
- Sync via MCP: `mcp__overleaf__write_file` then `commit_changes` then `push_changes`
- `draft/` is gitignored (Overleaf-managed, not in GitHub)

## Key Evaluation (as of 2026-03-20)
- **50 users, 7 versions**: 217/350 checkpoints complete, 133 tasks running/queued
- **18-user primary set**: V2/V4/V5/V6 complete (18/18). CALLM: 16/18, V1: 12/18, V3: 10/18.
- **Metrics**: BA and F1 only. No MAE (dropped). No AR baseline (unfair — assumes oracle access to previous EMA).
- **evaluation.json is STALE** (2026-03-12) — re-run after tasks complete
- Script: `PYTHONPATH=. python3 scripts/evaluate_pilot.py`

## Running Tests
```bash
PYTHONPATH=. python3 -m pytest tests/ -v
python3 scripts/integration_test.py --n-entries 10
```

## Experiment Execution
- **sonnet_watcher.sh**: daemon that maintains TARGET parallel `run_pilot.py` processes
- **Completion detection**: `quality_check.get_done_set()` checks checkpoint n_entries vs expected (≥95% = done)
- **Rate limits**: all wait indefinitely (5h rolling → 3h wait, weekly → 12h wait). Never produces empty predictions.
- **Peak hours**: weekdays 8AM-2PM, no new launches (running tasks continue)
- **Scheduling**: user-first (users closest to 7/7 completion get priority)
- **Legacy `outputs/pilot/`**: Haiku-contaminated, NOT used by evaluation

## Known Issues
1. Corrupted V1 checkpoints: user61 (1651), user86 (1317), user99 (1569) — auto-filtered by evaluate_pilot.py

## Conventions
- All code, comments, commit messages in English
- Auto commit + push after changes
