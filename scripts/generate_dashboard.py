#!/usr/bin/env python3
"""Generate interactive HTML dashboard for pilot experiment results.

Reads unified JSONL records (*_records.jsonl), V5/V6 checkpoints, evaluation.json,
and trace files. Generates a self-contained HTML dashboard with:
- Overview: aggregate metrics, bar charts, research design matrix
- Methodology: version descriptions, agent flow diagrams (Mermaid.js), data pipeline
- Per-User Analysis: time series, entry inspection, version comparison

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --output outputs/pilot/dashboard.html
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = PROJECT_ROOT / "outputs" / "pilot"

VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
VERSION_LABELS = {
    "callm": "CALLM",
    "v1": "V1 Structured",
    "v2": "V2 Agentic",
    "v3": "V3 Struct+Diary",
    "v4": "V4 Agent+Diary",
    "v5": "V5 Filtered",
    "v6": "V6 Filtered+Diary",
}
VERSION_COLORS = {
    "callm": "#f97583",
    "v1": "#79c0ff",
    "v2": "#d2a8ff",
    "v3": "#3fb950",
    "v4": "#f0883e",
    "v5": "#56d4dd",
    "v6": "#f778ba",
}

BINARY_STATES = [
    "PA", "NA", "happy", "sad", "afraid", "miserable", "worried",
    "cheerful", "pleased", "grateful", "lonely", "interactions_quality",
    "pain", "forecasting", "ER_desire",
]

PROMPT_TRUNCATE = 5000
RESPONSE_TRUNCATE = 5000


def load_all_records() -> dict:
    """Load all JSONL records, group by user_id, merge across versions.

    Returns dict: {user_id: {entry_idx: {date, slot, ground_truth, versions: {ver: record}}}}
    """
    users = {}
    files = sorted(PILOT_DIR.glob("*_records.jsonl"))
    total = 0

    for fpath in files:
        m = re.match(r"^(\w+)_user(\d+)_records\.jsonl$", fpath.name)
        if not m:
            continue
        version = m.group(1)
        user_id = int(m.group(2))

        if user_id not in users:
            users[user_id] = {}

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_idx = rec.get("entry_idx", -1)
                total += 1

                if entry_idx not in users[user_id]:
                    users[user_id][entry_idx] = {
                        "date": rec.get("date_local", ""),
                        "timestamp": rec.get("timestamp_local", ""),
                        "slot": rec.get("ema_slot", "unknown"),
                        "ground_truth": rec.get("ground_truth", {}),
                        "versions": {},
                    }

                full_prompt = rec.get("full_prompt", "") or ""
                full_response = rec.get("full_response", "") or ""
                system_prompt = rec.get("system_prompt", "") or ""

                prompt_truncated = len(full_prompt) > PROMPT_TRUNCATE
                response_truncated = len(full_response) > RESPONSE_TRUNCATE

                users[user_id][entry_idx]["versions"][version] = {
                    "prediction": rec.get("prediction", {}),
                    "reasoning": rec.get("reasoning", ""),
                    "confidence": rec.get("confidence"),
                    "sensing_summary": rec.get("sensing_summary", ""),
                    "tool_calls": rec.get("tool_calls"),
                    "n_tool_calls": rec.get("n_tool_calls"),
                    "n_rounds": rec.get("n_rounds"),
                    "rag_cases": rec.get("rag_cases", []),
                    "memory_excerpt": rec.get("memory_excerpt", ""),
                    "emotion_driver": rec.get("emotion_driver", ""),
                    "trait_summary": rec.get("trait_summary", ""),
                    "has_diary": rec.get("has_diary", False),
                    "diary_length": rec.get("diary_length"),
                    "model": rec.get("model", ""),
                    "elapsed_seconds": rec.get("elapsed_seconds"),
                    "input_tokens": rec.get("input_tokens", 0),
                    "output_tokens": rec.get("output_tokens", 0),
                    "total_tokens": rec.get("total_tokens", 0),
                    "cost_usd": rec.get("cost_usd", 0),
                    "llm_calls": rec.get("llm_calls"),
                    "conversation_length": rec.get("conversation_length"),
                    "modalities_available": rec.get("modalities_available", []),
                    "modalities_missing": rec.get("modalities_missing", []),
                    "full_prompt": full_prompt[:PROMPT_TRUNCATE],
                    "prompt_truncated": prompt_truncated,
                    "full_response": full_response[:RESPONSE_TRUNCATE],
                    "response_truncated": response_truncated,
                    "system_prompt": system_prompt[:PROMPT_TRUNCATE],
                }

    print(f"  Loaded {total} records from {len(files)} files")
    print(f"  Users: {sorted(users.keys())}")
    for uid in sorted(users.keys()):
        vers = set()
        for e in users[uid].values():
            vers.update(e["versions"].keys())
        print(f"    User {uid}: {len(users[uid])} entries, versions: {sorted(vers)}")

    return users


def load_v5v6_checkpoints(users: dict) -> dict:
    """Load V5/V6 checkpoint data and merge into users dict.

    V5/V6 only have checkpoint files (no JSONL records), so we
    convert them to the same record format.
    """
    pilot_users = [71, 119, 164, 310, 458]
    added = 0

    for ver in ["v5", "v6"]:
        for uid in pilot_users:
            cp_path = PILOT_DIR / "checkpoints" / f"{ver}_user{uid}_checkpoint.json"
            if not cp_path.exists():
                continue

            data = json.loads(cp_path.read_text())
            preds = data.get("predictions", [])
            gts = data.get("ground_truths", [])
            metadata = data.get("metadata", [])

            if uid not in users:
                users[uid] = {}

            for i, (pred, gt) in enumerate(zip(preds, gts)):
                meta = metadata[i] if i < len(metadata) else {}
                ts = meta.get("timestamp_local", "")
                date_str = ts.split(" ")[0] if ts else ""
                # Determine slot from timestamp
                slot = "unknown"
                if ts:
                    try:
                        hour = int(ts.split(" ")[1].split(":")[0])
                        if hour < 12:
                            slot = "morning"
                        elif hour < 17:
                            slot = "afternoon"
                        else:
                            slot = "evening"
                    except (IndexError, ValueError):
                        pass

                # Find matching entry_idx or create new
                entry_idx = None
                for idx, entry in users[uid].items():
                    if entry.get("timestamp", "").startswith(ts[:16] if ts else "NONE"):
                        entry_idx = idx
                        break
                    if entry.get("date") == date_str and entry.get("slot") == slot:
                        # Check if this entry already has this version
                        if ver not in entry.get("versions", {}):
                            entry_idx = idx
                            break

                if entry_idx is None:
                    # Create new entry
                    entry_idx = max(users[uid].keys(), default=-1) + 1
                    users[uid][entry_idx] = {
                        "date": date_str,
                        "timestamp": ts,
                        "slot": slot,
                        "ground_truth": gt,
                        "versions": {},
                    }

                # Build version record from checkpoint prediction
                users[uid][entry_idx]["versions"][ver] = {
                    "prediction": {k: v for k, v in pred.items()
                                   if k not in ("reasoning", "confidence")},
                    "reasoning": pred.get("reasoning", ""),
                    "confidence": pred.get("confidence"),
                    "sensing_summary": "",
                    "tool_calls": None,
                    "n_tool_calls": None,
                    "n_rounds": None,
                    "rag_cases": [],
                    "memory_excerpt": "",
                    "emotion_driver": "",
                    "trait_summary": "",
                    "has_diary": ver == "v6",
                    "diary_length": None,
                    "model": "sonnet",
                    "elapsed_seconds": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0,
                    "llm_calls": None,
                    "conversation_length": None,
                    "modalities_available": [],
                    "modalities_missing": [],
                    "full_prompt": "",
                    "prompt_truncated": False,
                    "full_response": "",
                    "response_truncated": False,
                    "system_prompt": "",
                }
                added += 1

    print(f"  Loaded {added} V5/V6 entries from checkpoints")
    return users


def load_evaluation() -> dict:
    """Load evaluation.json for aggregate metrics."""
    eval_path = PILOT_DIR / "evaluation.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text())
    return {}


def load_token_stats() -> dict:
    """Load token stats from trace files."""
    traces_dir = PILOT_DIR / "traces"
    stats = {}
    if not traces_dir.exists():
        return stats

    for f in traces_dir.glob("*.json"):
        parts = f.stem.split("_")
        ver = parts[0]
        try:
            data = json.loads(f.read_text())
            inp = data.get("_input_tokens", 0) or 0
            out = data.get("_output_tokens", 0) or 0
            if inp > 0:
                if ver not in stats:
                    stats[ver] = {"input": 0, "output": 0, "count": 0, "calls": 0}
                stats[ver]["input"] += inp
                stats[ver]["output"] += out
                stats[ver]["count"] += 1
                stats[ver]["calls"] += data.get("_llm_calls", 1) or 1
        except Exception:
            pass

    return stats


def load_elapsed_stats() -> dict:
    """Load elapsed time stats from JSONL records and V5/V6 results files."""
    stats = {}

    # From JSONL records (CALLM, V1-V4)
    for fpath in sorted(PILOT_DIR.glob("*_records.jsonl")):
        m = re.match(r"^(\w+)_user(\d+)_records\.jsonl$", fpath.name)
        if not m:
            continue
        ver = m.group(1)
        if ver not in stats:
            stats[ver] = {"total": 0, "count": 0, "max": 0}

        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    elapsed = rec.get("elapsed_seconds")
                    if elapsed and elapsed > 0:
                        stats[ver]["total"] += elapsed
                        stats[ver]["count"] += 1
                        stats[ver]["max"] = max(stats[ver]["max"], elapsed)
                except Exception:
                    pass

    # From V5/V6 results files and JSONL checkpoints
    filtered_dir = PROJECT_ROOT / "outputs" / "filtered_pilot"
    if filtered_dir.exists():
        # Results JSON files (have metadata with elapsed_seconds)
        for fpath in sorted(filtered_dir.glob("*_results_*.json")):
            try:
                data = json.loads(fpath.read_text())
                ver = data.get("run_config", {}).get("version", "")
                if not ver:
                    continue
                if ver not in stats:
                    stats[ver] = {"total": 0, "count": 0, "max": 0}
                for m in data.get("metadata", []):
                    elapsed = m.get("elapsed_seconds")
                    if elapsed and elapsed > 0:
                        stats[ver]["total"] += elapsed
                        stats[ver]["count"] += 1
                        stats[ver]["max"] = max(stats[ver]["max"], elapsed)
            except Exception:
                pass

        # JSONL checkpoint files (may have elapsed_seconds per entry)
        cp_dir = filtered_dir / "checkpoints"
        if cp_dir.exists():
            for fpath in sorted(cp_dir.glob("*.jsonl")):
                m = re.match(r"^user_\d+_(v\d+)\.jsonl$", fpath.name)
                if not m:
                    continue
                ver = m.group(1)
                if ver not in stats:
                    stats[ver] = {"total": 0, "count": 0, "max": 0}
                with open(fpath) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                            elapsed = rec.get("elapsed_seconds")
                            if elapsed and elapsed > 0:
                                stats[ver]["total"] += elapsed
                                stats[ver]["count"] += 1
                                stats[ver]["max"] = max(stats[ver]["max"], elapsed)
                        except Exception:
                            pass

    return stats


def build_js_data(users: dict) -> dict:
    """Convert internal data structure to JS-friendly format."""
    js_data = {}
    for uid in sorted(users.keys()):
        entries = []
        for idx in sorted(users[uid].keys()):
            entry = users[uid][idx]
            entry["idx"] = idx
            entries.append(entry)
        js_data[uid] = entries
    return js_data


def generate_html(js_data: dict, eval_data: dict, token_stats: dict,
                  elapsed_stats: dict, output_path: Path):
    """Generate the self-contained HTML dashboard."""
    data_json = json.dumps(js_data, default=str, ensure_ascii=False)
    versions_json = json.dumps(VERSIONS)
    labels_json = json.dumps(VERSION_LABELS)
    colors_json = json.dumps(VERSION_COLORS)
    binary_json = json.dumps(BINARY_STATES)
    eval_json = json.dumps(eval_data, default=str)
    token_json = json.dumps(token_stats)
    elapsed_json = json.dumps(elapsed_stats)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Proactive Affective Agent — Pilot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
:root {{
  --bg-primary: #0d1117;
  --bg-secondary: #161b22;
  --bg-tertiary: #1c2128;
  --bg-input: #21262d;
  --border: #30363d;
  --text-primary: #f0f6fc;
  --text-secondary: #c9d1d9;
  --text-muted: #8b949e;
  --accent-blue: #58a6ff;
  --accent-green: #3fb950;
  --accent-red: #f97583;
  --accent-orange: #f0883e;
  --accent-purple: #d2a8ff;
  --accent-cyan: #56d4dd;
  --accent-pink: #f778ba;
  --sidebar-width: 280px;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; background: var(--bg-primary); color: var(--text-secondary); font-size: 13px; }}

/* Header */
.header {{
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
  height: 48px;
}}
.header h1 {{ font-size: 15px; color: var(--text-primary); font-weight: 600; white-space: nowrap; }}
.nav-tabs {{
  display: flex;
  gap: 0;
  height: 100%;
  margin-left: 32px;
}}
.nav-tab {{
  padding: 0 20px;
  height: 100%;
  display: flex;
  align-items: center;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
  white-space: nowrap;
}}
.nav-tab:hover {{ color: var(--text-secondary); }}
.nav-tab.active {{ color: var(--accent-blue); border-bottom-color: var(--accent-blue); font-weight: 600; }}
.header-stats {{ font-size: 12px; color: var(--text-muted); display: flex; gap: 16px; }}
.header-stat {{ display: flex; align-items: center; gap: 4px; }}
.header-stat .num {{ color: var(--text-primary); font-weight: 600; }}

/* Layout */
.page {{ display: none; height: calc(100vh - 48px); overflow-y: auto; }}
.page.active {{ display: block; }}
.page-analysis {{ display: none; height: calc(100vh - 48px); }}
.page-analysis.active {{ display: flex; }}

/* Analysis layout with sidebar */
.sidebar {{
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
  overflow-y: auto;
}}
.sidebar-filters {{
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.filter-row {{ display: flex; gap: 6px; align-items: center; }}
.filter-row label {{ font-size: 11px; color: var(--text-muted); width: 40px; flex-shrink: 0; }}
.filter-row select {{ flex: 1; background: var(--bg-input); border: 1px solid var(--border); color: var(--text-secondary); padding: 3px 6px; border-radius: 4px; font-size: 11px; }}
.user-list {{ flex: 1; overflow-y: auto; }}
.user-item {{ padding: 8px 12px; cursor: pointer; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; transition: background 0.1s; }}
.user-item:hover {{ background: var(--bg-tertiary); }}
.user-item.active {{ background: var(--bg-tertiary); border-left: 3px solid var(--accent-blue); padding-left: 9px; }}
.user-item .uid {{ font-weight: 600; color: var(--text-primary); font-size: 13px; }}
.user-item .meta {{ font-size: 11px; color: var(--text-muted); text-align: right; }}
.user-item .ver-dots {{ display: flex; gap: 3px; margin-top: 2px; justify-content: flex-end; }}
.ver-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

.main {{ flex: 1; overflow-y: auto; padding: 16px 20px; }}

/* ─── Overview Page ─── */
.overview-content {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
.overview-content h2 {{ font-size: 20px; color: var(--text-primary); margin-bottom: 16px; font-weight: 600; }}
.overview-content h3 {{ font-size: 15px; color: var(--text-primary); margin: 20px 0 12px; font-weight: 600; }}

/* Research Design Matrix */
.design-matrix {{
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 24px;
  font-size: 13px;
}}
.design-matrix th {{
  padding: 10px 16px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.design-matrix td {{
  padding: 12px 16px;
  border: 1px solid var(--border);
  text-align: center;
  vertical-align: middle;
}}
.design-matrix td.row-header {{
  background: var(--bg-tertiary);
  color: var(--text-muted);
  font-weight: 600;
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
}}
.matrix-cell {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 6px;
  font-weight: 600;
  font-size: 13px;
}}
.matrix-cell .dot {{ width: 10px; height: 10px; border-radius: 50%; }}

/* Metric Cards */
.metric-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}}
.metric-card {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}}
.metric-card h4 {{
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}}
.metric-card .chart-wrap {{ position: relative; height: 220px; }}

/* Results table */
.results-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-bottom: 24px;
}}
.results-table th {{
  padding: 8px 12px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-weight: 600;
  text-align: center;
}}
.results-table th:first-child {{ text-align: left; }}
.results-table td {{
  padding: 8px 12px;
  border: 1px solid var(--border);
  text-align: center;
}}
.results-table td:first-child {{ text-align: left; font-weight: 600; }}
.results-table tr:hover {{ background: var(--bg-tertiary); }}
.best-val {{ color: var(--accent-green); font-weight: 700; }}
.warn-val {{ color: var(--accent-red); font-size: 10px; }}

/* Token stats */
.token-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}}
.token-stat {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px;
  text-align: center;
}}
.token-stat .ts-label {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
.token-stat .ts-val {{ font-size: 18px; font-weight: 700; color: var(--text-primary); }}
.token-stat .ts-sub {{ font-size: 10px; color: var(--text-muted); margin-top: 2px; }}

/* ─── Methodology Page ─── */
.method-content {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
.method-content h2 {{ font-size: 20px; color: var(--text-primary); margin-bottom: 16px; font-weight: 600; }}

.version-card {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 20px;
  overflow: hidden;
}}
.version-card-header {{
  padding: 16px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}}
.version-card-header .v-badge {{
  padding: 4px 12px;
  border-radius: 16px;
  font-weight: 700;
  font-size: 13px;
  color: #fff;
}}
.version-card-header .v-title {{ font-size: 15px; color: var(--text-primary); font-weight: 600; }}
.version-card-header .v-tags {{ display: flex; gap: 6px; margin-left: auto; }}
.version-card-header .v-tag {{
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  background: var(--bg-tertiary);
  color: var(--text-muted);
  border: 1px solid var(--border);
}}
.version-card-body {{
  padding: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}}
.version-card-body .v-desc {{ font-size: 13px; line-height: 1.7; color: var(--text-secondary); }}
.version-card-body .v-desc strong {{ color: var(--text-primary); }}
.version-card-body .v-flow {{ min-height: 120px; }}

/* Collapsible tool details */
.tool-details {{
  margin-top: 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}}
.tool-details summary {{
  padding: 10px 14px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  cursor: pointer;
  background: var(--bg-tertiary);
  user-select: none;
  transition: color 0.2s;
}}
.tool-details summary:hover {{
  color: var(--text-primary);
}}
.tool-details[open] summary {{
  border-bottom: 1px solid var(--border);
  color: var(--text-primary);
}}
.tool-list {{
  list-style: none;
  padding: 10px 14px;
  margin: 0;
  font-size: 12px;
  line-height: 1.8;
  color: var(--text-secondary);
}}
.tool-list li {{
  padding: 2px 0;
}}
.tool-list li strong {{
  color: var(--accent-blue);
  font-family: monospace;
  font-size: 11px;
}}

/* Data pipeline */
.data-compare {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}}
.data-box {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}}
.data-box h4 {{
  font-size: 13px;
  color: var(--text-primary);
  margin-bottom: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.data-box .data-tag {{
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
}}
.data-box pre {{
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg-primary);
  padding: 12px;
  border-radius: 6px;
  font-size: 11px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--text-secondary);
  line-height: 1.5;
  max-height: 400px;
  overflow-y: auto;
}}
.data-box .file-list {{
  font-size: 11px;
  color: var(--text-muted);
  margin-bottom: 8px;
  font-family: monospace;
}}
.data-box .file-list span {{ color: var(--accent-blue); }}

/* Mermaid overrides for dark theme */
.mermaid {{ background: transparent !important; }}

/* ─── Per-User Analysis (existing styles) ─── */
.user-summary {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.user-summary h2 {{ font-size: 16px; color: var(--text-primary); }}
.user-summary .stats {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-muted); }}
.user-summary .stat-val {{ color: var(--text-primary); font-weight: 600; }}

.chart-container {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
}}
.chart-controls {{ display: flex; gap: 6px; margin-bottom: 10px; }}
.chart-btn {{
  padding: 4px 12px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-primary);
  color: var(--text-muted);
  cursor: pointer;
  font-size: 11px;
  transition: all 0.15s;
}}
.chart-btn:hover {{ background: var(--bg-tertiary); }}
.chart-btn.active {{ background: var(--bg-tertiary); border-color: var(--accent-blue); color: var(--accent-blue); }}
.chart-wrap {{ position: relative; height: 220px; }}

.timeline-strip {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 16px;
  margin-bottom: 16px;
}}
.timeline-strip h3 {{ font-size: 12px; color: var(--text-muted); margin-bottom: 8px; font-weight: 500; }}
.timeline-dots {{ display: flex; flex-wrap: wrap; gap: 4px; }}
.t-dot {{
  width: 14px; height: 14px;
  border-radius: 3px;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.1s;
  position: relative;
}}
.t-dot:hover {{ transform: scale(1.3); }}
.t-dot.active {{ border-color: var(--accent-blue); transform: scale(1.3); }}
.t-dot.slot-morning {{ background: #f0c040; }}
.t-dot.slot-afternoon {{ background: #e08040; }}
.t-dot.slot-evening {{ background: #6040c0; }}
.t-dot.slot-unknown {{ background: #484f58; }}
.t-dot-tip {{
  display: none;
  position: absolute;
  bottom: 20px; left: 50%;
  transform: translateX(-50%);
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 10px;
  white-space: nowrap;
  z-index: 10;
  color: var(--text-primary);
  pointer-events: none;
}}
.t-dot:hover .t-dot-tip {{ display: block; }}

.detail-panel {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 16px;
  overflow: hidden;
}}
.detail-header {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.detail-header h3 {{ font-size: 14px; color: var(--text-primary); }}
.detail-header .badges {{ display: flex; gap: 6px; }}
.badge {{ padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 600; }}
.badge-conf {{ background: #388bfd33; color: var(--accent-blue); }}
.badge-slot {{ background: #21262d; color: var(--text-muted); }}

.tab-bar {{ display: flex; border-bottom: 1px solid var(--border); background: var(--bg-primary); }}
.tab {{
  padding: 8px 16px;
  cursor: pointer;
  font-size: 12px;
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
}}
.tab:hover {{ color: var(--text-secondary); }}
.tab.active {{ color: var(--accent-blue); border-bottom-color: var(--accent-blue); }}
.tab-content {{ display: none; padding: 16px; }}
.tab-content.active {{ display: block; }}

.cont-preds {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; margin-bottom: 16px; }}
.cont-card {{ background: var(--bg-primary); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }}
.cont-card .metric-name {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 6px; }}
.cont-card .gt-row {{ display: flex; align-items: baseline; gap: 6px; margin-bottom: 4px; }}
.cont-card .gt-val {{ font-size: 18px; font-weight: 700; color: var(--text-primary); }}
.cont-card .gt-label {{ font-size: 10px; color: var(--text-muted); }}
.pred-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; margin: 2px 0; }}
.pred-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
.pred-val {{ font-weight: 600; }}
.pred-err {{ color: var(--text-muted); }}

.binary-section {{ margin-top: 8px; }}
.binary-section h4 {{ font-size: 12px; color: var(--text-muted); margin-bottom: 8px; font-weight: 500; }}
.binary-table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
.binary-table th {{ padding: 4px 8px; text-align: center; font-weight: 600; font-size: 10px; color: var(--text-muted); border-bottom: 1px solid var(--border); }}
.binary-table td {{ padding: 4px 8px; text-align: center; border-bottom: 1px solid #21262d; }}
.binary-table td:first-child {{ text-align: left; color: var(--text-secondary); }}
.b-match {{ color: var(--accent-green); font-weight: 600; }}
.b-miss {{ color: var(--accent-red); font-weight: 600; }}
.b-na {{ color: #484f58; }}

.process-section {{ border: 1px solid var(--border); border-radius: 6px; margin-bottom: 8px; overflow: hidden; }}
.process-header {{
  padding: 8px 12px;
  background: var(--bg-primary);
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: var(--text-secondary);
  user-select: none;
}}
.process-header:hover {{ background: var(--bg-tertiary); }}
.process-header .arrow {{ color: var(--text-muted); font-size: 10px; }}
.process-body {{
  display: none;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.6;
  max-height: 500px;
  overflow-y: auto;
  border-top: 1px solid var(--border);
}}
.process-body.open {{ display: block; }}
.process-body pre {{
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg-primary);
  padding: 10px;
  border-radius: 4px;
  font-size: 11px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  color: var(--text-secondary);
  margin: 4px 0;
}}
.tool-call {{ background: var(--bg-primary); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; margin: 4px 0; font-size: 11px; }}
.tool-call .tool-name {{ color: var(--accent-purple); font-weight: 600; }}
.tool-call .tool-input {{ color: var(--text-muted); font-family: monospace; font-size: 10px; }}
.rag-case {{ background: var(--bg-primary); border: 1px solid var(--border); border-radius: 4px; padding: 6px 10px; margin: 4px 0; font-size: 11px; }}
.rag-sim {{ color: var(--accent-green); font-weight: 600; font-size: 10px; }}

.raw-stats {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin-bottom: 12px; }}
.raw-stat {{ background: var(--bg-primary); border: 1px solid var(--border); border-radius: 4px; padding: 8px; text-align: center; }}
.raw-stat .rs-label {{ font-size: 10px; color: var(--text-muted); }}
.raw-stat .rs-val {{ font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 2px; }}

.version-comparison {{ background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; overflow: hidden; }}
.vc-header {{ padding: 10px 16px; border-bottom: 1px solid var(--border); font-size: 13px; color: var(--text-primary); font-weight: 600; }}
.vc-grid {{ display: grid; gap: 0; }}
.vc-card {{ padding: 12px 16px; border-bottom: 1px solid var(--border); display: grid; grid-template-columns: 130px 1fr 1fr 1fr 100px; align-items: center; gap: 8px; font-size: 12px; }}
.vc-card:last-child {{ border-bottom: none; }}
.vc-version {{ font-weight: 600; font-size: 12px; }}
.vc-val {{ text-align: center; }}
.vc-val .v {{ font-size: 14px; font-weight: 600; }}
.vc-val .l {{ font-size: 9px; color: var(--text-muted); display: block; }}
.vc-conf {{ text-align: right; color: var(--text-muted); font-size: 11px; }}

.ver-selector {{ display: flex; gap: 4px; }}
.ver-btn {{
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--bg-primary);
  color: var(--text-muted);
  cursor: pointer;
  font-size: 11px;
  transition: all 0.15s;
}}
.ver-btn:hover {{ background: var(--bg-tertiary); }}
.ver-btn.active {{ border-color: currentColor; font-weight: 600; }}

.welcome {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 60vh;
  color: var(--text-muted);
  text-align: center;
}}
.welcome h2 {{ font-size: 18px; color: var(--text-primary); margin-bottom: 8px; }}
.welcome p {{ max-width: 400px; line-height: 1.6; }}
.welcome .legend {{ display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap; justify-content: center; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}

/* Notice box */
.notice {{
  border-radius: 8px;
  padding: 14px 18px;
  margin-bottom: 20px;
  font-size: 12px;
  line-height: 1.6;
}}
.notice-info {{
  background: #58a6ff15;
  border: 1px solid #58a6ff55;
}}
.notice-info strong {{ color: var(--accent-blue); }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
</style>
</head>
<body>

<div class="header">
  <h1>Proactive Affective Agent</h1>
  <div class="nav-tabs">
    <div class="nav-tab active" onclick="switchPage('overview')">Overview</div>
    <div class="nav-tab" onclick="switchPage('methodology')">Methodology</div>
    <div class="nav-tab" onclick="switchPage('analysis')">Per-User Analysis</div>
  </div>
  <div class="header-stats" id="headerStats"></div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- OVERVIEW PAGE -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="page active" id="page-overview">
<div class="overview-content">

<h2>Pilot Study Results — 7 Agent Versions</h2>
<p style="color:var(--text-muted);margin-bottom:20px;line-height:1.6">
  Comparison of 7 LLM-based emotional state prediction approaches for cancer survivors
  using passive smartphone sensing and diary data. 5 pilot users, 427 EMA entries each.
</p>


<h3>Research Design Matrix</h3>
<table class="design-matrix">
  <tr>
    <th></th>
    <th>Structured (fixed pipeline)</th>
    <th>Agentic (autonomous tool-use)</th>
    <th>Agentic (filtered + tool-use)</th>
  </tr>
  <tr>
    <td class="row-header">Sensing only</td>
    <td><span class="matrix-cell"><span class="dot" style="background:#79c0ff"></span>V1</span></td>
    <td><span class="matrix-cell"><span class="dot" style="background:#d2a8ff"></span>V2</span></td>
    <td><span class="matrix-cell"><span class="dot" style="background:#56d4dd"></span>V5</span></td>
  </tr>
  <tr>
    <td class="row-header">Multimodal (+ diary)</td>
    <td><span class="matrix-cell"><span class="dot" style="background:#3fb950"></span>V3</span></td>
    <td><span class="matrix-cell"><span class="dot" style="background:#f0883e"></span>V4</span></td>
    <td><span class="matrix-cell"><span class="dot" style="background:#f778ba"></span>V6</span></td>
  </tr>
  <tr>
    <td class="row-header">Baseline (diary + RAG)</td>
    <td colspan="3"><span class="matrix-cell"><span class="dot" style="background:#f97583"></span>CALLM (CHI 2025 baseline)</span></td>
  </tr>
</table>

<h3>Aggregate Performance Comparison</h3>
<div class="metric-grid">
  <div class="metric-card">
    <h4>Balanced Accuracy (higher is better)</h4>
    <div class="chart-wrap"><canvas id="chartBA"></canvas></div>
  </div>
  <div class="metric-card">
    <h4>Mean Macro F1 (higher is better)</h4>
    <div class="chart-wrap"><canvas id="chartF1"></canvas></div>
  </div>
  <div class="metric-card">
    <h4>Avg Elapsed Time per Prediction</h4>
    <div class="chart-wrap"><canvas id="chartTime"></canvas></div>
  </div>
</div>

<h3>Detailed Results</h3>
<table class="results-table" id="resultsTable"></table>

<h3>All Binary Targets — Balanced Accuracy</h3>
<table class="results-table" id="binaryBATable"></table>

<h3>All Binary Targets — Macro F1</h3>
<table class="results-table" id="binaryF1Table"></table>

<h3>Token Consumption</h3>
<div class="notice-info notice">
  <strong>Note:</strong> Token data is only available for structured versions (CALLM, V1, V3) which report
  token counts directly. Agentic versions (V2, V4, V5, V6) do not report token counts.
  Elapsed time is shown as a proxy for computational cost.
</div>
<div id="tokenSection"></div>

</div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- METHODOLOGY PAGE -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="page" id="page-methodology">
<div class="method-content">

<h2>Methodology — Agent Architectures</h2>
<p style="color:var(--text-muted);margin-bottom:20px;line-height:1.6">
  Each version implements a different approach to predicting emotional states from
  smartphone sensing data. All versions use Claude Sonnet via Max subscription (free inference).
</p>

<h3>Data Pipeline Comparison</h3>
<div class="data-compare">
  <div class="data-box">
    <h4>
      Raw Hourly Data
      <span class="data-tag" style="background:#d2a8ff33;color:#d2a8ff">V2, V4</span>
    </h4>
    <div class="file-list">
      <span>data/processed/hourly/screen/</span> — screen on/off events<br>
      <span>data/processed/hourly/motion/</span> — accelerometer activity<br>
      <span>data/processed/hourly/keyinput/</span> — keyboard events<br>
      <span>data/processed/hourly/light/</span> — ambient light (Android only)<br>
      <span>data/processed/hourly/mus/</span> — app usage snapshots<br>
    </div>
    <pre>hour  screen_on  screen_off  unlock_count  total_min
0       2         2          1            12.3
1       0         0          0             0.0
2       0         0          0             0.0
...
23      5         5          3            45.7

[One file per user per day, ~24 rows each]
[Agent queries data autonomously, forms
 hypotheses from raw behavioral signals]</pre>
  </div>
  <div class="data-box">
    <h4>
      Filtered Daily Narratives
      <span class="data-tag" style="background:#56d4dd33;color:#56d4dd">V5, V6</span>
    </h4>
    <div class="file-list">
      <span>data/processed/filtered/{{pid}}_daily_filtered.parquet</span><br>
      One row per user per day with structured narrative
    </div>
    <pre>[Motion] Mostly sedentary indoor day.
  walk=15 min (z=-0.31), car=53 min (z=+5.67),
  stationary=1372 min.
  Anomaly: car travel 53 min unusually high.

[Screen] 12 screen opens, 0.0 min total.
  Zero recorded screen-on time suggests
  tracking gap or phone-off period.

[Apps] Top: Social-Networking=82 min,
  Entertainment=45 min, Productivity=23 min.
  Total app time 3.2 hr.

[Keyboard] No keyboard activity recorded.

[Environment] Light levels suggest indoor
  environment (avg 42 lux, dim conditions).</pre>
  </div>
</div>

<!-- CALLM -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#f97583">CALLM</span>
    <span class="v-title">Diary + TF-IDF RAG Baseline (CHI 2025)</span>
    <div class="v-tags">
      <span class="v-tag">structured</span>
      <span class="v-tag">diary-only</span>
      <span class="v-tag">single LLM call</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p>The <strong>CALLM</strong> baseline uses diary text as primary input. A TF-IDF retriever
      finds similar diary entries from other participants, and their emotional outcomes are
      provided as reference cases. The LLM makes predictions in a single call.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Diary text + trait profile + memory doc + RAG examples (with outcomes)</p>
      <p><strong>Inference:</strong> Single LLM call (structured output)</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Diary Text"] --> B["TF-IDF RAG"]
        B --> C["Claude Sonnet<br/>(single call)"]
        D["Trait Profile"] --> C
        E["Memory Doc"] --> C
        C --> F["Prediction"]
        style A fill:#f97583,color:#000
        style F fill:#3fb950,color:#000
        style C fill:#58a6ff,color:#000
      </div>
    </div>
  </div>
</div>

<!-- V1 -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#79c0ff;color:#000">V1</span>
    <span class="v-title">Structured Sensing-Only</span>
    <div class="v-tags">
      <span class="v-tag">structured</span>
      <span class="v-tag">sensing-only</span>
      <span class="v-tag">single LLM call</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p><strong>V1</strong> uses a fixed pipeline to extract features from hourly sensing data
      (screen, motion, keyboard, light). The day's behavioral patterns are summarized and
      formatted into a structured prompt for a single LLM call.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Sensing summary (fixed format) + trait profile + memory doc</p>
      <p><strong>Inference:</strong> Single LLM call (structured output)</p>
      <p><strong>No diary text</strong> — purely behavioral signal</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Hourly<br/>Parquet"] --> B["SensingDay<br/>Extractor"]
        B --> C["Format<br/>Summary"]
        C --> D["Claude Sonnet<br/>(single call)"]
        E["Trait Profile"] --> D
        D --> F["Prediction"]
        style A fill:#79c0ff,color:#000
        style F fill:#3fb950,color:#000
        style D fill:#58a6ff,color:#000
      </div>
    </div>
  </div>
</div>

<!-- V2 -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#d2a8ff;color:#000">V2</span>
    <span class="v-title">Agentic Sensing-Only</span>
    <div class="v-tags">
      <span class="v-tag">agentic</span>
      <span class="v-tag">sensing-only</span>
      <span class="v-tag">multi-turn tool-use</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p><strong>V2</strong> gives the LLM autonomous access to raw sensing data via tool-use.
      The agent decides which data to query, forms hypotheses, and iteratively explores
      the data before making predictions.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Sensing tools + trait profile</p>
      <p><strong>Inference:</strong> Multi-turn agentic loop (up to 16 tool calls)</p>
      <p><strong>No diary text</strong> — agent must find signals autonomously from raw data</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Claude Sonnet<br/>(autonomous)"] -->|"query tools"| B["Sensing<br/>Tools"]
        B -->|"raw data"| A
        C["Trait Profile"] --> A
        A --> D["Prediction"]
        style A fill:#d2a8ff,color:#000
        style B fill:#f0883e,color:#000
        style D fill:#3fb950,color:#000
      </div>
      <details class="tool-details">
        <summary>Available sensing tools (click to expand)</summary>
        <ul class="tool-list">
          <li><strong>query_sensing</strong> — Query raw sensor data (screen, motion, keyboard, light, apps) for a time window</li>
          <li><strong>get_daily_summary</strong> — Natural language summary of a full day's behavioral patterns</li>
          <li><strong>compare_to_baseline</strong> — Compare a sensor reading to the person's historical baseline</li>
          <li><strong>get_receptivity_history</strong> — Past intervention receptivity and mood patterns</li>
          <li><strong>find_similar_days</strong> — Find past days with similar behavioral patterns and mood outcomes</li>
          <li><strong>query_raw_events</strong> — Query raw event streams (app launches, screen events, etc.)</li>
        </ul>
      </details>
    </div>
  </div>
</div>

<!-- V3 -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#3fb950;color:#000">V3</span>
    <span class="v-title">Structured Multimodal (Sensing + Diary)</span>
    <div class="v-tags">
      <span class="v-tag">structured</span>
      <span class="v-tag">multimodal</span>
      <span class="v-tag">single LLM call</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p><strong>V3</strong> combines diary text with structured sensing data in a fixed pipeline.
      Uses MultiModal RAG to find similar cases (diary + sensing patterns), then makes
      predictions in a single call.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Diary + sensing summary + multimodal RAG + trait profile</p>
      <p><strong>Inference:</strong> Single LLM call (structured output)</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Diary Text"] --> B["MultiModal<br/>RAG"]
        G["Sensing<br/>Summary"] --> B
        B --> C["Claude Sonnet<br/>(single call)"]
        D["Trait Profile"] --> C
        G --> C
        A --> C
        C --> F["Prediction"]
        style A fill:#3fb950,color:#000
        style G fill:#79c0ff,color:#000
        style F fill:#3fb950,color:#000
        style C fill:#58a6ff,color:#000
      </div>
    </div>
  </div>
</div>

<!-- V4 -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#f0883e;color:#000">V4</span>
    <span class="v-title">Agentic Multimodal (Sensing + Diary)</span>
    <div class="v-tags">
      <span class="v-tag">agentic</span>
      <span class="v-tag">multimodal</span>
      <span class="v-tag">multi-turn tool-use</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p><strong>V4</strong> combines autonomous tool-use with diary text. The agent reads the diary
      first, forms hypotheses about the person's emotional state, then selectively queries
      raw sensing data to validate and refine predictions.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Diary text + sensing tools + session memory + trait profile</p>
      <p><strong>Inference:</strong> Multi-turn agentic loop (up to 16 tool calls)</p>
      <p><strong>Session memory:</strong> accumulates receptivity feedback across entries</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Claude Sonnet<br/>(autonomous)"] -->|"query tools"| B["Sensing<br/>Tools"]
        B -->|"raw data"| A
        C["Diary Text"] --> A
        D["Session Memory"] --> A
        E["Trait Profile"] --> A
        A --> F["Prediction"]
        style A fill:#f0883e,color:#000
        style B fill:#d2a8ff,color:#000
        style C fill:#3fb950,color:#000
        style F fill:#3fb950,color:#000
      </div>
      <details class="tool-details">
        <summary>Available sensing tools (click to expand)</summary>
        <ul class="tool-list">
          <li><strong>query_sensing</strong> — Query raw sensor data (screen, motion, keyboard, light, apps) for a time window</li>
          <li><strong>get_daily_summary</strong> — Natural language summary of a full day's behavioral patterns</li>
          <li><strong>compare_to_baseline</strong> — Compare a sensor reading to the person's historical baseline</li>
          <li><strong>get_receptivity_history</strong> — Past intervention receptivity and mood patterns</li>
          <li><strong>find_similar_days</strong> — Find past days with similar behavioral patterns and mood outcomes</li>
          <li><strong>query_raw_events</strong> — Query raw event streams (app launches, screen events, etc.)</li>
        </ul>
      </details>
    </div>
  </div>
</div>

<!-- V5 -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#56d4dd;color:#000">V5</span>
    <span class="v-title">Agentic Filtered Sensing-Only</span>
    <div class="v-tags">
      <span class="v-tag">agentic</span>
      <span class="v-tag">filtered narrative</span>
      <span class="v-tag">sensing-only</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p><strong>V5</strong> provides a pre-computed filtered behavioral narrative as primary context.
      The narrative includes structured sections (motion, screen, apps, keyboard, environment)
      with anomaly detection and z-scores. The agent starts with this rich summary and can
      drill down into raw data when needed.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Filtered narrative + sensing tools (for drill-down) + trait profile</p>
      <p><strong>Inference:</strong> Multi-turn agentic loop (up to 16 tool calls)</p>
      <p><strong>No diary text</strong> — narrative provides comprehensive behavioral context</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Filtered<br/>Narrative"] --> B["Claude Sonnet<br/>(autonomous)"]
        B -->|"drill-down"| C["Sensing<br/>Tools"]
        C -->|"raw details"| B
        D["Session Memory"] --> B
        E["Trait Profile"] --> B
        B --> F["Prediction"]
        style A fill:#56d4dd,color:#000
        style B fill:#56d4dd,color:#000
        style C fill:#d2a8ff,color:#000
        style F fill:#3fb950,color:#000
      </div>
      <details class="tool-details">
        <summary>Available sensing tools (click to expand)</summary>
        <ul class="tool-list">
          <li><strong>query_sensing</strong> — Query raw sensor data (screen, motion, keyboard, light, apps) for a time window</li>
          <li><strong>get_daily_summary</strong> — Natural language summary of a full day's behavioral patterns</li>
          <li><strong>compare_to_baseline</strong> — Compare a sensor reading to the person's historical baseline</li>
          <li><strong>get_receptivity_history</strong> — Past intervention receptivity and mood patterns</li>
          <li><strong>find_similar_days</strong> — Find past days with similar behavioral patterns and mood outcomes</li>
          <li><strong>query_raw_events</strong> — Query raw event streams (app launches, screen events, etc.)</li>
        </ul>
      </details>
    </div>
  </div>
</div>

<!-- V6 -->
<div class="version-card">
  <div class="version-card-header">
    <span class="v-badge" style="background:#f778ba;color:#000">V6</span>
    <span class="v-title">Agentic Filtered Multimodal (Best)</span>
    <div class="v-tags">
      <span class="v-tag">agentic</span>
      <span class="v-tag">filtered narrative</span>
      <span class="v-tag">multimodal</span>
      <span class="v-tag" style="border-color:var(--accent-green);color:var(--accent-green)">BEST</span>
    </div>
  </div>
  <div class="version-card-body">
    <div class="v-desc">
      <p><strong>V6</strong> combines all modalities: diary text + filtered behavioral narrative + tool access.
      The agent reads the diary first for emotional context, then uses the pre-computed narrative
      to understand behavioral patterns, and can drill into raw data for specific details.</p>
      <p style="margin-top:8px"><strong>Input:</strong> Diary text + filtered narrative + sensing tools + session memory + trait profile</p>
      <p><strong>Inference:</strong> Multi-turn agentic loop (up to 16 tool calls)</p>
      <p style="color:var(--accent-green)"><strong>Best overall:</strong> BA=0.676, F1=0.558 (highest binary classification)</p>
    </div>
    <div class="v-flow">
      <div class="mermaid">
      graph LR
        A["Diary Text"] --> B["Claude Sonnet<br/>(autonomous)"]
        G["Filtered<br/>Narrative"] --> B
        B -->|"drill-down"| C["Sensing<br/>Tools"]
        C -->|"raw details"| B
        D["Session Memory"] --> B
        E["Trait Profile"] --> B
        B --> F["Prediction"]
        style A fill:#f778ba,color:#000
        style G fill:#56d4dd,color:#000
        style B fill:#f778ba,color:#000
        style C fill:#d2a8ff,color:#000
        style F fill:#3fb950,color:#000
      </div>
      <details class="tool-details">
        <summary>Available sensing tools (click to expand)</summary>
        <ul class="tool-list">
          <li><strong>query_sensing</strong> — Query raw sensor data (screen, motion, keyboard, light, apps) for a time window</li>
          <li><strong>get_daily_summary</strong> — Natural language summary of a full day's behavioral patterns</li>
          <li><strong>compare_to_baseline</strong> — Compare a sensor reading to the person's historical baseline</li>
          <li><strong>get_receptivity_history</strong> — Past intervention receptivity and mood patterns</li>
          <li><strong>find_similar_days</strong> — Find past days with similar behavioral patterns and mood outcomes</li>
          <li><strong>query_raw_events</strong> — Query raw event streams (app launches, screen events, etc.)</li>
        </ul>
      </details>
    </div>
  </div>
</div>

<h3>Backend Architecture</h3>
<div class="data-compare">
  <div class="data-box">
    <h4>
      Structured Versions
      <span class="data-tag" style="background:#58a6ff33;color:#58a6ff">CALLM, V1, V3</span>
    </h4>
    <pre>Fixed Data Pipeline
  |
  +-- Extract features from sensing data
  |     (screen, motion, keyboard, light)
  |
  +-- Format structured prompt
  |     (sensing summary + diary + RAG examples)
  |
  +-- Single LLM Call (Claude Sonnet)
  |     → Returns structured prediction
  |
  No autonomous decision-making
  — same pipeline for every input</pre>
  </div>
  <div class="data-box">
    <h4>
      Agentic Versions
      <span class="data-tag" style="background:#d2a8ff33;color:#d2a8ff">V2, V4, V5, V6</span>
    </h4>
    <pre>Agentic Loop (up to 16 turns)
  |
  +-- Claude Sonnet (autonomous agent)
  |     |
  |     +-- 1. Reads context (narrative/diary)
  |     +-- 2. Decides which data to query
  |     +-- 3. Calls sensing tools iteratively
  |     +-- 4. Reasons about behavioral patterns
  |     +-- 5. Returns prediction
  |
  +-- Sensing Tools (6 tools available)
        query_sensing, get_daily_summary,
        compare_to_baseline, get_receptivity_history,
        find_similar_days, query_raw_events</pre>
  </div>
</div>

</div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- PER-USER ANALYSIS PAGE -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="page-analysis" id="page-analysis">
  <div class="sidebar">
    <div class="sidebar-filters">
      <div class="filter-row">
        <label>Version</label>
        <select id="filterVersion" onchange="applyFilters()">
          <option value="all">All versions</option>
        </select>
      </div>
      <div class="filter-row">
        <label>Slot</label>
        <select id="filterSlot" onchange="applyFilters()">
          <option value="all">All slots</option>
          <option value="morning">Morning</option>
          <option value="afternoon">Afternoon</option>
          <option value="evening">Evening</option>
          <option value="unknown">Unknown</option>
        </select>
      </div>
    </div>
    <div class="user-list" id="userList"></div>
  </div>
  <div class="main" id="main">
    <div class="welcome">
      <h2>Select a user to get started</h2>
      <p>Choose a user from the sidebar to view their time series, entry details, and version comparisons.</p>
      <div class="legend" id="legend"></div>
    </div>
  </div>
</div>

<script>
// ── Data ─────────────────────────────────────────────────────────────────
const DATA = {data_json};
const VERSIONS = {versions_json};
const VERSION_LABELS = {labels_json};
const VERSION_COLORS = {colors_json};
const BINARY_STATES = {binary_json};
const EVAL_DATA = {eval_json};
const TOKEN_STATS = {token_json};
const ELAPSED_STATS = {elapsed_json};
const USERS = Object.keys(DATA).map(Number).sort((a,b) => a - b);

let activeUser = null;
let activeEntryIdx = null;
let activeVersion = null;
let activeMetric = 'PANAS_Pos';
let activeTab = 'predictions';
let timeChart = null;
let currentPage = 'overview';

// ── Page navigation ──────────────────────────────────────────────────────
function switchPage(page) {{
  currentPage = page;
  document.querySelectorAll('.nav-tab').forEach((t, i) => {{
    const pages = ['overview', 'methodology', 'analysis'];
    t.classList.toggle('active', pages[i] === page);
  }});
  document.getElementById('page-overview').classList.toggle('active', page === 'overview');
  document.getElementById('page-methodology').classList.toggle('active', page === 'methodology');
  document.getElementById('page-analysis').classList.toggle('active', page === 'analysis');

  // Re-init mermaid when methodology page shown
  if (page === 'methodology') {{
    setTimeout(() => mermaid.run(), 100);
  }}
}}

// ── Init ────────────────────────────────────────────────────────────────
function init() {{
  // Mermaid dark theme
  mermaid.initialize({{
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {{
      primaryColor: '#1c2128',
      primaryTextColor: '#f0f6fc',
      primaryBorderColor: '#30363d',
      lineColor: '#8b949e',
      secondaryColor: '#161b22',
      tertiaryColor: '#21262d',
    }}
  }});

  // Populate version filter
  const vSel = document.getElementById('filterVersion');
  VERSIONS.forEach(v => {{
    const opt = document.createElement('option');
    opt.value = v; opt.textContent = VERSION_LABELS[v];
    vSel.appendChild(opt);
  }});

  // Build legend
  const legend = document.getElementById('legend');
  legend.innerHTML = '<div class="legend-item"><div class="legend-swatch" style="background:#f0f6fc"></div>Ground Truth</div>' +
    VERSIONS.map(v => `<div class="legend-item"><div class="legend-swatch" style="background:${{VERSION_COLORS[v]}}"></div>${{VERSION_LABELS[v]}}</div>`).join('');

  // Header stats
  let totalRecords = 0, totalEntries = 0;
  const allVersions = new Set();
  USERS.forEach(uid => {{
    const entries = DATA[uid];
    totalEntries += entries.length;
    entries.forEach(e => {{
      Object.keys(e.versions).forEach(v => {{ allVersions.add(v); totalRecords++; }});
    }});
  }});
  document.getElementById('headerStats').innerHTML =
    `<div class="header-stat"><span class="num">${{USERS.length}}</span> users</div>` +
    `<div class="header-stat"><span class="num">${{totalEntries}}</span> entries</div>` +
    `<div class="header-stat"><span class="num">${{allVersions.size}}</span> versions</div>`;

  renderUserList();
  renderOverviewCharts();
  renderResultsTables();
  renderTokenSection();
}}

// ── Overview Charts ─────────────────────────────────────────────────────
function renderOverviewCharts() {{
  const versions = VERSIONS.filter(v => EVAL_DATA[v]?.available);
  const labels = versions.map(v => VERSION_LABELS[v]);
  const colors = versions.map(v => VERSION_COLORS[v]);

  // BA chart
  new Chart(document.getElementById('chartBA'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: versions.map(v => EVAL_DATA[v]?.aggregate?.mean_ba ?? 0),
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors,
        borderWidth: 1,
      }}]
    }},
    options: barOpts('BA', false, [0.4, 0.75])
  }});

  // F1 chart
  new Chart(document.getElementById('chartF1'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: versions.map(v => EVAL_DATA[v]?.aggregate?.mean_f1 ?? 0),
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors,
        borderWidth: 1,
      }}]
    }},
    options: barOpts('Macro F1', false, [0.2, 0.65])
  }});

  // Elapsed time chart
  const elapsedData = versions.map(v => {{
    const s = ELAPSED_STATS[v];
    if (s && s.count > 0) return s.total / s.count;
    return 0;
  }});
  new Chart(document.getElementById('chartTime'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        data: elapsedData,
        backgroundColor: colors.map(c => c + 'bb'),
        borderColor: colors,
        borderWidth: 1,
      }}]
    }},
    options: barOpts('Seconds', true)
  }});
}}

function barOpts(yLabel, lowerBetter, suggestedRange) {{
  return {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#1c2128',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#f0f6fc',
        bodyColor: '#c9d1d9',
        callbacks: {{
          label: ctx => `${{yLabel}}: ${{ctx.parsed.y.toFixed(3)}}`
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
      y: {{
        ticks: {{ color: '#8b949e', font: {{ size: 10 }} }},
        grid: {{ color: '#21262d' }},
        suggestedMin: suggestedRange ? suggestedRange[0] : undefined,
        suggestedMax: suggestedRange ? suggestedRange[1] : undefined,
      }}
    }}
  }};
}}

// ── Results Tables ──────────────────────────────────────────────────────
function renderResultsTables() {{
  const versions = VERSIONS.filter(v => EVAL_DATA[v]?.available);

  // Aggregate results
  let html = '<tr><th>Metric</th>';
  versions.forEach(v => html += `<th style="color:${{VERSION_COLORS[v]}}">${{VERSION_LABELS[v]}}</th>`);
  html += '</tr>';

  // Find bests
  const metrics = ['mean_ba', 'mean_f1'];
  const metricLabels = ['Mean BA', 'Mean Macro F1'];
  const lowerBetter = [false, false];

  metrics.forEach((m, i) => {{
    const vals = versions.map(v => EVAL_DATA[v]?.aggregate?.[m] ?? null);
    const best = lowerBetter[i]
      ? Math.min(...vals.filter(x => x !== null))
      : Math.max(...vals.filter(x => x !== null));
    html += `<tr><td>${{metricLabels[i]}}</td>`;
    vals.forEach(v => {{
      if (v === null) {{ html += '<td>-</td>'; return; }}
      const cls = Math.abs(v - best) < 0.001 ? 'best-val' : '';
      html += `<td class="${{cls}}">${{v.toFixed(3)}}</td>`;
    }});
    html += '</tr>';
  }});

  // N entries
  html += '<tr><td>N entries</td>';
  versions.forEach(v => html += `<td>${{EVAL_DATA[v]?.n_entries ?? '-'}}</td>`);
  html += '</tr>';

  document.getElementById('resultsTable').innerHTML = html;

  // All binary targets — BA table
  const allBinTargets = [
    'Individual_level_PA_State', 'Individual_level_NA_State',
    'Individual_level_happy_State', 'Individual_level_sad_State',
    'Individual_level_afraid_State', 'Individual_level_miserable_State',
    'Individual_level_worried_State', 'Individual_level_cheerful_State',
    'Individual_level_pleased_State', 'Individual_level_grateful_State',
    'Individual_level_lonely_State', 'Individual_level_interactions_quality_State',
    'Individual_level_pain_State', 'Individual_level_forecasting_State',
    'Individual_level_ER_desire_State', 'INT_availability',
  ];
  html = '<tr><th>Target</th>';
  versions.forEach(v => html += `<th style="color:${{VERSION_COLORS[v]}}">${{VERSION_LABELS[v]}}</th>`);
  html += '</tr>';

  allBinTargets.forEach(t => {{
    const vals = versions.map(v => EVAL_DATA[v]?.binary?.[t]?.ba ?? null);
    const best = Math.max(...vals.filter(x => x !== null));
    const label = t.replace('Individual_level_', '').replace('_State', '');
    html += `<tr><td>${{label}}</td>`;
    vals.forEach(v => {{
      if (v === null) {{ html += '<td>-</td>'; return; }}
      const cls = Math.abs(v - best) < 0.001 ? 'best-val' : '';
      html += `<td class="${{cls}}">${{v.toFixed(3)}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '<tr style="border-top:2px solid var(--border);font-weight:600"><td>Mean BA</td>';
  versions.forEach(v => {{
    const agg = EVAL_DATA[v]?.aggregate?.mean_ba;
    const allBAs = versions.map(vv => EVAL_DATA[vv]?.aggregate?.mean_ba ?? 0);
    const best = Math.max(...allBAs);
    const cls = agg && Math.abs(agg - best) < 0.001 ? 'best-val' : '';
    html += agg != null ? `<td class="${{cls}}">${{agg.toFixed(3)}}</td>` : '<td>-</td>';
  }});
  html += '</tr>';
  document.getElementById('binaryBATable').innerHTML = html;

  // All binary targets — Macro F1 table
  html = '<tr><th>Target</th>';
  versions.forEach(v => html += `<th style="color:${{VERSION_COLORS[v]}}">${{VERSION_LABELS[v]}}</th>`);
  html += '</tr>';

  allBinTargets.forEach(t => {{
    const vals = versions.map(v => EVAL_DATA[v]?.binary?.[t]?.f1 ?? null);
    const best = Math.max(...vals.filter(x => x !== null));
    const label = t.replace('Individual_level_', '').replace('_State', '');
    html += `<tr><td>${{label}}</td>`;
    vals.forEach(v => {{
      if (v === null) {{ html += '<td>-</td>'; return; }}
      const cls = Math.abs(v - best) < 0.001 ? 'best-val' : '';
      html += `<td class="${{cls}}">${{v.toFixed(3)}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '<tr style="border-top:2px solid var(--border);font-weight:600"><td>Mean Macro F1</td>';
  versions.forEach(v => {{
    const agg = EVAL_DATA[v]?.aggregate?.mean_f1;
    const allF1s = versions.map(vv => EVAL_DATA[vv]?.aggregate?.mean_f1 ?? 0);
    const best = Math.max(...allF1s);
    const cls = agg && Math.abs(agg - best) < 0.001 ? 'best-val' : '';
    html += agg != null ? `<td class="${{cls}}">${{agg.toFixed(3)}}</td>` : '<td>-</td>';
  }});
  html += '</tr>';
  document.getElementById('binaryF1Table').innerHTML = html;
}}

// ── Token Section ───────────────────────────────────────────────────────
function renderTokenSection() {{
  const container = document.getElementById('tokenSection');
  const versions = ['callm', 'v1', 'v3'];  // only structured have token data

  let html = '<div class="token-grid">';
  versions.forEach(v => {{
    const s = TOKEN_STATS[v];
    if (!s) return;
    const avgIn = Math.round(s.input / s.count);
    const avgOut = Math.round(s.output / s.count);
    const avgTotal = avgIn + avgOut;

    html += `
      <div class="token-stat" style="border-color:${{VERSION_COLORS[v]}}55">
        <div class="ts-label" style="color:${{VERSION_COLORS[v]}}">${{VERSION_LABELS[v]}}</div>
        <div class="ts-val">${{(avgTotal / 1000).toFixed(1)}}K</div>
        <div class="ts-sub">avg tokens/pred</div>
      </div>
      <div class="token-stat">
        <div class="ts-label">Input</div>
        <div class="ts-val">${{(avgIn / 1000).toFixed(1)}}K</div>
        <div class="ts-sub">avg input tokens</div>
      </div>
      <div class="token-stat">
        <div class="ts-label">Output</div>
        <div class="ts-val">${{(avgOut / 1000).toFixed(1)}}K</div>
        <div class="ts-sub">avg output tokens</div>
      </div>
    `;
  }});
  html += '</div>';

  // Elapsed time comparison for all versions
  html += '<h4 style="font-size:12px;color:var(--text-muted);margin:16px 0 8px;text-transform:uppercase;letter-spacing:0.3px">Avg Elapsed Time per Prediction (all versions)</h4>';
  html += '<div class="token-grid">';
  VERSIONS.forEach(v => {{
    const s = ELAPSED_STATS[v];
    if (!s || s.count === 0) return;
    const avg = (s.total / s.count).toFixed(1);
    html += `
      <div class="token-stat" style="border-color:${{VERSION_COLORS[v]}}55">
        <div class="ts-label" style="color:${{VERSION_COLORS[v]}}">${{VERSION_LABELS[v]}}</div>
        <div class="ts-val">${{avg}}s</div>
        <div class="ts-sub">${{s.count}} samples, max ${{s.max.toFixed(0)}}s</div>
      </div>
    `;
  }});
  html += '</div>';

  container.innerHTML = html;
}}

// ── User list ───────────────────────────────────────────────────────────
function renderUserList() {{
  const list = document.getElementById('userList');
  list.innerHTML = '';
  const fVer = document.getElementById('filterVersion').value;

  USERS.forEach(uid => {{
    const entries = DATA[uid];
    const vers = new Set();
    entries.forEach(e => Object.keys(e.versions).forEach(v => vers.add(v)));

    if (fVer !== 'all' && !vers.has(fVer)) return;

    const count = fVer === 'all' ? entries.length
      : entries.filter(e => e.versions[fVer]).length;

    const div = document.createElement('div');
    div.className = 'user-item' + (uid === activeUser ? ' active' : '');
    div.onclick = () => selectUser(uid);
    div.innerHTML = `
      <div><span class="uid">User ${{uid}}</span></div>
      <div class="meta">
        ${{count}} entries
        <div class="ver-dots">
          ${{VERSIONS.map(v => vers.has(v) ? `<div class="ver-dot" style="background:${{VERSION_COLORS[v]}}" title="${{VERSION_LABELS[v]}}"></div>` : '').join('')}}
        </div>
      </div>`;
    list.appendChild(div);
  }});
}}

function applyFilters() {{
  renderUserList();
  if (activeUser) selectUser(activeUser);
}}

// ── Select user ─────────────────────────────────────────────────────────
function selectUser(uid) {{
  activeUser = uid;
  activeEntryIdx = null;
  renderUserList();
  renderUserView(uid);
}}

function renderUserView(uid) {{
  const main = document.getElementById('main');
  const entries = getFilteredEntries(uid);
  const vers = new Set();
  entries.forEach(e => Object.keys(e.versions).forEach(v => vers.add(v)));

  let html = '';

  html += `<div class="user-summary">
    <h2>User ${{uid}}</h2>
    <div class="stats">
      <div><span class="stat-val">${{entries.length}}</span> entries</div>
      <div><span class="stat-val">${{[...vers].map(v => VERSION_LABELS[v]).join(', ')}}</span></div>
    </div>
  </div>`;

  html += `<div class="chart-container">
    <div class="chart-controls">
      ${{['PANAS_Pos', 'PANAS_Neg', 'ER_desire'].map(m =>
        `<div class="chart-btn ${{m === activeMetric ? 'active' : ''}}" onclick="switchMetric('${{m}}')">${{m.replace('_', ' ')}}</div>`
      ).join('')}}
    </div>
    <div class="chart-wrap"><canvas id="timeChart"></canvas></div>
  </div>`;

  html += `<div class="timeline-strip">
    <h3>Entry Timeline (click to inspect)</h3>
    <div class="timeline-dots">
      ${{entries.map(e => {{
        const slot = e.slot || 'unknown';
        const nVers = Object.keys(e.versions).length;
        return `<div class="t-dot slot-${{slot}} ${{e.idx === activeEntryIdx ? 'active' : ''}}"
          onclick="selectEntry(${{uid}}, ${{e.idx}})" title="Entry ${{e.idx}}">
          <div class="t-dot-tip">#${{e.idx}} ${{e.date}} ${{slot}} (${{nVers}}v)</div>
        </div>`;
      }}).join('')}}
    </div>
  </div>`;

  html += '<div id="entryDetail"></div>';
  main.innerHTML = html;
  renderTimeChart(uid, entries);

  if (activeEntryIdx !== null) {{
    const entry = entries.find(e => e.idx === activeEntryIdx);
    if (entry) renderEntryDetail(uid, entry);
  }}
}}

// ── Time series chart ───────────────────────────────────────────────────
function renderTimeChart(uid, entries) {{
  const canvas = document.getElementById('timeChart');
  if (!canvas) return;
  if (timeChart) {{ timeChart.destroy(); timeChart = null; }}

  const labels = entries.map(e => e.date || `#${{e.idx}}`);
  const datasets = [];

  datasets.push({{
    label: 'Ground Truth',
    data: entries.map(e => e.ground_truth?.[activeMetric] ?? null),
    borderColor: '#f0f6fc',
    backgroundColor: '#f0f6fc33',
    borderWidth: 2,
    pointRadius: 3,
    pointHoverRadius: 5,
    tension: 0.3,
    spanGaps: true,
  }});

  const fVer = document.getElementById('filterVersion').value;
  const versionsToShow = fVer === 'all' ? VERSIONS : [fVer];

  versionsToShow.forEach(v => {{
    const data = entries.map(e => {{
      const pred = e.versions[v]?.prediction;
      return pred?.[activeMetric] ?? null;
    }});
    if (data.some(d => d !== null)) {{
      datasets.push({{
        label: VERSION_LABELS[v],
        data,
        borderColor: VERSION_COLORS[v],
        backgroundColor: VERSION_COLORS[v] + '33',
        borderWidth: 1.5,
        pointRadius: 2,
        pointHoverRadius: 4,
        tension: 0.3,
        spanGaps: true,
      }});
    }}
  }});

  timeChart = new Chart(canvas, {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      interaction: {{ mode: 'index', intersect: false }},
      onClick: (evt, elements) => {{
        if (elements.length > 0) {{
          const idx = elements[0].index;
          const entry = entries[idx];
          if (entry) selectEntry(uid, entry.idx);
        }}
      }},
      plugins: {{
        legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }}, boxWidth: 12 }} }},
        tooltip: {{
          backgroundColor: '#1c2128',
          borderColor: '#30363d',
          borderWidth: 1,
          titleColor: '#f0f6fc',
          bodyColor: '#c9d1d9',
          bodyFont: {{ size: 11 }},
        }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }}, maxRotation: 45 }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ color: '#21262d' }} }}
      }}
    }}
  }});
}}

function switchMetric(m) {{
  activeMetric = m;
  if (activeUser) renderUserView(activeUser);
}}

// ── Select entry ────────────────────────────────────────────────────────
function selectEntry(uid, idx) {{
  activeEntryIdx = idx;
  const entries = getFilteredEntries(uid);
  const entry = entries.find(e => e.idx === idx);
  if (!entry) return;

  document.querySelectorAll('.t-dot').forEach(d => d.classList.remove('active'));
  const dot = document.querySelector(`.t-dot[onclick="selectEntry(${{uid}}, ${{idx}})"]`);
  if (dot) dot.classList.add('active');

  const availVers = Object.keys(entry.versions);
  if (!activeVersion || !availVers.includes(activeVersion)) {{
    activeVersion = availVers[0] || VERSIONS[0];
  }}

  renderEntryDetail(uid, entry);
}}

function renderEntryDetail(uid, entry) {{
  const container = document.getElementById('entryDetail');
  if (!container) return;

  const availVers = Object.keys(entry.versions).sort((a,b) => VERSIONS.indexOf(a) - VERSIONS.indexOf(b));
  const vd = entry.versions[activeVersion] || {{}};
  const gt = entry.ground_truth || {{}};

  let html = '';

  html += `<div class="version-comparison">
    <div class="vc-header">Version Comparison — Entry #${{entry.idx}} (${{entry.date}} ${{entry.slot}})</div>
    <div class="vc-grid">
      <div class="vc-card" style="background:var(--bg-tertiary);">
        <div class="vc-version" style="color:var(--text-muted)">Version</div>
        <div class="vc-val"><span class="l">PANAS Pos</span></div>
        <div class="vc-val"><span class="l">PANAS Neg</span></div>
        <div class="vc-val"><span class="l">ER desire</span></div>
        <div class="vc-conf">Confidence</div>
      </div>
      <div class="vc-card">
        <div class="vc-version" style="color:var(--text-primary)">Ground Truth</div>
        <div class="vc-val"><span class="v">${{fmt(gt.PANAS_Pos)}}</span></div>
        <div class="vc-val"><span class="v">${{fmt(gt.PANAS_Neg)}}</span></div>
        <div class="vc-val"><span class="v">${{fmt(gt.ER_desire)}}</span></div>
        <div class="vc-conf">—</div>
      </div>
      ${{availVers.map(v => {{
        const p = entry.versions[v]?.prediction || {{}};
        const c = entry.versions[v]?.confidence;
        return `<div class="vc-card">
          <div class="vc-version" style="color:${{VERSION_COLORS[v]}}">${{VERSION_LABELS[v]}}</div>
          <div class="vc-val"><span class="v" style="color:${{errColor(p.PANAS_Pos, gt.PANAS_Pos)}}">${{fmt(p.PANAS_Pos)}}</span> ${{errBadge(p.PANAS_Pos, gt.PANAS_Pos)}}</div>
          <div class="vc-val"><span class="v" style="color:${{errColor(p.PANAS_Neg, gt.PANAS_Neg)}}">${{fmt(p.PANAS_Neg)}}</span> ${{errBadge(p.PANAS_Neg, gt.PANAS_Neg)}}</div>
          <div class="vc-val"><span class="v" style="color:${{errColor(p.ER_desire, gt.ER_desire)}}">${{fmt(p.ER_desire)}}</span> ${{errBadge(p.ER_desire, gt.ER_desire)}}</div>
          <div class="vc-conf">${{c != null ? (c * 100).toFixed(0) + '%' : '—'}}</div>
        </div>`;
      }}).join('')}}
    </div>
  </div>`;

  html += `<div class="detail-panel">
    <div class="detail-header">
      <h3>Entry #${{entry.idx}} Detail</h3>
      <div class="badges">
        <span class="badge badge-slot">${{entry.slot}}</span>
        ${{vd.model ? `<span class="badge badge-conf">${{vd.model}}</span>` : ''}}
      </div>
    </div>

    <div style="padding: 8px 16px; border-bottom: 1px solid var(--border); display:flex; justify-content:space-between; align-items:center;">
      <div class="ver-selector">
        ${{availVers.map(v =>
          `<div class="ver-btn ${{v === activeVersion ? 'active' : ''}}" style="color:${{VERSION_COLORS[v]}}" onclick="switchDetailVersion('${{v}}')">${{VERSION_LABELS[v]}}</div>`
        ).join('')}}
      </div>
    </div>

    <div class="tab-bar">
      <div class="tab ${{activeTab === 'predictions' ? 'active' : ''}}" onclick="switchTab('predictions')">Predictions</div>
      <div class="tab ${{activeTab === 'process' ? 'active' : ''}}" onclick="switchTab('process')">Process</div>
      <div class="tab ${{activeTab === 'raw' ? 'active' : ''}}" onclick="switchTab('raw')">Raw</div>
    </div>

    ${{renderPredictionsTab(entry, activeVersion)}}
    ${{renderProcessTab(entry, activeVersion)}}
    ${{renderRawTab(entry, activeVersion)}}
  </div>`;

  container.innerHTML = html;
}}

// ── Predictions tab ─────────────────────────────────────────────────────
function renderPredictionsTab(entry, version) {{
  const vd = entry.versions[version] || {{}};
  const pred = vd.prediction || {{}};
  const gt = entry.ground_truth || {{}};

  let html = `<div class="tab-content ${{activeTab === 'predictions' ? 'active' : ''}}" id="tab-predictions">`;

  html += '<div class="cont-preds">';
  ['PANAS_Pos', 'PANAS_Neg', 'ER_desire'].forEach(m => {{
    const gv = gt[m]; const pv = pred[m];
    const err = (gv != null && pv != null) ? Math.abs(gv - pv).toFixed(1) : '—';
    html += `<div class="cont-card">
      <div class="metric-name">${{m.replace('_', ' ')}}</div>
      <div class="gt-row"><span class="gt-val">${{fmt(gv)}}</span><span class="gt-label">ground truth</span></div>
      <div class="pred-row">
        <div class="pred-dot" style="background:${{VERSION_COLORS[version]}}"></div>
        <span class="pred-val" style="color:${{VERSION_COLORS[version]}}">${{fmt(pv)}}</span>
        <span class="pred-err">err: ${{err}}</span>
      </div>
    </div>`;
  }});
  html += '</div>';

  html += `<div class="binary-section">
    <h4>Binary Emotional States</h4>
    <table class="binary-table">
      <tr><th style="text-align:left">State</th><th>Truth</th><th style="color:${{VERSION_COLORS[version]}}">${{VERSION_LABELS[version]}}</th><th>Match</th></tr>`;

  BINARY_STATES.forEach(s => {{
    const key = 'Individual_level_' + s + '_State';
    const gv = gt[key]; const pv = pred[key];
    const gtStr = gv == null ? '—' : gv ? 'T' : 'F';
    const pvStr = pv == null ? '—' : pv ? 'T' : 'F';
    let matchCls = 'b-na', matchStr = '—';
    if (gv != null && pv != null) {{
      const match = gv === pv;
      matchCls = match ? 'b-match' : 'b-miss';
      matchStr = match ? '+' : 'x';
    }}
    html += `<tr>
      <td>${{s}}</td>
      <td style="color:${{gv ? 'var(--accent-green)' : 'var(--text-muted)'}}">${{gtStr}}</td>
      <td style="color:${{pv ? VERSION_COLORS[version] : 'var(--text-muted)'}}">${{pvStr}}</td>
      <td class="${{matchCls}}">${{matchStr}}</td>
    </tr>`;
  }});

  let matches = 0, total = 0;
  BINARY_STATES.forEach(s => {{
    const key = 'Individual_level_' + s + '_State';
    const gv = gt[key]; const pv = pred[key];
    if (gv != null && pv != null) {{ total++; if (gv === pv) matches++; }}
  }});
  if (total > 0) {{
    html += `<tr style="border-top:2px solid var(--border)"><td colspan="3" style="text-align:right;color:var(--text-muted)">Accuracy</td>
      <td class="${{matches/total > 0.7 ? 'b-match' : 'b-miss'}}">${{matches}}/${{total}} (${{(matches/total*100).toFixed(0)}}%)</td></tr>`;
  }}
  html += '</table></div>';
  html += '</div>';
  return html;
}}

// ── Process tab ─────────────────────────────────────────────────────────
function renderProcessTab(entry, version) {{
  const vd = entry.versions[version] || {{}};

  let html = `<div class="tab-content ${{activeTab === 'process' ? 'active' : ''}}" id="tab-process">`;

  if (vd.reasoning) {{
    html += collapsible('Reasoning', `<pre>${{esc(vd.reasoning)}}</pre>`, true);
  }}
  if (vd.sensing_summary) {{
    html += collapsible('Sensing Summary', `<pre>${{esc(vd.sensing_summary)}}</pre>`);
  }}
  if (vd.tool_calls && vd.tool_calls.length > 0) {{
    let tc = `<div style="margin-bottom:4px;color:var(--text-muted);font-size:11px">${{vd.tool_calls.length}} tool calls across ${{vd.n_rounds || '?'}} rounds</div>`;
    vd.tool_calls.forEach((t, i) => {{
      const inputStr = t.input ? JSON.stringify(t.input) : '';
      tc += `<div class="tool-call">
        <span style="color:var(--text-muted);font-size:10px">#${{t.index || i+1}}</span>
        <span class="tool-name">${{esc(t.tool_name || '')}}</span>
        ${{inputStr ? `<div class="tool-input">${{esc(inputStr.substring(0, 120))}}</div>` : ''}}
      </div>`;
    }});
    html += collapsible(`Tool Calls (${{vd.tool_calls.length}})`, tc, true);
  }}
  if (vd.rag_cases && vd.rag_cases.length > 0) {{
    let rc = '';
    vd.rag_cases.forEach(r => {{
      rc += `<div class="rag-case">
        <span class="rag-sim">${{r.similarity?.toFixed(2) || '?'}}</span>
        "${{esc(String(r.text || '').substring(0, 100))}}"
        <span style="color:var(--text-muted);font-size:10px;margin-left:8px">PA=${{r.PANAS_Pos ?? '?'}} NA=${{r.PANAS_Neg ?? '?'}}</span>
      </div>`;
    }});
    html += collapsible(`RAG Cases (${{vd.rag_cases.length}})`, rc);
  }}
  if (vd.memory_excerpt) {{
    html += collapsible('Memory Excerpt', `<pre>${{esc(vd.memory_excerpt)}}</pre>`);
  }}
  if (vd.emotion_driver) {{
    html += collapsible('Diary / Emotion Driver', `<pre>${{esc(vd.emotion_driver)}}</pre>`);
  }}
  if (vd.trait_summary) {{
    html += collapsible('Trait Summary', `<pre>${{esc(vd.trait_summary)}}</pre>`);
  }}
  if (vd.modalities_available && vd.modalities_available.length > 0) {{
    html += collapsible('Modalities', `
      <div style="font-size:11px">
        <div><strong>Available:</strong> ${{vd.modalities_available.join(', ') || 'none'}}</div>
        <div style="color:var(--text-muted)"><strong>Missing:</strong> ${{(vd.modalities_missing || []).join(', ') || 'none'}}</div>
      </div>`);
  }}

  if (!html.includes('process-section')) {{
    html += '<div style="padding:20px;color:var(--text-muted);text-align:center">No process data for this version.</div>';
  }}

  html += '</div>';
  return html;
}}

// ── Raw tab ─────────────────────────────────────────────────────────────
function renderRawTab(entry, version) {{
  const vd = entry.versions[version] || {{}};

  let html = `<div class="tab-content ${{activeTab === 'raw' ? 'active' : ''}}" id="tab-raw">`;

  html += '<div class="raw-stats">';
  const stats = [
    ['Input Tokens', vd.input_tokens || 0],
    ['Output Tokens', vd.output_tokens || 0],
    ['Total Tokens', vd.total_tokens || 0],
    ['LLM Calls', vd.llm_calls || 1],
    ['Elapsed', vd.elapsed_seconds ? vd.elapsed_seconds.toFixed(1) + 's' : '—'],
    ['Cost', vd.cost_usd ? '$' + vd.cost_usd.toFixed(4) : '$0'],
    ['Prompt Len', vd.full_prompt ? vd.full_prompt.length + (vd.prompt_truncated ? '+' : '') : '—'],
    ['Model', vd.model || '—'],
  ];
  stats.forEach(([label, val]) => {{
    html += `<div class="raw-stat"><div class="rs-label">${{label}}</div><div class="rs-val">${{val}}</div></div>`;
  }});
  html += '</div>';

  if (vd.system_prompt) {{
    html += collapsible('System Prompt', `<pre>${{esc(vd.system_prompt)}}</pre>`);
  }}
  if (vd.full_prompt) {{
    const trunc = vd.prompt_truncated ? ' (truncated)' : '';
    html += collapsible(`Full Prompt${{trunc}}`, `<pre>${{esc(vd.full_prompt)}}</pre>`);
  }}
  if (vd.full_response) {{
    const trunc = vd.response_truncated ? ' (truncated)' : '';
    html += collapsible(`Full Response${{trunc}}`, `<pre>${{esc(vd.full_response)}}</pre>`);
  }}

  html += '</div>';
  return html;
}}

// ── Helpers ─────────────────────────────────────────────────────────────
function collapsible(title, content, openByDefault) {{
  return `<div class="process-section">
    <div class="process-header" onclick="toggleCollapse(this)">
      ${{title}} <span class="arrow">${{openByDefault ? 'v' : '>'}}</span>
    </div>
    <div class="process-body ${{openByDefault ? 'open' : ''}}">${{content}}</div>
  </div>`;
}}

function toggleCollapse(el) {{
  const body = el.nextElementSibling;
  body.classList.toggle('open');
  el.querySelector('.arrow').textContent = body.classList.contains('open') ? 'v' : '>';
}}

function switchDetailVersion(v) {{
  activeVersion = v;
  if (activeUser && activeEntryIdx !== null) {{
    const entries = getFilteredEntries(activeUser);
    const entry = entries.find(e => e.idx === activeEntryIdx);
    if (entry) renderEntryDetail(activeUser, entry);
  }}
}}

function switchTab(tab) {{
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.textContent.toLowerCase() === tab));
  document.querySelectorAll('.tab-content').forEach(tc => {{
    tc.classList.toggle('active', tc.id === 'tab-' + tab);
  }});
}}

function getFilteredEntries(uid) {{
  const fVer = document.getElementById('filterVersion').value;
  const fSlot = document.getElementById('filterSlot').value;
  let entries = DATA[uid] || [];
  if (fVer !== 'all') entries = entries.filter(e => e.versions[fVer]);
  if (fSlot !== 'all') entries = entries.filter(e => e.slot === fSlot);
  return entries;
}}

function fmt(v) {{
  if (v == null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(1);
  return String(v);
}}

function esc(s) {{
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function errColor(pred, gt) {{
  if (pred == null || gt == null) return 'var(--text-muted)';
  const err = Math.abs(pred - gt);
  if (err <= 2) return 'var(--accent-green)';
  if (err <= 5) return 'var(--accent-orange)';
  return 'var(--accent-red)';
}}

function errBadge(pred, gt) {{
  if (pred == null || gt == null) return '';
  const err = Math.abs(pred - gt).toFixed(1);
  const color = errColor(pred, gt);
  return `<span style="font-size:9px;color:${{color}}">+/-${{err}}</span>`;
}}

init();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Dashboard written to: {output_path} ({size_mb:.1f} MB)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate pilot study dashboard")
    parser.add_argument("--output", default=str(PILOT_DIR / "dashboard.html"))
    args = parser.parse_args()

    print("Loading JSONL records...")
    users = load_all_records()

    print("Loading V5/V6 checkpoints...")
    users = load_v5v6_checkpoints(users)

    print("Loading evaluation data...")
    eval_data = load_evaluation()

    print("Loading token stats...")
    token_stats = load_token_stats()

    print("Loading elapsed time stats...")
    elapsed_stats = load_elapsed_stats()

    print("Building dashboard data...")
    js_data = build_js_data(users)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html(js_data, eval_data, token_stats, elapsed_stats, output_path)


if __name__ == "__main__":
    main()
