#!/usr/bin/env python3
"""Generate interactive HTML dashboard for pilot experiment results.

Reads unified JSONL records (*_records.jsonl) and generates a self-contained
HTML dashboard with Chart.js visualizations, dark theme, and detailed
per-entry inspection including agent process, predictions, and raw data.

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

VERSIONS = ["callm", "v1", "v2", "v3", "v4"]
VERSION_LABELS = {
    "callm": "CALLM",
    "v1": "V1 Structured",
    "v2": "V2 Agentic",
    "v3": "V3 Struct+Diary",
    "v4": "V4 Agent+Diary",
}
VERSION_COLORS = {
    "callm": "#f97583",
    "v1": "#79c0ff",
    "v2": "#d2a8ff",
    "v3": "#3fb950",
    "v4": "#f0883e",
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
        # Parse version and user from filename: {version}_user{id}_records.jsonl
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

                # Truncate large text fields to manage output size
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


def build_js_data(users: dict) -> dict:
    """Convert internal data structure to JS-friendly format.

    Returns: {user_id: [sorted list of entry dicts]}
    """
    js_data = {}
    for uid in sorted(users.keys()):
        entries = []
        for idx in sorted(users[uid].keys()):
            entry = users[uid][idx]
            entry["idx"] = idx
            entries.append(entry)
        js_data[uid] = entries
    return js_data


def generate_html(js_data: dict, output_path: Path):
    """Generate the self-contained HTML dashboard."""
    data_json = json.dumps(js_data, default=str, ensure_ascii=False)
    versions_json = json.dumps(VERSIONS)
    labels_json = json.dumps(VERSION_LABELS)
    colors_json = json.dumps(VERSION_COLORS)
    binary_json = json.dumps(BINARY_STATES)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pilot Study Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
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
  --sidebar-width: 280px;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, sans-serif; background: var(--bg-primary); color: var(--text-secondary); font-size: 13px; }}

/* Header */
.header {{
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky; top: 0; z-index: 100;
}}
.header h1 {{ font-size: 15px; color: var(--text-primary); font-weight: 600; }}
.header-stats {{ font-size: 12px; color: var(--text-muted); display: flex; gap: 16px; }}
.header-stat {{ display: flex; align-items: center; gap: 4px; }}
.header-stat .num {{ color: var(--text-primary); font-weight: 600; }}

/* Layout */
.container {{ display: flex; height: calc(100vh - 45px); }}

/* Sidebar */
.sidebar {{
  width: var(--sidebar-width);
  min-width: var(--sidebar-width);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
}}
.sidebar-filters {{
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 6px;
}}
.filter-row {{
  display: flex;
  gap: 6px;
  align-items: center;
}}
.filter-row label {{
  font-size: 11px;
  color: var(--text-muted);
  width: 40px;
  flex-shrink: 0;
}}
.filter-row select {{
  flex: 1;
  background: var(--bg-input);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 3px 6px;
  border-radius: 4px;
  font-size: 11px;
}}
.user-list {{
  flex: 1;
  overflow-y: auto;
}}
.user-item {{
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background 0.1s;
}}
.user-item:hover {{ background: var(--bg-tertiary); }}
.user-item.active {{ background: var(--bg-tertiary); border-left: 3px solid var(--accent-blue); padding-left: 9px; }}
.user-item .uid {{ font-weight: 600; color: var(--text-primary); font-size: 13px; }}
.user-item .meta {{ font-size: 11px; color: var(--text-muted); text-align: right; }}
.user-item .ver-dots {{ display: flex; gap: 3px; margin-top: 2px; justify-content: flex-end; }}
.ver-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

/* Main content */
.main {{
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}}

/* User Summary */
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

/* Chart container */
.chart-container {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
}}
.chart-controls {{
  display: flex;
  gap: 6px;
  margin-bottom: 10px;
}}
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

/* Entry timeline */
.timeline-strip {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 16px;
  margin-bottom: 16px;
}}
.timeline-strip h3 {{
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 8px;
  font-weight: 500;
}}
.timeline-dots {{
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}}
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
  bottom: 20px;
  left: 50%;
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

/* Entry detail */
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
.badge {{
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 600;
}}
.badge-conf {{ background: #388bfd33; color: var(--accent-blue); }}
.badge-slot {{ background: #21262d; color: var(--text-muted); }}

/* Tabs */
.tab-bar {{
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--bg-primary);
}}
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

/* Predictions tab */
.cont-preds {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}}
.cont-card {{
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
}}
.cont-card .metric-name {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 6px; }}
.cont-card .gt-row {{ display: flex; align-items: baseline; gap: 6px; margin-bottom: 4px; }}
.cont-card .gt-val {{ font-size: 18px; font-weight: 700; color: var(--text-primary); }}
.cont-card .gt-label {{ font-size: 10px; color: var(--text-muted); }}
.pred-row {{ display: flex; align-items: center; gap: 6px; font-size: 11px; margin: 2px 0; }}
.pred-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
.pred-val {{ font-weight: 600; }}
.pred-err {{ color: var(--text-muted); }}

/* Binary grid */
.binary-section {{ margin-top: 8px; }}
.binary-section h4 {{ font-size: 12px; color: var(--text-muted); margin-bottom: 8px; font-weight: 500; }}
.binary-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}}
.binary-table th {{
  padding: 4px 8px;
  text-align: center;
  font-weight: 600;
  font-size: 10px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}}
.binary-table td {{
  padding: 4px 8px;
  text-align: center;
  border-bottom: 1px solid #21262d;
}}
.binary-table td:first-child {{ text-align: left; color: var(--text-secondary); }}
.b-match {{ color: var(--accent-green); font-weight: 600; }}
.b-miss {{ color: var(--accent-red); font-weight: 600; }}
.b-na {{ color: #484f58; }}

/* Process tab */
.process-section {{
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-bottom: 8px;
  overflow: hidden;
}}
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
.tool-call {{
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 10px;
  margin: 4px 0;
  font-size: 11px;
}}
.tool-call .tool-name {{ color: var(--accent-purple); font-weight: 600; }}
.tool-call .tool-input {{ color: var(--text-muted); font-family: monospace; font-size: 10px; }}
.tool-call .tool-preview {{ color: #484f58; font-size: 10px; margin-top: 2px; }}

/* RAG case */
.rag-case {{
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 10px;
  margin: 4px 0;
  font-size: 11px;
}}
.rag-sim {{ color: var(--accent-green); font-weight: 600; font-size: 10px; }}

/* Raw tab */
.raw-stats {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}}
.raw-stat {{
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px;
  text-align: center;
}}
.raw-stat .rs-label {{ font-size: 10px; color: var(--text-muted); }}
.raw-stat .rs-val {{ font-size: 14px; font-weight: 600; color: var(--text-primary); margin-top: 2px; }}

/* Version comparison */
.version-comparison {{
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 16px;
  overflow: hidden;
}}
.vc-header {{
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  color: var(--text-primary);
  font-weight: 600;
}}
.vc-grid {{
  display: grid;
  gap: 0;
}}
.vc-card {{
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: grid;
  grid-template-columns: 110px 1fr 1fr 1fr 120px;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}}
.vc-card:last-child {{ border-bottom: none; }}
.vc-version {{ font-weight: 600; font-size: 12px; }}
.vc-val {{ text-align: center; }}
.vc-val .v {{ font-size: 14px; font-weight: 600; }}
.vc-val .l {{ font-size: 9px; color: var(--text-muted); display: block; }}
.vc-conf {{ text-align: right; color: var(--text-muted); font-size: 11px; }}

/* Version selector for detail */
.ver-selector {{
  display: flex;
  gap: 4px;
  margin-bottom: 0;
}}
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

/* Welcome state */
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
.welcome .legend {{
  display: flex;
  gap: 16px;
  margin-top: 20px;
  flex-wrap: wrap;
  justify-content: center;
}}
.legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 11px; }}
.legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}
</style>
</head>
<body>

<div class="header">
  <h1>Proactive Affective Agent — Pilot Dashboard</h1>
  <div class="header-stats" id="headerStats"></div>
</div>

<div class="container">
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
const DATA = {data_json};
const VERSIONS = {versions_json};
const VERSION_LABELS = {labels_json};
const VERSION_COLORS = {colors_json};
const BINARY_STATES = {binary_json};
const USERS = Object.keys(DATA).map(Number).sort((a,b) => a - b);

let activeUser = null;
let activeEntryIdx = null;
let activeVersion = null;
let activeMetric = 'PANAS_Pos';
let activeTab = 'predictions';
let timeChart = null;

// ── Init ────────────────────────────────────────────────────────────────
function init() {{
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
    `<div class="header-stat"><span class="num">${{totalRecords}}</span> records</div>` +
    `<div class="header-stat"><span class="num">${{allVersions.size}}</span> versions</div>`;

  renderUserList();
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

    // Filter: skip users without selected version
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

  // User summary card
  html += `<div class="user-summary">
    <h2>User ${{uid}}</h2>
    <div class="stats">
      <div><span class="stat-val">${{entries.length}}</span> entries</div>
      <div><span class="stat-val">${{[...vers].map(v => VERSION_LABELS[v]).join(', ')}}</span></div>
    </div>
  </div>`;

  // Time series chart
  html += `<div class="chart-container">
    <div class="chart-controls">
      ${{['PANAS_Pos', 'PANAS_Neg', 'ER_desire'].map(m =>
        `<div class="chart-btn ${{m === activeMetric ? 'active' : ''}}" onclick="switchMetric('${{m}}')">${{m.replace('_', ' ')}}</div>`
      ).join('')}}
    </div>
    <div class="chart-wrap"><canvas id="timeChart"></canvas></div>
  </div>`;

  // Entry timeline
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

  // Entry detail
  html += '<div id="entryDetail"></div>';

  main.innerHTML = html;

  // Render chart
  renderTimeChart(uid, entries);

  // If entry was selected, re-render detail
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

  // Ground truth
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

  // Each version
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
        legend: {{
          labels: {{ color: '#8b949e', font: {{ size: 11 }}, boxWidth: 12 }}
        }},
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
        x: {{
          ticks: {{ color: '#8b949e', font: {{ size: 10 }}, maxRotation: 45 }},
          grid: {{ color: '#21262d' }}
        }},
        y: {{
          ticks: {{ color: '#8b949e', font: {{ size: 10 }} }},
          grid: {{ color: '#21262d' }}
        }}
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

  // Update timeline dot highlights
  document.querySelectorAll('.t-dot').forEach(d => d.classList.remove('active'));
  const dot = document.querySelector(`.t-dot[onclick="selectEntry(${{uid}}, ${{idx}})"]`);
  if (dot) dot.classList.add('active');

  // Pick default version
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

  // Version comparison bar
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

  // Detail panel with tabs
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

  // Continuous predictions
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

  // Binary states
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
      matchStr = match ? '✓' : '✗';
    }}
    html += `<tr>
      <td>${{s}}</td>
      <td style="color:${{gv ? 'var(--accent-green)' : 'var(--text-muted)'}}">${{gtStr}}</td>
      <td style="color:${{pv ? VERSION_COLORS[version] : 'var(--text-muted)'}}">${{pvStr}}</td>
      <td class="${{matchCls}}">${{matchStr}}</td>
    </tr>`;
  }});

  // Count matches
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

  // Reasoning
  if (vd.reasoning) {{
    html += collapsible('Reasoning', `<pre>${{esc(vd.reasoning)}}</pre>`, true);
  }}

  // Sensing summary
  if (vd.sensing_summary) {{
    html += collapsible('Sensing Summary', `<pre>${{esc(vd.sensing_summary)}}</pre>`);
  }}

  // Tool calls (V2/V4)
  if (vd.tool_calls && vd.tool_calls.length > 0) {{
    let tc = `<div style="margin-bottom:4px;color:var(--text-muted);font-size:11px">${{vd.tool_calls.length}} tool calls across ${{vd.n_rounds || '?'}} rounds</div>`;
    vd.tool_calls.forEach((t, i) => {{
      const inputStr = t.input ? JSON.stringify(t.input) : '';
      tc += `<div class="tool-call">
        <span style="color:var(--text-muted);font-size:10px">#${{t.index || i+1}}</span>
        <span class="tool-name">${{esc(t.tool_name || '')}}</span>
        ${{inputStr ? `<div class="tool-input">${{esc(inputStr.substring(0, 120))}}</div>` : ''}}
        ${{t.result_preview ? `<div class="tool-preview">${{esc(String(t.result_preview).substring(0, 150))}}</div>` : ''}}
      </div>`;
    }});
    html += collapsible(`Tool Calls (${{vd.tool_calls.length}})`, tc, true);
  }}

  // RAG cases
  if (vd.rag_cases && vd.rag_cases.length > 0) {{
    let rc = '';
    vd.rag_cases.forEach((r, i) => {{
      rc += `<div class="rag-case">
        <span class="rag-sim">${{r.similarity?.toFixed(2) || '?'}}</span>
        "${{esc(String(r.text || '').substring(0, 100))}}"
        <span style="color:var(--text-muted);font-size:10px;margin-left:8px">PA=${{r.PANAS_Pos ?? '?'}} NA=${{r.PANAS_Neg ?? '?'}}</span>
      </div>`;
    }});
    html += collapsible(`RAG Cases (${{vd.rag_cases.length}})`, rc);
  }}

  // Memory excerpt
  if (vd.memory_excerpt) {{
    html += collapsible('Memory Excerpt', `<pre>${{esc(vd.memory_excerpt)}}</pre>`);
  }}

  // Emotion driver / diary
  if (vd.emotion_driver) {{
    html += collapsible('Diary / Emotion Driver', `<pre>${{esc(vd.emotion_driver)}}</pre>`);
  }}

  // Trait summary
  if (vd.trait_summary) {{
    html += collapsible('Trait Summary', `<pre>${{esc(vd.trait_summary)}}</pre>`);
  }}

  // Modalities
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

  // Stats grid
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

  // System prompt
  if (vd.system_prompt) {{
    html += collapsible('System Prompt', `<pre>${{esc(vd.system_prompt)}}</pre>`);
  }}

  // Full prompt
  if (vd.full_prompt) {{
    const trunc = vd.prompt_truncated ? ' (truncated)' : '';
    html += collapsible(`Full Prompt${{trunc}}`, `<pre>${{esc(vd.full_prompt)}}</pre>`);
  }}

  // Full response
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
      ${{title}} <span class="arrow">${{openByDefault ? '▾' : '▸'}}</span>
    </div>
    <div class="process-body ${{openByDefault ? 'open' : ''}}">${{content}}</div>
  </div>`;
}}

function toggleCollapse(el) {{
  const body = el.nextElementSibling;
  body.classList.toggle('open');
  el.querySelector('.arrow').textContent = body.classList.contains('open') ? '▾' : '▸';
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
  return `<span style="font-size:9px;color:${{color}}">±${{err}}</span>`;
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

    print("Building dashboard data...")
    js_data = build_js_data(users)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html(js_data, output_path)


if __name__ == "__main__":
    main()
