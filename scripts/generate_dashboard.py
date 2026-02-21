#!/usr/bin/env python3
"""Generate interactive HTML dashboard for pilot experiment results.

Reads trace files and checkpoints to build a timeline-based visualization
where each user's EMA entries are clickable to reveal full details:
- Ground truth vs predictions (all 3 versions)
- Agent reasoning / LLM response
- Sensing data summary
- RAG retrieval results
- Memory excerpt

Auto-detects which entries have real LLM results vs dry-run placeholders.

Usage:
    python scripts/generate_dashboard.py
    python scripts/generate_dashboard.py --output outputs/pilot/dashboard.html
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRACES_DIR = PROJECT_ROOT / "outputs" / "pilot" / "traces"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints"
USERS = [71, 164, 119, 458, 310]
VERSIONS = ["callm", "v1", "v2"]
DRY_RUN_VAL = (15.0, 8.0, 3.0)


def load_checkpoint(version):
    path = CHECKPOINT_DIR / f"{version}_checkpoint.json"
    if not path.exists():
        return {"predictions": [], "ground_truths": [], "metadata": []}
    with open(path) as f:
        return json.load(f)


def load_trace(version, user_id, entry_idx):
    path = TRACES_DIR / f"{version}_user{user_id}_entry{entry_idx}.json"
    if not path.exists():
        return None
    mtime = os.path.getmtime(path)
    with open(path) as f:
        data = json.load(f)
    data["_file_mtime"] = mtime
    return data


def is_real_trace(trace, dry_run_cutoff_ts):
    """Check if trace is from real LLM run (not dry-run leftover)."""
    if trace is None:
        return False
    return trace.get("_file_mtime", 0) > dry_run_cutoff_ts


def extract_v2_pred_from_trace(trace):
    """Extract prediction values from V2 trace response using regex."""
    if not trace or "_trace" not in trace:
        return {}
    rounds = trace.get("_trace", [])
    for r in rounds:
        resp = r.get("response", "")
        pa = re.search(r'"PANAS_Pos":\s*([\d.]+)', resp)
        na = re.search(r'"PANAS_Neg":\s*([\d.]+)', resp)
        er = re.search(r'"ER_desire":\s*([\d.]+)', resp)
        if pa:
            return {
                "PANAS_Pos": float(pa.group(1)),
                "PANAS_Neg": float(na.group(1)) if na else None,
                "ER_desire": float(er.group(1)) if er else None,
            }
    return {}


def build_entries_data():
    """Build structured data for all users and entries."""
    # Find dry-run cutoff: earliest real trace timestamp
    all_mtimes = []
    for f in TRACES_DIR.glob("*.json"):
        all_mtimes.append(os.path.getmtime(f))
    if not all_mtimes:
        return {}

    # Dry run files are all written at nearly the same second
    # Real files come later. Find the gap.
    sorted_times = sorted(set(all_mtimes))
    dry_run_cutoff = sorted_times[0] + 10  # 10 seconds after earliest file

    # Load all checkpoints for ground truth
    ckpt_data = {}
    for v in VERSIONS:
        ckpt = load_checkpoint(v)
        for p, g, m in zip(ckpt["predictions"], ckpt["ground_truths"], ckpt["metadata"]):
            key = (m.get("study_id"), m.get("entry_idx"))
            if key not in ckpt_data:
                ckpt_data[key] = {"gt": g, "meta": m, "preds": {}}
            ckpt_data[key]["preds"][v] = p

    users_data = {}
    for uid in USERS:
        entries = []
        # Find all entries for this user from any checkpoint
        user_entries = {k: v for k, v in ckpt_data.items() if k[0] == uid}

        for (sid, eidx), entry_data in sorted(user_entries.items()):
            gt = entry_data["gt"]
            meta = entry_data["meta"]

            entry = {
                "idx": eidx,
                "date": meta.get("date", ""),
                "ground_truth": gt,
                "versions": {},
            }

            for v in VERSIONS:
                trace = load_trace(v, uid, eidx)
                is_real = is_real_trace(trace, dry_run_cutoff)

                version_data = {
                    "is_real": is_real,
                    "prediction": entry_data["preds"].get(v, {}),
                    "trace": {},
                }

                if trace and is_real:
                    version_data["trace"] = {
                        "emotion_driver": trace.get("_emotion_driver", ""),
                        "sensing_summary": trace.get("_sensing_summary", ""),
                        "prompt_length": trace.get("_prompt_length", 0),
                        "full_prompt": trace.get("_full_prompt", ""),
                        "full_response": trace.get("_full_response", ""),
                        "rag_top5": trace.get("_rag_top5", []),
                        "memory_excerpt": trace.get("_memory_excerpt", ""),
                        "trait_summary": trace.get("_trait_summary", ""),
                        "system_prompt": trace.get("_system_prompt", ""),
                        "llm_calls": trace.get("_llm_calls"),
                        "v2_trace": trace.get("_trace", []),
                    }

                    # For V2, extract prediction from trace response
                    if v == "v2":
                        v2_pred = extract_v2_pred_from_trace(trace)
                        if v2_pred:
                            version_data["prediction"].update(v2_pred)

                entry["versions"][v] = version_data

            entries.append(entry)

        users_data[uid] = entries

    return users_data


def generate_html(users_data, output_path):
    """Generate the interactive HTML dashboard."""

    # Serialize data for JS
    js_data = json.dumps(users_data, default=str, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pilot Experiment Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; }}
.header {{ background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 16px; }}
.header h1 {{ font-size: 18px; color: #f0f6fc; }}
.header .stats {{ font-size: 13px; color: #8b949e; }}
.container {{ display: flex; height: calc(100vh - 53px); }}
.sidebar {{ width: 340px; border-right: 1px solid #30363d; overflow-y: auto; background: #0d1117; }}
.user-section {{ border-bottom: 1px solid #30363d; }}
.user-header {{ padding: 10px 16px; background: #161b22; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }}
.user-header:hover {{ background: #1c2128; }}
.user-header h3 {{ font-size: 14px; color: #58a6ff; }}
.user-header .badge {{ font-size: 11px; background: #30363d; padding: 2px 8px; border-radius: 10px; }}
.timeline {{ padding: 4px 0; display: none; }}
.timeline.open {{ display: block; }}
.entry {{ padding: 6px 16px; cursor: pointer; display: flex; align-items: center; gap: 8px; font-size: 12px; border-left: 3px solid transparent; }}
.entry:hover {{ background: #161b22; }}
.entry.active {{ background: #1c2128; border-left-color: #58a6ff; }}
.entry .date {{ color: #8b949e; width: 75px; flex-shrink: 0; }}
.entry .dots {{ display: flex; gap: 3px; }}
.entry .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
.dot.real {{ background: #3fb950; }}
.dot.dry {{ background: #30363d; }}
.dot.missing {{ background: transparent; border: 1px solid #30363d; }}
.entry .pa-bar {{ flex: 1; height: 6px; background: #21262d; border-radius: 3px; position: relative; min-width: 60px; }}
.pa-bar .gt-marker {{ position: absolute; height: 10px; width: 2px; background: #f0f6fc; top: -2px; }}
.pa-bar .pred-marker {{ position: absolute; height: 6px; width: 6px; border-radius: 50%; top: 0; }}
.pred-marker.callm {{ background: #f97583; }}
.pred-marker.v1 {{ background: #79c0ff; }}
.pred-marker.v2 {{ background: #d2a8ff; }}
.diary-preview {{ color: #8b949e; font-size: 11px; max-width: 100px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

.main {{ flex: 1; overflow-y: auto; padding: 20px; }}
.detail-panel {{ max-width: 900px; }}
.detail-header {{ margin-bottom: 16px; }}
.detail-header h2 {{ font-size: 16px; color: #f0f6fc; margin-bottom: 4px; }}
.detail-header .sub {{ font-size: 13px; color: #8b949e; }}

.metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }}
.metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }}
.metric-card .label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }}
.metric-card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
.metric-card .sub {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
.metric-card.gt .value {{ color: #f0f6fc; }}
.metric-card.callm .value {{ color: #f97583; }}
.metric-card.v1 .value {{ color: #79c0ff; }}
.metric-card.v2 .value {{ color: #d2a8ff; }}

.version-tabs {{ display: flex; gap: 4px; margin-bottom: 12px; }}
.version-tab {{ padding: 6px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; border: 1px solid #30363d; background: #0d1117; }}
.version-tab:hover {{ background: #161b22; }}
.version-tab.active {{ background: #161b22; border-color: #58a6ff; color: #58a6ff; }}
.version-tab.callm.active {{ border-color: #f97583; color: #f97583; }}
.version-tab.v1.active {{ border-color: #79c0ff; color: #79c0ff; }}
.version-tab.v2.active {{ border-color: #d2a8ff; color: #d2a8ff; }}

.trace-section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 12px; }}
.trace-header {{ padding: 10px 14px; font-size: 13px; font-weight: 600; border-bottom: 1px solid #30363d; cursor: pointer; display: flex; justify-content: space-between; }}
.trace-header:hover {{ background: #1c2128; }}
.trace-content {{ padding: 14px; font-size: 12px; line-height: 1.6; display: none; max-height: 400px; overflow-y: auto; }}
.trace-content.open {{ display: block; }}
.trace-content pre {{ white-space: pre-wrap; word-break: break-word; color: #c9d1d9; background: #0d1117; padding: 10px; border-radius: 4px; margin-top: 6px; font-size: 11px; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 4px; }}
.tag.real {{ background: #238636; color: #fff; }}
.tag.dry {{ background: #30363d; color: #8b949e; }}
.no-data {{ text-align: center; padding: 60px; color: #484f58; }}
.no-data h3 {{ font-size: 16px; margin-bottom: 8px; }}

.binary-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin: 8px 0; }}
.binary-cell {{ padding: 4px 6px; font-size: 10px; text-align: center; border-radius: 4px; }}
.binary-cell.true-true {{ background: #238636; color: #fff; }}
.binary-cell.true-false {{ background: #8b949e33; color: #f97583; }}
.binary-cell.false-true {{ background: #8b949e33; color: #f0883e; }}
.binary-cell.false-false {{ background: #8b949e33; color: #3fb950; }}

.legend {{ display: flex; gap: 16px; font-size: 11px; color: #8b949e; margin: 8px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
</style>
</head>
<body>

<div class="header">
  <h1>Proactive Affective Agent — Pilot Dashboard</h1>
  <div class="stats" id="headerStats"></div>
</div>

<div class="container">
  <div class="sidebar" id="sidebar"></div>
  <div class="main" id="main">
    <div class="no-data">
      <h3>Select an entry from the timeline</h3>
      <p>Click on any EMA entry to view predictions, ground truth, and agent reasoning.</p>
      <div class="legend" style="justify-content:center; margin-top:16px;">
        <div class="legend-item"><div class="legend-dot" style="background:#3fb950"></div> Real LLM result</div>
        <div class="legend-item"><div class="legend-dot" style="background:#30363d"></div> Dry-run placeholder</div>
        <div class="legend-item"><div class="legend-dot dot callm" style="background:#f97583"></div> CALLM</div>
        <div class="legend-item"><div class="legend-dot dot v1" style="background:#79c0ff"></div> V1</div>
        <div class="legend-item"><div class="legend-dot dot v2" style="background:#d2a8ff"></div> V2</div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {js_data};
const USERS = {json.dumps(USERS)};
const VERSIONS = ["callm", "v1", "v2"];
const VERSION_COLORS = {{callm: "#f97583", v1: "#79c0ff", v2: "#d2a8ff"}};
const VERSION_LABELS = {{callm: "CALLM (diary+RAG)", v1: "V1 Structured (sensing)", v2: "V2 Autonomous (ReAct)"}};

let currentEntry = null;
let currentVersion = "callm";

function init() {{
  const sidebar = document.getElementById("sidebar");
  let totalReal = 0, totalEntries = 0;

  USERS.forEach(uid => {{
    const entries = DATA[uid] || [];
    const realCount = entries.filter(e =>
      VERSIONS.some(v => e.versions[v] && e.versions[v].is_real)
    ).length;
    totalReal += realCount;
    totalEntries += entries.length;

    const section = document.createElement("div");
    section.className = "user-section";
    section.innerHTML = `
      <div class="user-header" onclick="toggleTimeline(${{uid}})">
        <h3>User ${{uid}}</h3>
        <span class="badge">${{realCount}}/${{entries.length}} real</span>
      </div>
      <div class="timeline" id="timeline-${{uid}}">
        ${{entries.map(e => renderTimelineEntry(uid, e)).join("")}}
      </div>
    `;
    sidebar.appendChild(section);
  }});

  document.getElementById("headerStats").textContent =
    `${{totalReal}} real results / ${{totalEntries}} total entries across ${{USERS.length}} users`;

  // Open first user by default
  if (USERS.length > 0) toggleTimeline(USERS[0]);
}}

function renderTimelineEntry(uid, entry) {{
  const dots = VERSIONS.map(v => {{
    const vd = entry.versions[v];
    const cls = vd && vd.is_real ? "real" : "dry";
    return `<div class="dot ${{cls}}" title="${{v}}: ${{vd && vd.is_real ? 'Real' : 'Dry-run'}}"></div>`;
  }}).join("");

  const gt = entry.ground_truth || {{}};
  const paGt = gt.PANAS_Pos;
  const diary = (entry.versions.callm?.trace?.emotion_driver || "").substring(0, 25).replace(/\\n/g, " ");

  // Mini bar showing GT position
  let barHtml = "";
  if (paGt != null) {{
    const gtPct = (paGt / 30 * 100).toFixed(0);
    barHtml = `<div class="pa-bar"><div class="gt-marker" style="left:${{gtPct}}%"></div>`;
    VERSIONS.forEach(v => {{
      const vd = entry.versions[v];
      if (vd && vd.is_real && vd.prediction) {{
        const pv = vd.prediction.PANAS_Pos;
        if (pv != null) {{
          const pPct = (pv / 30 * 100).toFixed(0);
          barHtml += `<div class="pred-marker ${{v}}" style="left:calc(${{pPct}}% - 3px)" title="${{v}}: ${{pv}}"></div>`;
        }}
      }}
    }});
    barHtml += `</div>`;
  }}

  return `<div class="entry" id="entry-${{uid}}-${{entry.idx}}" onclick="selectEntry(${{uid}}, ${{entry.idx}})">
    <span class="date">${{entry.date || "?"}}</span>
    <span class="dots">${{dots}}</span>
    ${{barHtml}}
    <span class="diary-preview" title="${{diary}}">${{diary}}</span>
  </div>`;
}}

function toggleTimeline(uid) {{
  const el = document.getElementById(`timeline-${{uid}}`);
  el.classList.toggle("open");
}}

function selectEntry(uid, idx) {{
  // Deselect previous
  document.querySelectorAll(".entry.active").forEach(e => e.classList.remove("active"));
  const el = document.getElementById(`entry-${{uid}}-${{idx}}`);
  if (el) el.classList.add("active");

  const entries = DATA[uid] || [];
  const entry = entries.find(e => e.idx === idx);
  if (!entry) return;

  currentEntry = {{ uid, idx, entry }};
  renderDetail(uid, entry);
}}

function renderDetail(uid, entry) {{
  const main = document.getElementById("main");
  const gt = entry.ground_truth || {{}};
  const diary = entry.versions.callm?.trace?.emotion_driver || "";

  // Find which versions have real data
  const realVersions = VERSIONS.filter(v => entry.versions[v]?.is_real);

  let html = `<div class="detail-panel">`;
  html += `<div class="detail-header">
    <h2>User ${{uid}} — Entry ${{entry.idx}} — ${{entry.date}}</h2>
    <div class="sub">${{diary ? `Diary: "${{diary}}"` : "No diary text"}}</div>
  </div>`;

  // Metrics cards: GT + each version
  html += `<div class="metrics-grid">
    <div class="metric-card gt">
      <div class="label">Ground Truth</div>
      <div class="value">PA ${{fmt(gt.PANAS_Pos)}}</div>
      <div class="sub">NA ${{fmt(gt.PANAS_Neg)}} · ER ${{fmt(gt.ER_desire)}} · Avail: ${{gt.INT_availability || "?"}}</div>
    </div>`;

  VERSIONS.forEach(v => {{
    const vd = entry.versions[v];
    const pred = vd?.prediction || {{}};
    const isReal = vd?.is_real;
    const pa = pred.PANAS_Pos;
    const err = (pa != null && gt.PANAS_Pos != null) ? Math.abs(pa - gt.PANAS_Pos).toFixed(0) : "?";
    html += `<div class="metric-card ${{v}}">
      <div class="label">${{v.toUpperCase()}} <span class="tag ${{isReal ? 'real' : 'dry'}}">${{isReal ? 'Real' : 'Dry'}}</span></div>
      <div class="value">PA ${{fmt(pa)}}</div>
      <div class="sub">NA ${{fmt(pred.PANAS_Neg)}} · ER ${{fmt(pred.ER_desire)}} · err=${{err}}</div>
    </div>`;
  }});
  html += `</div>`;

  // Binary states comparison
  html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">Binary States Comparison <span>▸</span></div><div class="trace-content">`;
  const binaryTargets = [
    "PA", "NA", "happy", "sad", "afraid", "miserable", "worried",
    "cheerful", "pleased", "grateful", "lonely", "interactions_quality", "pain", "forecasting", "ER_desire"
  ];
  html += `<div class="binary-grid">`;
  html += `<div style="font-weight:600;font-size:10px">State</div>`;
  html += `<div style="font-weight:600;font-size:10px;text-align:center">Truth</div>`;
  VERSIONS.forEach(v => html += `<div style="font-weight:600;font-size:10px;text-align:center;color:${{VERSION_COLORS[v]}}">${{v.toUpperCase()}}</div>`);

  binaryTargets.forEach(t => {{
    const key = `Individual_level_${{t}}_State`;
    const gtv = gt[key];
    html += `<div class="binary-cell" style="text-align:left">${{t}}</div>`;
    html += `<div class="binary-cell" style="color:${{gtv ? '#3fb950' : '#8b949e'}}">${{gtv == null ? "?" : gtv ? "T" : "F"}}</div>`;
    VERSIONS.forEach(v => {{
      const pred = entry.versions[v]?.prediction || {{}};
      const pv = pred[key];
      let cls = "";
      if (gtv != null && pv != null) {{
        if (gtv && pv) cls = "true-true";
        else if (gtv && !pv) cls = "true-false";
        else if (!gtv && pv) cls = "false-true";
        else cls = "false-false";
      }}
      html += `<div class="binary-cell ${{cls}}">${{pv == null ? "?" : pv ? "T" : "F"}}</div>`;
    }});
  }});
  html += `</div></div></div>`;

  // Version tabs for detailed traces
  html += `<div class="version-tabs">`;
  VERSIONS.forEach(v => {{
    const isReal = entry.versions[v]?.is_real;
    html += `<div class="version-tab ${{v}} ${{v === currentVersion ? 'active' : ''}}" onclick="switchVersion('${{v}}')">${{VERSION_LABELS[v]}} ${{isReal ? '✓' : ''}}</div>`;
  }});
  html += `</div>`;

  // Detailed trace for selected version
  html += renderVersionTrace(entry, currentVersion);

  html += `</div>`;
  main.innerHTML = html;
}}

function renderVersionTrace(entry, version) {{
  const vd = entry.versions[version];
  if (!vd) return `<div class="no-data">No data for ${{version}}</div>`;

  const trace = vd.trace || {{}};
  let html = "";

  // Sensing summary
  if (trace.sensing_summary) {{
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">Sensing Data <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(trace.sensing_summary)}}</pre></div></div>`;
  }}

  // RAG results (CALLM)
  if (trace.rag_top5 && trace.rag_top5.length > 0) {{
    let ragHtml = trace.rag_top5.map((r, i) =>
      `${{i+1}}. [sim=${{r.similarity?.toFixed(2) || "?"}}] "${{r.text || ""}}" (PA=${{r.PANAS_Pos || "?"}}, NA=${{r.PANAS_Neg || "?"}})`
    ).join("\\n");
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">RAG Retrieval (Top 5) <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(ragHtml)}}</pre></div></div>`;
  }}

  // Memory excerpt
  if (trace.memory_excerpt) {{
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">Memory Document <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(trace.memory_excerpt)}}</pre></div></div>`;
  }}

  // Trait summary
  if (trace.trait_summary) {{
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">Trait Profile <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(trace.trait_summary)}}</pre></div></div>`;
  }}

  // V2 multi-round trace
  if (version === "v2" && trace.v2_trace && trace.v2_trace.length > 0) {{
    trace.v2_trace.forEach(round => {{
      const label = round.request ? `Round ${{round.round}} (Request: ${{round.request}})` : `Round ${{round.round}}`;
      html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">${{label}} <span>▸</span></div>
        <div class="trace-content"><pre>${{escHtml(round.response || "")}}</pre></div></div>`;
    }});
  }}

  // Full prompt
  if (trace.full_prompt) {{
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">Full Prompt (${{trace.prompt_length || "?"}} chars) <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(trace.full_prompt)}}</pre></div></div>`;
  }}

  // Full response
  if (trace.full_response) {{
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">Full LLM Response <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(trace.full_response)}}</pre></div></div>`;
  }}

  // System prompt
  if (trace.system_prompt) {{
    html += `<div class="trace-section"><div class="trace-header" onclick="toggleSection(this)">System Prompt <span>▸</span></div>
      <div class="trace-content"><pre>${{escHtml(trace.system_prompt)}}</pre></div></div>`;
  }}

  if (!html) {{
    html = `<div class="trace-section"><div class="trace-content open" style="color:#8b949e">
      ${{vd.is_real ? "Trace saved with old code — limited details. Re-run experiment with updated code for full traces." : "Dry-run placeholder — no real LLM trace."}}
    </div></div>`;
  }}

  return html;
}}

function switchVersion(v) {{
  currentVersion = v;
  if (currentEntry) {{
    renderDetail(currentEntry.uid, currentEntry.entry);
  }}
}}

function toggleSection(header) {{
  const content = header.nextElementSibling;
  content.classList.toggle("open");
  const arrow = header.querySelector("span");
  arrow.textContent = content.classList.contains("open") ? "▾" : "▸";
}}

function fmt(v) {{
  if (v == null || v === undefined) return "?";
  if (typeof v === "number") return v.toFixed(1);
  return String(v);
}}

function escHtml(s) {{
  return String(s || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

init();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to: {output_path}")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "pilot" / "dashboard.html"))
    args = parser.parse_args()

    print("Building dashboard data...")
    users_data = build_entries_data()
    total_entries = sum(len(v) for v in users_data.values())
    print(f"  {len(users_data)} users, {total_entries} entries")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html(users_data, output_path)


if __name__ == "__main__":
    main()
