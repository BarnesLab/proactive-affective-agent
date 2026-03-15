#!/usr/bin/env python3
"""Live results dashboard for pilot experiment.

Serves a self-refreshing HTML dashboard at http://localhost:8877.
On each page load, re-reads all checkpoint files and computes metrics
from scratch — so results update in real time as experiments complete.

Incomplete data (partial users or versions) is clearly marked with
badges, progress bars, and tooltips so you always know what's final
vs. in-progress.

Usage:
    python scripts/results_dashboard.py
    python scripts/results_dashboard.py --port 8877
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIRS = [
    PROJECT_ROOT / "outputs" / "pilot_v2" / "checkpoints",
    PROJECT_ROOT / "outputs" / "pilot" / "checkpoints",
]

PILOT_USERS = [43, 71, 119, 164, 258, 275, 310, 338, 362, 399, 403, 437, 458, 513]
VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]

VERSION_LABELS = {
    "callm": "CALLM", "v1": "V1", "v2": "V2", "v3": "V3",
    "v4": "V4", "v5": "V5", "v6": "V6",
}
VERSION_DESCRIPTIONS = {
    "callm": "Diary + TF-IDF RAG (baseline)",
    "v1": "Sensing only, structured pipeline",
    "v2": "Sensing only, agentic tool-use",
    "v3": "Multimodal (diary+sensing), structured",
    "v4": "Multimodal (diary+sensing), agentic",
    "v5": "Sensing filtered, agentic",
    "v6": "Multimodal filtered, agentic",
}
VERSION_COLORS = {
    "callm": "#f97583", "v1": "#79c0ff", "v2": "#d2a8ff", "v3": "#3fb950",
    "v4": "#f0883e", "v5": "#56d4dd", "v6": "#f778ba",
}

# Expected entries per user (from EMA data)
USER_TOTALS = {
    43: 93, 71: 93, 119: 84, 164: 87, 258: 94, 275: 89,
    310: 81, 338: 81, 362: 88, 399: 96, 403: 82, 437: 88, 458: 82, 513: 90,
}

CONTINUOUS_TARGETS = ["PANAS_Pos", "PANAS_Neg", "ER_desire"]
BINARY_TARGETS = [
    "Individual_level_PA_State", "Individual_level_NA_State",
    "Individual_level_happy_State", "Individual_level_sad_State",
    "Individual_level_afraid_State", "Individual_level_miserable_State",
    "Individual_level_worried_State", "Individual_level_cheerful_State",
    "Individual_level_pleased_State", "Individual_level_grateful_State",
    "Individual_level_lonely_State",
    "Individual_level_interactions_quality_State",
    "Individual_level_pain_State", "Individual_level_forecasting_State",
    "Individual_level_ER_desire_State", "INT_availability",
]

AR_BASELINE_BA = 0.658
AR_BASELINE_F1 = 0.617
AR_BASELINE_MAE = 2.758

PORT = 8877


# ---------------------------------------------------------------------------
# Data loading & metrics (computed fresh on each request)
# ---------------------------------------------------------------------------

def find_checkpoint(version: str, uid: int) -> Path | None:
    names = [f"{version}_user{uid}_checkpoint.json"]
    if version.lower() == "callm":
        names.append(f"CALLM_user{uid}_checkpoint.json")
    for cp_dir in CHECKPOINT_DIRS:
        for name in names:
            f = cp_dir / name
            if f.exists():
                return f
    return None


def load_all_checkpoints() -> dict:
    """Load all checkpoint data. Returns {version: {uid: {n, total, preds, gts, metadata}}}."""
    data = {}
    for ver in VERSIONS:
        data[ver] = {}
        for uid in PILOT_USERS:
            f = find_checkpoint(ver, uid)
            if f is None:
                continue
            try:
                cp = json.loads(f.read_text())
                preds = cp.get("predictions", [])
                gts = cp.get("ground_truths", [])
                meta = cp.get("metadata", [])
                data[ver][uid] = {
                    "n": len(preds),
                    "total": USER_TOTALS.get(uid, 0),
                    "preds": preds,
                    "gts": gts,
                    "metadata": meta,
                }
            except Exception:
                pass
    return data


def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.lower().strip()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


def compute_metrics(all_preds: list[dict], all_gts: list[dict]) -> dict:
    """Compute aggregate metrics from predictions and ground truths."""
    result = {"continuous": {}, "binary": {}, "aggregate": {}}

    # Continuous
    for target in CONTINUOUS_TARGETS:
        y, yhat = [], []
        for p, g in zip(all_preds, all_gts):
            gv = _to_float(g.get(target))
            pv = _to_float(p.get(target))
            if gv is not None and pv is not None:
                y.append(gv)
                yhat.append(pv)
        if len(y) >= 5:
            result["continuous"][target] = {
                "mae": float(mean_absolute_error(y, yhat)),
                "n": len(y),
            }

    # Binary
    for target in BINARY_TARGETS:
        y, yhat = [], []
        for p, g in zip(all_preds, all_gts):
            gv = _to_bool(g.get(target))
            pv = _to_bool(p.get(target))
            if gv is not None and pv is not None:
                y.append(int(gv))
                yhat.append(int(pv))
        if len(y) >= 5:
            result["binary"][target] = {
                "ba": float(balanced_accuracy_score(y, yhat)),
                "f1": float(f1_score(y, yhat, average="macro", zero_division=0)),
                "n": len(y),
            }

    # Aggregates
    maes = [v["mae"] for v in result["continuous"].values()]
    bas = [v["ba"] for v in result["binary"].values()]
    f1s = [v["f1"] for v in result["binary"].values()]
    result["aggregate"] = {
        "mean_mae": float(np.mean(maes)) if maes else None,
        "mean_ba": float(np.mean(bas)) if bas else None,
        "mean_f1": float(np.mean(f1s)) if f1s else None,
    }

    return result


def build_dashboard_data() -> dict:
    """Build all data needed for the dashboard."""
    checkpoints = load_all_checkpoints()

    # Per-version metrics + completion info
    versions_data = {}
    for ver in VERSIONS:
        users_done = {}
        all_preds, all_gts = [], []
        for uid in PILOT_USERS:
            if uid in checkpoints[ver]:
                info = checkpoints[ver][uid]
                users_done[uid] = {"n": info["n"], "total": info["total"]}
                all_preds.extend(info["preds"])
                all_gts.extend(info["gts"])

        n_users = len(users_done)
        n_entries = sum(u["n"] for u in users_done.values())
        n_total = sum(u["total"] for u in users_done.values())
        n_complete = sum(1 for u in users_done.values() if u["n"] >= u["total"])

        metrics = compute_metrics(all_preds, all_gts) if all_preds else None

        versions_data[ver] = {
            "n_users": n_users,
            "n_complete_users": n_complete,
            "n_entries": n_entries,
            "n_total": n_total,
            "users": users_done,
            "metrics": metrics,
        }

    # Completion matrix (for the grid)
    matrix = {}
    for uid in PILOT_USERS:
        matrix[uid] = {}
        for ver in VERSIONS:
            if uid in checkpoints[ver]:
                info = checkpoints[ver][uid]
                n, total = info["n"], info["total"]
                if n >= total:
                    matrix[uid][ver] = "complete"
                elif n > 0:
                    matrix[uid][ver] = "partial"
                else:
                    matrix[uid][ver] = "empty"
            else:
                matrix[uid][ver] = "missing"

    return {
        "versions": versions_data,
        "matrix": matrix,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html() -> str:
    data = build_dashboard_data()
    now = data["timestamp"]

    # Build metrics comparison table
    metrics_rows = ""
    for ver in VERSIONS:
        vd = data["versions"][ver]
        m = vd["metrics"]
        label = VERSION_LABELS[ver]
        color = VERSION_COLORS[ver]
        desc = VERSION_DESCRIPTIONS[ver]
        n_users = vd["n_users"]
        n_complete = vd["n_complete_users"]
        n_entries = vd["n_entries"]
        n_total = vd["n_total"]
        pct = 100 * n_entries / n_total if n_total else 0

        # Completeness badge
        if n_complete == len(PILOT_USERS) and n_entries >= n_total:
            badge = '<span class="badge badge-complete">COMPLETE</span>'
        elif n_entries > 0:
            badge = f'<span class="badge badge-partial">{n_complete}/{len(PILOT_USERS)} users</span>'
        else:
            badge = '<span class="badge badge-missing">NO DATA</span>'

        mae_val = f"{m['aggregate']['mean_mae']:.3f}" if m and m["aggregate"]["mean_mae"] else "—"
        ba_val = f"{m['aggregate']['mean_ba']:.3f}" if m and m["aggregate"]["mean_ba"] else "—"
        f1_val = f"{m['aggregate']['mean_f1']:.3f}" if m and m["aggregate"]["mean_f1"] else "—"

        # Highlight if beats AR baseline
        ba_class = ""
        if m and m["aggregate"]["mean_ba"] and m["aggregate"]["mean_ba"] > AR_BASELINE_BA:
            ba_class = ' class="beats-ar"'

        metrics_rows += f"""<tr>
            <td><span class="ver-badge" style="background:{color}">{label}</span>
                <span class="ver-desc">{desc}</span></td>
            <td>{badge}</td>
            <td>{n_entries}</td>
            <td><div class="mini-bar-container"><div class="mini-bar" style="width:{max(pct,2):.0f}%"></div></div><span class="pct-text">{pct:.0f}%</span></td>
            <td>{mae_val}</td>
            <td{ba_class}>{ba_val}</td>
            <td>{f1_val}</td>
        </tr>"""

    # Build completion matrix
    matrix_header = "".join(
        f'<th><span class="ver-badge-sm" style="background:{VERSION_COLORS[v]}">{VERSION_LABELS[v]}</span></th>'
        for v in VERSIONS
    )
    matrix_rows = ""
    for uid in PILOT_USERS:
        total = USER_TOTALS.get(uid, 0)
        cells = ""
        for ver in VERSIONS:
            status = data["matrix"][uid][ver]
            vd = data["versions"][ver]
            if uid in vd["users"]:
                n = vd["users"][uid]["n"]
                pct = 100 * n / total if total else 0
            else:
                n = 0
                pct = 0

            if status == "complete":
                cells += f'<td class="cell-complete" title="{n}/{total}">{n}</td>'
            elif status == "partial":
                cells += f'<td class="cell-partial" title="{n}/{total} ({pct:.0f}%)">{n}<span class="cell-total">/{total}</span></td>'
            else:
                cells += '<td class="cell-missing">—</td>'

        matrix_rows += f"""<tr>
            <td class="uid-cell">User {uid}</td>
            <td class="total-cell">{total}</td>
            {cells}
        </tr>"""

    # Per-target detail table (only if we have data)
    target_rows = ""
    # Show key binary targets
    key_targets = [
        "Individual_level_PA_State", "Individual_level_NA_State",
        "Individual_level_happy_State", "Individual_level_sad_State",
        "Individual_level_worried_State", "INT_availability",
    ]
    for target in key_targets:
        short_name = target.replace("Individual_level_", "").replace("_State", "")
        row = f'<td class="target-name">{short_name}</td>'
        for ver in VERSIONS:
            vd = data["versions"][ver]
            m = vd["metrics"]
            if m and target in m["binary"]:
                ba = m["binary"][target]["ba"]
                ba_cls = " beats-ar" if ba > AR_BASELINE_BA else ""
                row += f'<td class="{ba_cls}">{ba:.3f}</td>'
            else:
                row += '<td class="no-data">—</td>'
        target_rows += f"<tr>{row}</tr>"

    # Continuous targets
    for target in CONTINUOUS_TARGETS:
        row = f'<td class="target-name">{target}</td>'
        for ver in VERSIONS:
            vd = data["versions"][ver]
            m = vd["metrics"]
            if m and target in m["continuous"]:
                mae = m["continuous"][target]["mae"]
                row += f'<td>{mae:.3f}</td>'
            else:
                row += '<td class="no-data">—</td>'
        target_rows += f"<tr>{row}</tr>"

    ver_headers = "".join(
        f'<th style="color:{VERSION_COLORS[v]}">{VERSION_LABELS[v]}</th>'
        for v in VERSIONS
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>PAA Results Dashboard</title>
<style>
:root {{
    --bg: #0d1117; --bg2: #161b22; --bg3: #1c2128; --border: #30363d;
    --text: #c9d1d9; --text-bright: #f0f6fc; --text-muted: #8b949e;
    --green: #3fb950; --yellow: #d29922; --red: #f85149; --blue: #58a6ff;
    --orange: #f0883e; --purple: #d2a8ff; --cyan: #56d4dd;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:var(--bg); color:var(--text); font-size:13px; }}

.header {{
    background: var(--bg2); border-bottom: 1px solid var(--border);
    padding: 16px 24px; display:flex; align-items:center; justify-content:space-between;
}}
.header h1 {{ font-size:18px; color:var(--text-bright); font-weight:600; }}
.header-meta {{ font-size:12px; color:var(--text-muted); display:flex; gap:16px; }}
.header-meta .val {{ color:var(--blue); font-weight:600; }}

.container {{ max-width:1400px; margin:0 auto; padding:20px 24px; }}

/* Section headers */
.section {{ margin-bottom:28px; }}
.section-title {{
    font-size:15px; color:var(--text-bright); font-weight:600; margin-bottom:12px;
    padding-bottom:8px; border-bottom:1px solid var(--border);
    display:flex; align-items:center; gap:10px;
}}
.section-subtitle {{ font-size:12px; color:var(--text-muted); font-weight:400; }}

/* AR baseline reference */
.ar-ref {{
    display:inline-block; background:var(--bg3); border:1px solid var(--border);
    border-radius:4px; padding:3px 10px; font-size:11px; color:var(--text-muted);
    margin-left:12px;
}}
.ar-ref .ar-val {{ color:var(--orange); font-weight:600; }}

/* Tables */
table {{ width:100%; border-collapse:collapse; }}
th {{ background:var(--bg3); color:var(--text-muted); padding:8px 10px; text-align:center;
     font-size:11px; text-transform:uppercase; letter-spacing:0.5px; font-weight:600;
     border-bottom:2px solid var(--border); }}
td {{ padding:7px 10px; text-align:center; border-bottom:1px solid var(--border); font-size:13px; }}
tr:hover {{ background: var(--bg2); }}

/* Version badges */
.ver-badge {{
    display:inline-block; padding:2px 8px; border-radius:3px; font-size:11px;
    font-weight:700; color:#000; margin-right:8px;
}}
.ver-badge-sm {{
    display:inline-block; padding:1px 5px; border-radius:2px; font-size:10px;
    font-weight:700; color:#000;
}}
.ver-desc {{ color:var(--text-muted); font-size:11px; }}

/* Status badges */
.badge {{
    display:inline-block; padding:2px 8px; border-radius:10px; font-size:10px;
    font-weight:600; letter-spacing:0.3px;
}}
.badge-complete {{ background:rgba(63,185,80,0.15); color:var(--green); border:1px solid rgba(63,185,80,0.3); }}
.badge-partial {{ background:rgba(210,153,34,0.15); color:var(--yellow); border:1px solid rgba(210,153,34,0.3); }}
.badge-missing {{ background:rgba(248,81,73,0.15); color:var(--red); border:1px solid rgba(248,81,73,0.3); }}

/* Mini progress bars */
.mini-bar-container {{
    display:inline-block; width:60px; height:6px; background:var(--bg3);
    border-radius:3px; vertical-align:middle; margin-right:4px;
}}
.mini-bar {{ height:100%; background:var(--green); border-radius:3px; transition:width 0.3s; }}
.pct-text {{ font-size:11px; color:var(--text-muted); }}

/* Beats AR highlight */
.beats-ar {{ color:var(--green) !important; font-weight:600; }}

/* Matrix cells */
.cell-complete {{ color:var(--green); font-weight:600; }}
.cell-partial {{ color:var(--yellow); }}
.cell-partial .cell-total {{ color:var(--text-muted); font-size:11px; }}
.cell-missing {{ color:#484f58; }}
.uid-cell {{ text-align:left !important; font-weight:600; color:var(--text-bright); }}
.total-cell {{ color:var(--text-muted); }}
.target-name {{ text-align:left !important; font-weight:500; color:var(--text-bright); }}
.no-data {{ color:#484f58; }}

/* Metrics table first column left-align */
.metrics-table td:first-child {{ text-align:left; }}

/* Legend */
.legend {{
    display:flex; gap:16px; margin:12px 0; font-size:11px; color:var(--text-muted);
    flex-wrap:wrap;
}}
.legend-item {{ display:flex; align-items:center; gap:5px; }}
.legend-dot {{ width:10px; height:10px; border-radius:2px; }}

/* Footer */
.footer {{ text-align:center; color:#484f58; font-size:11px; padding:16px; margin-top:20px; }}

/* Responsive */
@media (max-width: 900px) {{
    .container {{ padding:12px; }}
    td, th {{ padding:5px 4px; font-size:11px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>Proactive Affective Agent — Results Dashboard</h1>
    <div class="header-meta">
        <span>Updated: <span class="val">{now}</span></span>
        <span>Auto-refresh: <span class="val">30s</span></span>
    </div>
</div>

<div class="container">

<!-- Aggregate Metrics -->
<div class="section">
    <div class="section-title">
        Aggregate Metrics
        <span class="ar-ref">AR baseline: BA=<span class="ar-val">{AR_BASELINE_BA:.3f}</span> &nbsp; F1=<span class="ar-val">{AR_BASELINE_F1:.3f}</span> &nbsp; MAE=<span class="ar-val">{AR_BASELINE_MAE:.3f}</span></span>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--green)"></div> Beats AR baseline</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--yellow)"></div> Partial data (still running)</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--red)"></div> No data</div>
    </div>

    <table class="metrics-table">
    <tr>
        <th style="text-align:left">Version</th>
        <th>Status</th>
        <th>Entries</th>
        <th>Coverage</th>
        <th>Mean MAE &darr;</th>
        <th>Mean BA &uarr;</th>
        <th>Mean F1 &uarr;</th>
    </tr>
    {metrics_rows}
    </table>
</div>

<!-- Completion Matrix -->
<div class="section">
    <div class="section-title">
        Completion Matrix
        <span class="section-subtitle">— entries completed per user &times; version</span>
    </div>

    <table>
    <tr>
        <th style="text-align:left">User</th>
        <th>Total</th>
        {matrix_header}
    </tr>
    {matrix_rows}
    </table>
</div>

<!-- Per-Target Breakdown -->
<div class="section">
    <div class="section-title">
        Per-Target Breakdown
        <span class="section-subtitle">— BA for binary targets, MAE for continuous (green = beats AR)</span>
    </div>

    <table>
    <tr>
        <th style="text-align:left">Target</th>
        {ver_headers}
    </tr>
    {target_rows}
    </table>
</div>

</div>

<div class="footer">
    Reads checkpoint files from outputs/pilot_v2/ and outputs/pilot/ on each refresh.
    Metrics computed live — partial data is clearly marked. Refresh to see latest results.
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            # JSON API endpoint for programmatic access
            data = build_dashboard_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2, default=str).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(build_html().encode())

    def log_message(self, fmt, *args):
        pass  # suppress access logs


def main():
    parser = argparse.ArgumentParser(description="Live results dashboard")
    parser.add_argument("--port", type=int, default=PORT, help="Port to serve on")
    args = parser.parse_args()

    print(f"Results dashboard: http://localhost:{args.port}")
    print(f"JSON API:          http://localhost:{args.port}/api/data")
    print(f"Reading from: {[str(d) for d in CHECKPOINT_DIRS]}")
    print("Ctrl+C to stop")
    HTTPServer(("", args.port), DashboardHandler).serve_forever()


if __name__ == "__main__":
    main()
