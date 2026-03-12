#!/usr/bin/env python3
"""Haiku vs Sonnet comparison dashboard.

Shows checkpoint completion status AND performance comparison between models.
Serves at http://localhost:8877
"""

import json
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HAIKU_DIR = PROJECT_ROOT / "outputs" / "pilot_haiku"
SONNET_DIR = PROJECT_ROOT / "outputs" / "pilot"
VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
USERS = [71, 458, 310, 164, 119]
USER_TOTALS = {71: 93, 458: 82, 310: 81, 164: 87, 119: 84}
PORT = 8877

# Haiku results (from evaluate_pilot.py)
HAIKU = {
    "callm": {"mae": 3.566, "ba": 0.620, "f1": 0.603},
    "v1":    {"mae": 5.067, "ba": 0.515, "f1": 0.482},
    "v2":    {"mae": 6.324, "ba": 0.529, "f1": 0.489},
    "v3":    {"mae": 3.579, "ba": 0.598, "f1": 0.586},
    "v4":    {"mae": 5.977, "ba": 0.639, "f1": 0.627},
    "v5":    {"mae": 6.982, "ba": 0.537, "f1": 0.512},
    "v6":    {"mae": 6.353, "ba": 0.621, "f1": 0.606},
}

# Sonnet results (from evaluate_pilot.py)
SONNET = {
    "callm": {"mae": 3.528, "ba": 0.615, "f1": 0.601},
    "v1":    {"mae": 6.854, "ba": 0.534, "f1": 0.489},
    "v2":    {"mae": 5.881, "ba": 0.596, "f1": 0.578},
    "v3":    {"mae": 3.965, "ba": 0.617, "f1": 0.606},
    "v4":    {"mae": 5.259, "ba": 0.670, "f1": 0.656},
    "v5":    {"mae": 5.375, "ba": 0.593, "f1": 0.571},
    "v6":    {"mae": 4.492, "ba": 0.676, "f1": 0.656},
}

# Per-target BA for Haiku
HAIKU_BA = {
    "callm": {"happy": 0.749, "PA": 0.534, "NA": 0.696, "sad": 0.591, "worried": 0.709, "avail": 0.545},
    "v1":    {"happy": 0.514, "PA": 0.528, "NA": 0.507, "sad": 0.543, "worried": 0.508, "avail": 0.488},
    "v2":    {"happy": 0.501, "PA": 0.517, "NA": 0.554, "sad": 0.553, "worried": 0.559, "avail": 0.626},
    "v3":    {"happy": 0.698, "PA": 0.519, "NA": 0.632, "sad": 0.604, "worried": 0.690, "avail": 0.475},
    "v4":    {"happy": 0.663, "PA": 0.677, "NA": 0.723, "sad": 0.682, "worried": 0.699, "avail": 0.680},
    "v5":    {"happy": 0.537, "PA": 0.542, "NA": 0.577, "sad": 0.560, "worried": 0.570, "avail": 0.623},
    "v6":    {"happy": 0.661, "PA": 0.684, "NA": 0.755, "sad": 0.646, "worried": 0.689, "avail": 0.632},
}

# Per-target BA for Sonnet
SONNET_BA = {
    "callm": {"happy": 0.696, "PA": 0.558, "NA": 0.685, "sad": 0.644, "worried": 0.695, "avail": 0.531},
    "v1":    {"happy": 0.544, "PA": 0.532, "NA": 0.548, "sad": 0.596, "worried": 0.533, "avail": 0.486},
    "v2":    {"happy": 0.596, "PA": 0.597, "NA": 0.674, "sad": 0.640, "worried": 0.649, "avail": 0.738},
    "v3":    {"happy": 0.651, "PA": 0.571, "NA": 0.724, "sad": 0.687, "worried": 0.679, "avail": 0.580},
    "v4":    {"happy": 0.732, "PA": 0.738, "NA": 0.770, "sad": 0.680, "worried": 0.747, "avail": 0.780},
    "v5":    {"happy": 0.573, "PA": 0.601, "NA": 0.642, "sad": 0.612, "worried": 0.614, "avail": 0.781},
    "v6":    {"happy": 0.725, "PA": 0.735, "NA": 0.806, "sad": 0.702, "worried": 0.728, "avail": 0.775},
}

VERSION_LABELS = {
    "callm": "CALLM (diary+RAG)",
    "v1": "V1 (structured, sensing)",
    "v2": "V2 (agentic, sensing)",
    "v3": "V3 (structured, multimodal)",
    "v4": "V4 (agentic, multimodal)",
    "v5": "V5 (agentic, sensing+filtered)",
    "v6": "V6 (agentic, multimodal+filtered)",
}


def delta_html(haiku_val, sonnet_val, lower_better=False):
    """Return colored delta string."""
    d = haiku_val - sonnet_val
    if abs(d) < 0.005:
        return '<span style="color:#888">~0</span>'
    if lower_better:
        better = d < 0
    else:
        better = d > 0
    color = "#00e676" if better else "#ff5252"
    sign = "+" if d > 0 else ""
    return f'<span style="color:{color}">{sign}{d:.3f}</span>'


def build_html():
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    # === Aggregate comparison table ===
    agg_rows = ""
    for v in VERSIONS:
        h = HAIKU[v]
        s = SONNET[v]
        label = VERSION_LABELS[v]

        # Highlight best in each metric
        agg_rows += f"""<tr>
            <td style="text-align:left"><strong>{label}</strong></td>
            <td>{h['mae']:.3f}</td><td>{s['mae']:.3f}</td><td>{delta_html(h['mae'], s['mae'], lower_better=True)}</td>
            <td>{h['ba']:.3f}</td><td>{s['ba']:.3f}</td><td>{delta_html(h['ba'], s['ba'])}</td>
            <td>{h['f1']:.3f}</td><td>{s['f1']:.3f}</td><td>{delta_html(h['f1'], s['f1'])}</td>
        </tr>\n"""

    # Averages
    h_mae_avg = sum(HAIKU[v]["mae"] for v in VERSIONS) / len(VERSIONS)
    s_mae_avg = sum(SONNET[v]["mae"] for v in VERSIONS) / len(VERSIONS)
    h_ba_avg = sum(HAIKU[v]["ba"] for v in VERSIONS) / len(VERSIONS)
    s_ba_avg = sum(SONNET[v]["ba"] for v in VERSIONS) / len(VERSIONS)
    h_f1_avg = sum(HAIKU[v]["f1"] for v in VERSIONS) / len(VERSIONS)
    s_f1_avg = sum(SONNET[v]["f1"] for v in VERSIONS) / len(VERSIONS)

    agg_rows += f"""<tr style="border-top:2px solid #64ffda">
        <td style="text-align:left"><strong>Average</strong></td>
        <td><strong>{h_mae_avg:.3f}</strong></td><td><strong>{s_mae_avg:.3f}</strong></td><td>{delta_html(h_mae_avg, s_mae_avg, lower_better=True)}</td>
        <td><strong>{h_ba_avg:.3f}</strong></td><td><strong>{s_ba_avg:.3f}</strong></td><td>{delta_html(h_ba_avg, s_ba_avg)}</td>
        <td><strong>{h_f1_avg:.3f}</strong></td><td><strong>{s_f1_avg:.3f}</strong></td><td>{delta_html(h_f1_avg, s_f1_avg)}</td>
    </tr>"""

    # === Per-target BA comparison ===
    targets = ["happy", "PA", "NA", "sad", "worried", "avail"]
    target_labels = ["Happy", "Pos Affect", "Neg Affect", "Sad", "Worried", "Availability"]

    target_rows = ""
    for v in VERSIONS:
        hb = HAIKU_BA[v]
        sb = SONNET_BA[v]
        cells = ""
        for t in targets:
            cells += f"<td>{hb[t]:.3f}</td><td>{sb[t]:.3f}</td><td>{delta_html(hb[t], sb[t])}</td>"
        target_rows += f"""<tr>
            <td style="text-align:left"><strong>{v.upper()}</strong></td>
            {cells}
        </tr>\n"""

    # Per-target averages
    tavg_cells = ""
    for t in targets:
        h_avg = sum(HAIKU_BA[v][t] for v in VERSIONS) / len(VERSIONS)
        s_avg = sum(SONNET_BA[v][t] for v in VERSIONS) / len(VERSIONS)
        tavg_cells += f"<td><strong>{h_avg:.3f}</strong></td><td><strong>{s_avg:.3f}</strong></td><td>{delta_html(h_avg, s_avg)}</td>"
    target_rows += f"""<tr style="border-top:2px solid #64ffda">
        <td style="text-align:left"><strong>Avg</strong></td>
        {tavg_cells}
    </tr>"""

    target_headers = ""
    for tl in target_labels:
        target_headers += f'<th colspan="3">{tl}</th>'

    target_subheaders = '<th style="text-align:left">Version</th>'
    for _ in targets:
        target_subheaders += '<th>H</th><th>S</th><th>Δ</th>'

    # === Key findings ===
    # Count wins
    haiku_wins_ba = sum(1 for v in VERSIONS if HAIKU[v]["ba"] > SONNET[v]["ba"])
    sonnet_wins_ba = sum(1 for v in VERSIONS if SONNET[v]["ba"] > HAIKU[v]["ba"])
    haiku_wins_mae = sum(1 for v in VERSIONS if HAIKU[v]["mae"] < SONNET[v]["mae"])
    sonnet_wins_mae = sum(1 for v in VERSIONS if SONNET[v]["mae"] < HAIKU[v]["mae"])

    # Best overall
    best_ba_h = max(VERSIONS, key=lambda v: HAIKU[v]["ba"])
    best_ba_s = max(VERSIONS, key=lambda v: SONNET[v]["ba"])
    best_mae_h = min(VERSIONS, key=lambda v: HAIKU[v]["mae"])
    best_mae_s = min(VERSIONS, key=lambda v: SONNET[v]["mae"])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Haiku vs Sonnet Comparison</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; max-width: 1400px; }}
  h1 {{ color: #64ffda; margin-bottom: 5px; }}
  h2 {{ color: #bb86fc; margin-top: 30px; margin-bottom: 10px; font-size: 18px; }}
  .timestamp {{ color: #666; font-size: 13px; margin-bottom: 25px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #16213e; color: #64ffda; padding: 8px 6px; text-align: center; font-size: 12px; white-space: nowrap; }}
  td {{ padding: 7px 6px; text-align: center; border-bottom: 1px solid #2a2a4a; font-size: 13px; }}
  tr:hover {{ background: #16213e; }}
  .findings {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 15px; border-left: 3px solid #64ffda; }}
  .card h3 {{ color: #64ffda; margin: 0 0 8px 0; font-size: 14px; }}
  .card p {{ margin: 4px 0; font-size: 13px; color: #ccc; }}
  .card .val {{ font-weight: bold; }}
  .win {{ color: #00e676; font-weight: bold; }}
  .lose {{ color: #ff5252; }}
  .tie {{ color: #ffd700; }}
  .section-note {{ color: #888; font-size: 12px; margin-bottom: 8px; }}
  .legend {{ font-size: 12px; color: #888; margin-top: 8px; }}
</style>
</head>
<body>
<h1>Haiku vs Sonnet — Pilot Comparison</h1>
<div class="timestamp">5 users (71, 458, 310, 164, 119) x 7 versions | 427 predictions each | Updated: {now}</div>

<div class="findings">
    <div class="card">
        <h3>Balanced Accuracy Wins</h3>
        <p>Haiku: <span class="val {'win' if haiku_wins_ba > sonnet_wins_ba else 'lose'}">{haiku_wins_ba}/7</span> &nbsp; Sonnet: <span class="val {'win' if sonnet_wins_ba > haiku_wins_ba else 'lose'}">{sonnet_wins_ba}/7</span></p>
        <p>Best Haiku: {best_ba_h.upper()} ({HAIKU[best_ba_h]['ba']:.3f})</p>
        <p>Best Sonnet: {best_ba_s.upper()} ({SONNET[best_ba_s]['ba']:.3f})</p>
    </div>
    <div class="card">
        <h3>MAE Wins (lower = better)</h3>
        <p>Haiku: <span class="val {'win' if haiku_wins_mae > sonnet_wins_mae else 'lose'}">{haiku_wins_mae}/7</span> &nbsp; Sonnet: <span class="val {'win' if sonnet_wins_mae > haiku_wins_mae else 'lose'}">{sonnet_wins_mae}/7</span></p>
        <p>Best Haiku: {best_mae_h.upper()} ({HAIKU[best_mae_h]['mae']:.3f})</p>
        <p>Best Sonnet: {best_mae_s.upper()} ({SONNET[best_mae_s]['mae']:.3f})</p>
    </div>
    <div class="card">
        <h3>Average Performance</h3>
        <p>BA: Haiku <span class="val">{h_ba_avg:.3f}</span> vs Sonnet <span class="val">{s_ba_avg:.3f}</span> ({delta_html(h_ba_avg, s_ba_avg)})</p>
        <p>F1: Haiku <span class="val">{h_f1_avg:.3f}</span> vs Sonnet <span class="val">{s_f1_avg:.3f}</span> ({delta_html(h_f1_avg, s_f1_avg)})</p>
        <p>MAE: Haiku <span class="val">{h_mae_avg:.3f}</span> vs Sonnet <span class="val">{s_mae_avg:.3f}</span> ({delta_html(h_mae_avg, s_mae_avg, lower_better=True)})</p>
    </div>
    <div class="card">
        <h3>Key Patterns</h3>
        <p>Agentic versions (V2/V4/V5/V6): Sonnet advantage in tool use</p>
        <p>Structured (CALLM/V1/V3): Haiku competitive or better on MAE</p>
        <p>AR baseline: BA=0.658, PA_MAE=2.758</p>
    </div>
</div>

<h2>Aggregate Metrics</h2>
<p class="section-note">H = Haiku, S = Sonnet, Delta = Haiku - Sonnet. <span style="color:#00e676">Green</span> = Haiku better, <span style="color:#ff5252">Red</span> = Sonnet better.</p>
<table>
<tr>
    <th style="text-align:left">Version</th>
    <th colspan="3">Mean MAE (lower better)</th>
    <th colspan="3">Mean BA (higher better)</th>
    <th colspan="3">Mean F1 (higher better)</th>
</tr>
<tr>
    <th></th><th>H</th><th>S</th><th>Δ</th><th>H</th><th>S</th><th>Δ</th><th>H</th><th>S</th><th>Δ</th>
</tr>
{agg_rows}
</table>

<h2>Per-Target Balanced Accuracy</h2>
<p class="section-note">H = Haiku, S = Sonnet, Δ = Haiku - Sonnet</p>
<table>
<tr>
    <th></th>
    {target_headers}
</tr>
<tr>
    {target_subheaders}
</tr>
{target_rows}
</table>

<div class="legend">
    <p>AR baseline: BA=0.658 | ML best (MiniLM): BA=0.629, F1=0.588 | All models evaluated on same 5 users, 427 entries</p>
</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(build_html().encode())

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"Haiku vs Sonnet Dashboard: http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()
