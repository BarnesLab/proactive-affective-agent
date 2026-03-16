#!/usr/bin/env python3
"""Live progress dashboard for Codex-mini pilot experiment.

Serves a self-refreshing HTML table at http://localhost:8878
and treats fallback / empty / tool-refusal outputs as errors.
"""

import json
import subprocess
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOTS = [
    PROJECT_ROOT / "outputs" / "pilot_codexmini",
    PROJECT_ROOT / "outputs" / "pilot_codexmini_v2",
]
VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
USERS_BY_ROOT = {
    "pilot_codexmini": [71, 458, 310, 164, 119],
    "pilot_codexmini_v2": [399, 258, 43, 403, 338],
}
USER_TOTALS_BY_ROOT = {
    "pilot_codexmini": {71: 93, 458: 82, 310: 81, 164: 87, 119: 84},
    "pilot_codexmini_v2": {399: 96, 258: 94, 43: 93, 403: 82, 338: 81},
}
PORT = 8878
REQUIRED_KEYS = ("PANAS_Pos", "PANAS_Neg", "ER_desire", "INT_availability", "confidence")
TOOL_REFUSAL_MARKERS = (
    "tools unavailable",
    "tool is unavailable",
    "need to be enabled",
    "enabling the tool calls",
    "system isn’t letting me issue requests",
    "system isn't letting me issue requests",
)


def is_valid_prediction(pred: object) -> bool:
    if not isinstance(pred, dict):
        return False
    if not all(k in pred for k in REQUIRED_KEYS):
        return False
    reasoning = str(pred.get("reasoning", "") or "")
    return "fallback" not in reasoning.lower()


def classify_record(obj: dict) -> bool:
    pred = obj.get("prediction")
    if not is_valid_prediction(pred):
        return False
    full_response = str(obj.get("full_response", "") or "").lower()
    reasoning = str(obj.get("reasoning", "") or "").lower()
    if any(marker in full_response for marker in TOOL_REFUSAL_MARKERS):
        return False
    if any(marker in reasoning for marker in TOOL_REFUSAL_MARKERS):
        return False
    return True


def _checkpoint_candidates(output_dir: Path, version: str, uid: int) -> list[Path]:
    ckpt_prefix = f"gpt-{version}"
    return [
        output_dir / "groupA" / "checkpoints" / f"{ckpt_prefix}_user{uid}_checkpoint.json",
        output_dir / "groupB" / "checkpoints" / f"{ckpt_prefix}_user{uid}_checkpoint.json",
        output_dir / "checkpoints" / f"{ckpt_prefix}_user{uid}_checkpoint.json",
    ]


def _record_candidates(output_dir: Path, version: str, uid: int) -> list[Path]:
    rec_prefix = f"gpt-{version}_user{uid}_records.jsonl"
    return [
        output_dir / "groupA" / rec_prefix,
        output_dir / "groupB" / rec_prefix,
        output_dir / rec_prefix,
    ]


def get_progress(output_dir: Path, users: list[int]):
    """Collect progress data from checkpoints + records."""
    rows = []
    total_ok = 0
    total_err = 0

    for v in VERSIONS:
        ok = 0
        err = 0
        last_entry = ""

        cp_done = {}
        for uid in users:
            cp_candidates = _checkpoint_candidates(output_dir, v, uid)
            done = 0
            best_payload = None
            for cp in cp_candidates:
                if not cp.exists():
                    continue
                try:
                    d = json.loads(cp.read_text())
                    preds = d.get("predictions", [])
                    if len(preds) >= done:
                        done = len(preds)
                        best_payload = d
                except Exception:
                    pass
            cp_done[uid] = done

            if best_payload:
                meta = best_payload.get("metadata", [])
                if isinstance(meta, list) and meta:
                    last = meta[-1]
                    if isinstance(last, dict):
                        last_date = last.get("date", "")
                        if last_date:
                            last_entry = max(last_entry, str(last_date))

            rec_candidates = _record_candidates(output_dir, v, uid)
            best_records = None
            for rec in rec_candidates:
                if not rec.exists():
                    continue
                try:
                    lines = [json.loads(line) for line in rec.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if best_records is None or len(lines) > len(best_records):
                        best_records = lines
                except Exception:
                    continue
            if best_records is not None:
                ok += sum(1 for row in best_records if classify_record(row))
                err += sum(1 for row in best_records if not classify_record(row))
            elif best_payload:
                preds = best_payload.get("predictions", [])
                ok += sum(1 for pred in preds if is_valid_prediction(pred))
                err += sum(1 for pred in preds if not is_valid_prediction(pred))

        total_ok += ok
        total_err += err
        rows.append({
            "version": v.upper(),
            "ok": ok,
            "err": err,
            "last": last_entry,
            "cp": cp_done,
        })

    return rows, total_ok, total_err


def build_section(output_dir: Path, users: list[int]) -> tuple[str, int, int, int]:
    rows, total_ok, total_err = get_progress(output_dir, users)
    user_totals = USER_TOTALS_BY_ROOT[output_dir.name]
    grand_total = sum(user_totals.values())

    user_headers = "".join(f"<th>U{uid}</th>" for uid in users)
    table_rows = ""
    all_done = 0
    for row in rows:
        cp = row["cp"]
        cells = ""
        v_total = 0
        for uid in users:
            done = cp.get(uid, 0)
            total = user_totals.get(uid, 0)
            v_total += done
            if done == 0:
                cls = "not-started"
                txt = "-"
            elif total and done >= total:
                cls = "complete"
                txt = f"{done}/{total}"
            else:
                cls = "in-progress"
                pct = 100 * done / total if total else 0
                txt = f"{done}/{total} ({pct:.0f}%)" if total else str(done)
            cells += f'<td class="{cls}">{txt}</td>'

        all_done += v_total
        v_pct = 100 * v_total / grand_total if grand_total else 0
        err_cls = ' class="has-errors"' if row["err"] > 0 else ""
        table_rows += f"""<tr>
            <td><strong>{row['version']}</strong></td>
            <td>{v_total}/{grand_total} ({v_pct:.0f}%)</td>
            <td>{row['ok']}</td>
            <td{err_cls}>{row['err']}</td>
            {cells}
            <td class="last-entry">{row['last']}</td>
        </tr>\n"""

    section_html = f"""
<h2>{output_dir.name}</h2>
<table>
<tr>
    <th>Version</th>
    <th>Total</th>
    <th>OK</th>
    <th>Err</th>
    {user_headers}
    <th style=\"text-align:left\">Latest</th>
</tr>
{table_rows}
</table>
"""
    return section_html, all_done, grand_total * len(VERSIONS), total_err


def build_html():
    now = time.strftime("%H:%M:%S")

    r = subprocess.run(["pgrep", "-f", "run_pilot"], capture_output=True, text=True)
    procs = len(r.stdout.strip().split("\n")) if r.stdout.strip() else 0

    # Usage stats
    usage_path = Path.home() / ".claude" / "hooks" / "usage-stats.json"
    pace = "?"
    rate = "?"
    try:
        u = json.loads(usage_path.read_text())
        pace = u.get("pace", "?")
        rate = u.get("rate_per_hour", "?")
    except Exception:
        pass

    sections = []
    all_done = 0
    all_total = 0
    all_err = 0
    for output_dir in OUTPUT_ROOTS:
        section_html, section_done, section_total, section_err = build_section(
            output_dir, USERS_BY_ROOT[output_dir.name]
        )
        sections.append(section_html)
        all_done += section_done
        all_total += section_total
        all_err += section_err
    all_pct = 100 * all_done / all_total if all_total else 0

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>Codex-Mini Pilot Progress</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #64ffda; margin-bottom: 5px; }}
  .meta {{ color: #888; margin-bottom: 20px; font-size: 14px; }}
  .meta span {{ margin-right: 20px; }}
  .meta .val {{ color: #64ffda; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #16213e; color: #64ffda; padding: 10px 8px; text-align: center; font-size: 13px; }}
  td {{ padding: 8px; text-align: center; border-bottom: 1px solid #2a2a4a; font-size: 13px; }}
  tr:hover {{ background: #16213e; }}
  .not-started {{ color: #555; }}
  .in-progress {{ color: #ffd700; }}
  .complete {{ color: #00e676; font-weight: bold; }}
  .has-errors {{ color: #ff5252; }}
  .last-entry {{ color: #888; font-size: 12px; text-align: left; max-width: 200px; }}
  .bar-container {{ width: 100%; background: #2a2a4a; border-radius: 4px; height: 30px; margin: 15px 0; }}
  .bar {{ height: 100%; background: linear-gradient(90deg, #64ffda, #00e676); border-radius: 4px;
          display: flex; align-items: center; justify-content: center; color: #1a1a2e; font-weight: bold; font-size: 14px;
          transition: width 0.5s ease; }}
</style>
</head>
<body>
<h1>Codex-Mini Pilot Dashboard</h1>
<div class="meta">
    <span>Updated: <span class="val">{now}</span></span>
    <span>Processes: <span class="val">{procs}</span></span>
    <span>Pace: <span class="val">{pace}</span></span>
    <span>Rate: <span class="val">{rate}/hr</span></span>
    <span>Total: <span class="val">{all_done}/{all_total} ({all_pct:.1f}%)</span></span>
    <span>Err: <span class="val">{all_err}</span></span>
</div>
<div class="bar-container">
    <div class="bar" style="width: {max(all_pct, 2):.1f}%">{all_pct:.1f}%</div>
</div>
{"".join(sections)}
<p style="color:#555; font-size:12px; margin-top:20px;">Auto-refreshes every 30s</p>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(build_html().encode())

    def log_message(self, format, *args):
        pass  # suppress access logs


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}")
    HTTPServer(("", PORT), Handler).serve_forever()
