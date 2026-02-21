#!/usr/bin/env python3
"""Visualize intermediate results for User 71 from running experiments.

Reads checkpoint data and trace files to show:
1. Prediction vs ground truth time series
2. Per-version comparison
3. Agent reasoning traces (what's available)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "pilot" / "checkpoints"
TRACES_DIR = PROJECT_ROOT / "outputs" / "pilot" / "traces"


def load_checkpoint(version: str) -> dict:
    path = CHECKPOINT_DIR / f"{version}_checkpoint.json"
    if not path.exists():
        return {"predictions": [], "ground_truths": [], "metadata": []}
    with open(path) as f:
        return json.load(f)


def filter_user(data: dict, study_id: int) -> tuple[list, list, list]:
    preds, gts, metas = [], [], []
    for p, g, m in zip(data["predictions"], data["ground_truths"], data["metadata"]):
        if m.get("study_id") == study_id:
            preds.append(p)
            gts.append(g)
            metas.append(m)
    return preds, gts, metas


def load_traces(version: str, study_id: int) -> list[dict]:
    traces = []
    for f in sorted(TRACES_DIR.glob(f"{version}_user{study_id}_*.json")):
        with open(f) as fh:
            traces.append(json.load(fh))
    return traces


def is_real_prediction(preds: list) -> bool:
    """Check if predictions are real (varied) vs dry-run (all same)."""
    if len(preds) < 3:
        return True
    vals = [p.get("PANAS_Pos") for p in preds[:10] if p.get("PANAS_Pos") is not None]
    return len(set(vals)) > 1


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_time_series(preds, gts, metas, label: str):
    """Print prediction vs ground truth for continuous targets."""
    print(f"\n--- {label}: PANAS_Pos / PANAS_Neg / ER_desire ---")
    print(f"{'Date':<12} {'PA_pred':>8} {'PA_true':>8} {'NA_pred':>8} {'NA_true':>8} {'ER_pred':>8} {'ER_true':>8} {'Avail_p':>8} {'Avail_t':>8}")
    print("-" * 92)

    for p, g, m in zip(preds, gts, metas):
        date = m.get("date", "?")[:10]
        pa_p = p.get("PANAS_Pos", "?")
        pa_t = g.get("PANAS_Pos", "?")
        na_p = p.get("PANAS_Neg", "?")
        na_t = g.get("PANAS_Neg", "?")
        er_p = p.get("ER_desire", "?")
        er_t = g.get("ER_desire", "?")
        av_p = p.get("INT_availability", "?")
        av_t = g.get("INT_availability", "?")

        def fmt(v):
            if v is None or v == "?":
                return "?"
            if isinstance(v, float):
                return f"{v:.1f}"
            return str(v)

        print(f"{date:<12} {fmt(pa_p):>8} {fmt(pa_t):>8} {fmt(na_p):>8} {fmt(na_t):>8} {fmt(er_p):>8} {fmt(er_t):>8} {str(av_p):>8} {str(av_t):>8}")


def print_trace_examples(traces: list, version: str, max_show: int = 3):
    """Print detailed trace examples."""
    print(f"\n--- {version.upper()}: Agent Reasoning Traces (first {max_show} entries) ---")

    for i, trace in enumerate(traces[:max_show]):
        print(f"\n  Entry {i}:")
        if version == "callm":
            print(f"    Diary text: \"{trace.get('_emotion_driver', 'N/A')}\"")
            print(f"    Prompt length: {trace.get('_prompt_length', '?')} chars")
            if trace.get("_rag_top5"):
                print(f"    Top RAG results:")
                for j, r in enumerate(trace["_rag_top5"]):
                    print(f"      {j+1}. sim={r.get('similarity', '?'):.2f} | \"{r.get('text', '')[:80]}...\"")
            if trace.get("_full_response"):
                resp = trace["_full_response"]
                # Show reasoning part (before JSON)
                json_start = resp.find("{")
                if json_start > 0:
                    print(f"    Reasoning: {resp[:json_start].strip()[:300]}")

        elif version == "v1":
            print(f"    Sensing summary: {trace.get('_sensing_summary', 'N/A')[:200]}...")
            print(f"    Prompt length: {trace.get('_prompt_length', '?')} chars")
            if trace.get("_full_response"):
                resp = trace["_full_response"]
                json_start = resp.find("{")
                if json_start > 0:
                    print(f"    Reasoning: {resp[:json_start].strip()[:300]}")

        elif version == "v2":
            print(f"    LLM calls: {trace.get('_llm_calls', '?')}")
            for step in trace.get("_trace", []):
                r = step.get("round", "?")
                resp = step.get("response", "")[:300]
                req = step.get("request", "")
                print(f"    Round {r}:")
                if req:
                    print(f"      Request: {req}")
                print(f"      Response: {resp}...")


def compute_mae(preds, gts, target: str) -> float | None:
    vals = []
    for p, g in zip(preds, gts):
        pv = p.get(target)
        gv = g.get(target)
        if pv is not None and gv is not None:
            try:
                vals.append(abs(float(pv) - float(gv)))
            except (ValueError, TypeError):
                pass
    return sum(vals) / len(vals) if vals else None


def print_comparison(all_data: dict, study_id: int):
    """Print side-by-side comparison of 3 versions."""
    print_header(f"User {study_id} — Version Comparison")

    for version in ["callm", "v1", "v2"]:
        preds, gts, metas = all_data.get(version, ([], [], []))
        if not preds:
            print(f"\n  {version.upper()}: No data yet")
            continue

        real = is_real_prediction(preds)
        status = "REAL LLM" if real else "DRY-RUN (placeholder)"
        n = len(preds)

        mae_pa = compute_mae(preds, gts, "PANAS_Pos")
        mae_na = compute_mae(preds, gts, "PANAS_Neg")
        mae_er = compute_mae(preds, gts, "ER_desire")

        print(f"\n  {version.upper()} [{status}] — {n} entries:")
        if mae_pa is not None:
            print(f"    MAE PANAS_Pos: {mae_pa:.2f}")
        if mae_na is not None:
            print(f"    MAE PANAS_Neg: {mae_na:.2f}")
        if mae_er is not None:
            print(f"    MAE ER_desire: {mae_er:.2f}")
        if mae_pa and mae_na and mae_er:
            print(f"    Mean MAE: {(mae_pa + mae_na + mae_er) / 3:.2f}")


def main():
    study_id = 71

    print_header(f"PILOT EXPERIMENT — User {study_id} Intermediate Results")

    all_data = {}
    for version in ["callm", "v1", "v2"]:
        data = load_checkpoint(version)
        preds, gts, metas = filter_user(data, study_id)
        all_data[version] = (preds, gts, metas)
        print(f"  {version.upper()}: {len(preds)} entries in checkpoint")

    # Show comparison
    print_comparison(all_data, study_id)

    # Show time series for each version (first 15 entries)
    for version in ["callm", "v1", "v2"]:
        preds, gts, metas = all_data[version]
        if preds and is_real_prediction(preds):
            print_time_series(preds[:15], gts[:15], metas[:15], version.upper())

    # Show trace examples
    for version in ["callm", "v1", "v2"]:
        traces = load_traces(version, study_id)
        if traces:
            print_trace_examples(traces, version, max_show=3)

    # ASCII chart for PANAS_Pos
    for version in ["callm", "v1", "v2"]:
        preds, gts, metas = all_data[version]
        if preds and is_real_prediction(preds):
            print(f"\n--- {version.upper()}: PANAS_Pos (pred=* true=o) ---")
            for i, (p, g, m) in enumerate(zip(preds[:20], gts[:20], metas[:20])):
                pv = p.get("PANAS_Pos")
                gv = g.get("PANAS_Pos")
                if pv is None or gv is None:
                    continue
                pv, gv = float(pv), float(gv)
                bar_len = 30
                p_pos = int(pv / 30 * bar_len)
                g_pos = int(gv / 30 * bar_len)
                bar = [" "] * (bar_len + 1)
                bar[g_pos] = "o"
                bar[p_pos] = "*" if p_pos != g_pos else "X"  # X = exact match
                print(f"  {i:2d} |{''.join(bar)}| p={pv:.0f} t={gv:.0f} e={abs(pv-gv):.0f}")


if __name__ == "__main__":
    main()
