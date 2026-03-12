#!/usr/bin/env python3
"""Evaluate temporal learning effects in agent predictions.

Tests whether prediction accuracy improves over time as agents accumulate
session context (memory of user patterns, receptivity signals).

Methodology
-----------
**Primary — Tertile Block Analysis:**
  Split each user's chronological entries into 3 equal blocks (T1/T2/T3).
  Compute mean Balanced Accuracy per block across 6 binary targets.
  Test H1: T3 > T1 via one-sided Wilcoxon signed-rank test.
  Report: mean delta, 95% bootstrap CI, Cohen's d, Holm-corrected p-values.

**Secondary — 5-Block Spearman Trend:**
  Split into 5 blocks. Per-user Spearman correlation (block_index x BA).
  Aggregate and test if mean rho > 0.

**Cross-version — Friedman Test:**
  Test whether learning effects (delta) differ across versions.

**Per-target Breakdown:**
  Which targets show the strongest temporal improvement?

Usage:
    PYTHONPATH=. python3 scripts/evaluate_learning_curve.py
    PYTHONPATH=. python3 scripts/evaluate_learning_curve.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

# Suppress expected warning for small sample sizes in Wilcoxon test
warnings.filterwarnings("ignore", message="Sample size too small for normal approximation")

from src.utils.mappings import BINARY_STATE_TARGETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PILOT_USERS = [43, 71, 119, 164, 258, 275, 310, 338, 403, 458, 513]
VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
VERSION_META = {
    "callm": {"label": "CALLM", "strategy": "Structured", "data": "Diary+RAG"},
    "v1":    {"label": "V1",    "strategy": "Structured", "data": "Sensing"},
    "v2":    {"label": "V2",    "strategy": "Agentic",    "data": "Sensing"},
    "v3":    {"label": "V3",    "strategy": "Structured", "data": "Multimodal"},
    "v4":    {"label": "V4",    "strategy": "Agentic",    "data": "Multimodal"},
    "v5":    {"label": "V5",    "strategy": "Agentic",    "data": "Sens+Filt"},
    "v6":    {"label": "V6",    "strategy": "Agentic",    "data": "Multi+Filt"},
}
DEFAULT_CHECKPOINT_DIRS = [
    "outputs/pilot_v2/checkpoints",
    "outputs/pilot/checkpoints",
]

# Binary targets for BA computation (matches evaluate_pilot.py KEY_BINARY)
KEY_BINARY = [
    "Individual_level_happy_State",
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_sad_State",
    "Individual_level_worried_State",
    "INT_availability",
]

# Short names for display
TARGET_SHORT = {
    "Individual_level_happy_State": "Happy",
    "Individual_level_PA_State": "Pos Affect",
    "Individual_level_NA_State": "Neg Affect",
    "Individual_level_sad_State": "Sad",
    "Individual_level_worried_State": "Worried",
    "INT_availability": "INT Avail",
}

AR_BASELINE_BA = 0.658
N_BOOTSTRAP = 10000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Data loading helpers (mirrors evaluate_pilot.py)
# ---------------------------------------------------------------------------

def find_checkpoint(version: str, uid: int, cp_dirs: list[Path]) -> Path | None:
    names = [f"{version}_user{uid}_checkpoint.json"]
    if version.lower() == "callm":
        names.append(f"CALLM_user{uid}_checkpoint.json")
    for cp_dir in cp_dirs:
        for name in names:
            f = cp_dir / name
            if f.exists():
                return f
    return None


def _to_bool(v) -> bool | None:
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


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_block_ba(preds: list[dict], gts: list[dict], targets: list[str]) -> float | None:
    """Compute mean BA across binary targets for a block of entries."""
    target_bas = []
    for target in targets:
        y_true, y_pred = [], []
        for p, g in zip(preds, gts):
            gv = _to_bool(g.get(target))
            pv = _to_bool(p.get(target))
            if gv is not None and pv is not None:
                y_true.append(int(gv))
                y_pred.append(int(pv))
        if len(y_true) >= 5 and len(set(y_true)) > 1:
            target_bas.append(balanced_accuracy_score(y_true, y_pred))
    return float(np.mean(target_bas)) if len(target_bas) >= 2 else None


def compute_single_target_ba(
    preds: list[dict], gts: list[dict], target: str
) -> float | None:
    """Compute BA for a single target within a block."""
    y_true, y_pred = [], []
    for p, g in zip(preds, gts):
        gv = _to_bool(g.get(target))
        pv = _to_bool(p.get(target))
        if gv is not None and pv is not None:
            y_true.append(int(gv))
            y_pred.append(int(pv))
    if len(y_true) >= 5 and len(set(y_true)) > 1:
        return float(balanced_accuracy_score(y_true, y_pred))
    return None


def split_blocks(preds, gts, n_blocks):
    """Split predictions/ground truths into n equal chronological blocks."""
    n = len(preds)
    block_size = n // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n
        blocks.append((preds[start:end], gts[start:end]))
    return blocks


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci_level=0.95):
    """Percentile bootstrap CI for the mean."""
    rng = np.random.default_rng(42)
    data = np.array(data)
    means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, (1 - ci_level) / 2 * 100)
    hi = np.percentile(means, (1 + ci_level) / 2 * 100)
    return float(lo), float(hi)


def cohens_d_one_sample(data):
    """Cohen's d for one-sample test against 0."""
    m = np.mean(data)
    s = np.std(data, ddof=1)
    return float(m / s) if s > 0 else 0.0


def holm_correct(p_values):
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    pv = np.array(p_values)
    order = np.argsort(pv)
    corrected = np.ones(n)
    prev_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = pv[idx] * (n - rank)
        adjusted = max(adjusted, prev_max)
        corrected[idx] = min(adjusted, 1.0)
        prev_max = corrected[idx]
    return corrected.tolist()


def wilcoxon_greater(data):
    """One-sided Wilcoxon signed-rank test: H1 median > 0."""
    data = np.array(data)
    if len(data) < 6 or np.all(data == 0):
        return np.nan, 1.0
    try:
        stat, p = stats.wilcoxon(data, alternative='greater')
        return float(stat), float(p)
    except ValueError:
        return np.nan, 1.0


# ---------------------------------------------------------------------------
# Per-version tertile analysis
# ---------------------------------------------------------------------------

def analyze_version_tertiles(version: str, cp_dirs: list[Path]) -> dict | None:
    """Analyze temporal improvement for one version using tertile blocks."""
    user_data = {}
    excluded = []  # track why users were excluded

    for uid in PILOT_USERS:
        f = find_checkpoint(version, uid, cp_dirs)
        if f is None:
            excluded.append((uid, "no checkpoint"))
            continue
        data = json.loads(f.read_text())
        preds = data.get("predictions", [])
        gts = data.get("ground_truths", [])
        if len(preds) < 15:
            excluded.append((uid, f"too few entries ({len(preds)})"))
            continue

        # Count non-empty predictions
        n_empty = sum(1 for p in preds if not p)
        empty_pct = n_empty / len(preds) * 100

        blocks = split_blocks(preds, gts, 3)
        block_bas = [compute_block_ba(bp, bg, KEY_BINARY) for bp, bg in blocks]

        failed_blocks = [i for i, b in enumerate(block_bas) if b is None]
        if failed_blocks:
            block_names = [f"T{i+1}" for i in failed_blocks]
            reason = f"block(s) {','.join(block_names)} have insufficient valid predictions"
            if empty_pct > 10:
                reason += f" ({empty_pct:.0f}% empty preds)"
            excluded.append((uid, reason))
            continue

        user_data[uid] = {
            "T1": block_bas[0],
            "T2": block_bas[1],
            "T3": block_bas[2],
            "n_entries": len(preds),
        }

    if excluded:
        meta = VERSION_META[version]
        for uid, reason in excluded:
            print(f"    {meta['label']} user {uid} excluded: {reason}")

    if len(user_data) < 4:
        return None

    uids = sorted(user_data.keys())
    t1s = np.array([user_data[u]["T1"] for u in uids])
    t2s = np.array([user_data[u]["T2"] for u in uids])
    t3s = np.array([user_data[u]["T3"] for u in uids])
    deltas = t3s - t1s

    stat, p_val = wilcoxon_greater(deltas)
    ci_lo, ci_hi = bootstrap_ci(deltas)
    d = cohens_d_one_sample(deltas)

    return {
        "n_users": len(user_data),
        "mean_T1": float(np.mean(t1s)),
        "mean_T2": float(np.mean(t2s)),
        "mean_T3": float(np.mean(t3s)),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wilcoxon_stat": stat,
        "p_value": p_val,
        "cohens_d": d,
        "per_user": {uid: user_data[uid] for uid in uids},
        "deltas": deltas.tolist(),
    }


# ---------------------------------------------------------------------------
# 5-block Spearman trend
# ---------------------------------------------------------------------------

def analyze_version_trend(version: str, cp_dirs: list[Path]) -> dict | None:
    """5-block Spearman trend analysis for one version."""
    user_rhos = []
    user_block_bas = {}

    for uid in PILOT_USERS:
        f = find_checkpoint(version, uid, cp_dirs)
        if f is None:
            continue
        data = json.loads(f.read_text())
        preds = data.get("predictions", [])
        gts = data.get("ground_truths", [])
        if len(preds) < 25:  # need at least 5 per block
            continue

        blocks = split_blocks(preds, gts, 5)
        block_bas = [compute_block_ba(bp, bg, KEY_BINARY) for bp, bg in blocks]
        valid = [(i, b) for i, b in enumerate(block_bas) if b is not None]

        if len(valid) >= 3:
            idxs, bas = zip(*valid)
            rho, _ = stats.spearmanr(idxs, bas)
            user_rhos.append(rho)
            user_block_bas[uid] = block_bas

    if len(user_rhos) < 4:
        return None

    rhos = np.array(user_rhos)
    stat, p_val = wilcoxon_greater(rhos)

    # Aggregate block means across users
    block_means = []
    for bi in range(5):
        vals = [user_block_bas[u][bi] for u in user_block_bas if user_block_bas[u][bi] is not None]
        block_means.append(float(np.mean(vals)) if vals else None)

    return {
        "n_users": len(user_rhos),
        "mean_rho": float(np.mean(rhos)),
        "median_rho": float(np.median(rhos)),
        "rho_p": float(p_val),
        "block_means": block_means,
    }


# ---------------------------------------------------------------------------
# Per-target breakdown
# ---------------------------------------------------------------------------

def analyze_per_target(cp_dirs: list[Path]) -> dict:
    """For each target x version, compute mean delta(T3-T1)."""
    results = {}

    for target in KEY_BINARY:
        target_results = {}
        for version in VERSIONS:
            deltas = []
            for uid in PILOT_USERS:
                f = find_checkpoint(version, uid, cp_dirs)
                if f is None:
                    continue
                data = json.loads(f.read_text())
                preds = data.get("predictions", [])
                gts = data.get("ground_truths", [])
                if len(preds) < 15:
                    continue

                blocks = split_blocks(preds, gts, 3)
                ba_t1 = compute_single_target_ba(blocks[0][0], blocks[0][1], target)
                ba_t3 = compute_single_target_ba(blocks[2][0], blocks[2][1], target)

                if ba_t1 is not None and ba_t3 is not None:
                    deltas.append(ba_t3 - ba_t1)

            if deltas:
                _, p = wilcoxon_greater(deltas)
                target_results[version] = {
                    "mean_delta": float(np.mean(deltas)),
                    "n": len(deltas),
                    "p_value": float(p),
                }
        results[target] = target_results

    return results


# ---------------------------------------------------------------------------
# Cross-version Friedman test
# ---------------------------------------------------------------------------

def friedman_test(tertile_results: dict[str, dict]) -> dict | None:
    """Friedman test: do learning effects differ across versions?

    Uses delta(T3-T1) per user, requiring complete cases across available versions.
    Versions without tertile data are excluded (instead of aborting entirely).
    """
    # Only include versions that have tertile results
    available_versions = [v for v in VERSIONS if tertile_results.get(v) is not None]
    if len(available_versions) < 3:
        return None

    # Find users present in ALL available versions
    all_version_users = None
    for v in available_versions:
        v_users = set(tertile_results[v]["per_user"].keys())
        all_version_users = v_users if all_version_users is None else all_version_users & v_users

    if len(all_version_users) < 4:
        return None

    users = sorted(all_version_users)
    # Build matrix: rows=users, columns=versions (delta values)
    matrix = []
    for v in available_versions:
        per_user = tertile_results[v]["per_user"]
        deltas = [per_user[u]["T3"] - per_user[u]["T1"] for u in users]
        matrix.append(deltas)

    matrix = np.array(matrix).T  # shape: (n_users, n_versions)

    try:
        stat, p = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
    except ValueError:
        return None

    # Rank versions by mean delta
    mean_deltas = {v: float(np.mean(matrix[:, i])) for i, v in enumerate(available_versions)}
    ranking = sorted(mean_deltas, key=mean_deltas.get, reverse=True)

    return {
        "n_users": len(users),
        "n_versions": len(available_versions),
        "versions_included": [VERSION_META[v]["label"] for v in available_versions],
        "versions_excluded": [VERSION_META[v]["label"] for v in VERSIONS if v not in available_versions],
        "chi2": float(stat),
        "p_value": float(p),
        "df": len(available_versions) - 1,
        "ranking": [VERSION_META[v]["label"] for v in ranking],
        "ranking_deltas": {VERSION_META[v]["label"]: mean_deltas[v] for v in ranking},
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(
    tertile_results: dict[str, dict],
    trend_results: dict[str, dict],
    target_results: dict,
    friedman: dict | None,
):
    W = 110
    print()
    print("=" * W)
    print("TEMPORAL LEARNING ANALYSIS")
    print("Do agents improve prediction accuracy over time?")
    print("=" * W)
    print()

    n_users = max(r["n_users"] for r in tertile_results.values() if r)
    print(f"  Participants: {n_users} users  |  Targets: {len(KEY_BINARY)} binary states")
    print(f"  AR baseline:  BA = {AR_BASELINE_BA:.3f} (autocorrelation ceiling)")
    print(f"  Correction:   Holm-Bonferroni across {len(VERSIONS)} versions")
    print()

    # ── Tertile analysis ──────────────────────────────────────────────

    print("─" * W)
    print("TERTILE BLOCK ANALYSIS")
    print("Entries split into T1 (early 1/3), T2 (mid 1/3), T3 (late 1/3)")
    print("H1: BA(T3) > BA(T1) — one-sided Wilcoxon signed-rank test")
    print("─" * W)
    print()

    header = (
        f"{'Version':<8} {'Strategy':<12} {'Data':<11} {'N':>3}"
        f"  {'T1':>6} {'T2':>6} {'T3':>6}"
        f"  {'Δ(T3-T1)':>9}  {'95% CI':>17}"
        f"  {'p':>7} {'p(Holm)':>8} {'d':>6}  {'Verdict'}"
    )
    print(header)
    print("─" * len(header))

    # Collect p-values for Holm correction
    raw_ps = []
    version_order = []
    for v in VERSIONS:
        r = tertile_results.get(v)
        if r:
            raw_ps.append(r["p_value"])
            version_order.append(v)

    corrected_ps = holm_correct(raw_ps)
    p_holm_map = dict(zip(version_order, corrected_ps))

    for v in VERSIONS:
        r = tertile_results.get(v)
        meta = VERSION_META[v]
        if r is None:
            print(f"{meta['label']:<8} {meta['strategy']:<12} {meta['data']:<11}  —  insufficient data")
            continue

        p_holm = p_holm_map.get(v, 1.0)
        sig_raw = "*" if r["p_value"] < ALPHA else ""
        sig_holm = "*" if p_holm < ALPHA else ""

        if p_holm < ALPHA:
            verdict = f"Sig. improvement{sig_holm}"
        elif r["p_value"] < ALPHA:
            verdict = f"Sig. (raw only){sig_raw}"
        elif r["mean_delta"] > 0 and r["p_value"] < 0.1:
            verdict = "Trend (p<0.1)"
        elif r["mean_delta"] > 0:
            verdict = "Weak positive"
        else:
            verdict = "No improvement"

        ci_str = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(
            f"{meta['label']:<8} {meta['strategy']:<12} {meta['data']:<11} {r['n_users']:>3}"
            f"  {r['mean_T1']:>6.3f} {r['mean_T2']:>6.3f} {r['mean_T3']:>6.3f}"
            f"  {r['mean_delta']:>+9.4f}  {ci_str:>17}"
            f"  {r['p_value']:>7.4f} {p_holm:>8.4f} {r['cohens_d']:>6.2f}  {verdict}"
        )

    print()

    # ── 5-block Spearman trend ────────────────────────────────────────

    print("─" * W)
    print("5-BLOCK SPEARMAN TREND")
    print("Per-user Spearman rho(block_index, block_BA), then Wilcoxon test rho > 0")
    print("─" * W)
    print()

    header2 = f"{'Version':<8} {'N':>3}  {'Mean ρ':>7} {'Median ρ':>9} {'p(ρ>0)':>7}  {'Block BAs (B1→B5)'}"
    print(header2)
    print("─" * len(header2))

    for v in VERSIONS:
        r = trend_results.get(v)
        meta = VERSION_META[v]
        if r is None:
            print(f"{meta['label']:<8}  —  insufficient data")
            continue

        bm = r["block_means"]
        bm_str = "  ".join(f"{b:.3f}" if b else "  —  " for b in bm)
        print(
            f"{meta['label']:<8} {r['n_users']:>3}"
            f"  {r['mean_rho']:>+7.3f} {r['median_rho']:>+9.3f} {r['rho_p']:>7.4f}"
            f"  {bm_str}"
        )

    print()

    # ── Friedman test ─────────────────────────────────────────────────

    print("─" * W)
    print("CROSS-VERSION COMPARISON (Friedman Test)")
    print("─" * W)
    print()

    if friedman:
        print(f"  Versions included: {', '.join(friedman['versions_included'])} ({friedman['n_versions']})")
        if friedman.get("versions_excluded"):
            print(f"  Versions excluded: {', '.join(friedman['versions_excluded'])} (insufficient tertile data)")
        print(f"  Complete cases: {friedman['n_users']} users")
        print(f"  Friedman χ²({friedman['df']}) = {friedman['chi2']:.3f}, p = {friedman['p_value']:.4f}")
        rank_str = " > ".join(
            f"{lab}({friedman['ranking_deltas'][lab]:+.4f})"
            for lab in friedman['ranking']
        )
        print(f"  Ranking by Δ: {rank_str}")
        if friedman['p_value'] < ALPHA:
            print("  → Learning effects significantly differ across versions.")
        else:
            print("  → No significant difference in learning effects across versions.")
    else:
        print("  Could not compute (insufficient complete cases).")

    print()

    # ── Per-target breakdown ──────────────────────────────────────────

    print("─" * W)
    print("PER-TARGET LEARNING EFFECTS (Mean Δ(T3-T1) per version)")
    print("─" * W)
    print()

    # Header
    v_labels = [VERSION_META[v]["label"] for v in VERSIONS]
    header3 = f"{'Target':<14}  " + "  ".join(f"{l:>8}" for l in v_labels)
    print(header3)
    print("─" * len(header3))

    for target in KEY_BINARY:
        short = TARGET_SHORT.get(target, target[:12])
        tr = target_results.get(target, {})
        cells = []
        for v in VERSIONS:
            vr = tr.get(v)
            if vr:
                sig = "*" if vr["p_value"] < ALPHA else ""
                cells.append(f"{vr['mean_delta']:>+7.3f}{sig}")
            else:
                cells.append(f"{'—':>8}")
        print(f"{short:<14}  " + "  ".join(cells))

    print()

    # ── Per-user detail ───────────────────────────────────────────────

    print("─" * W)
    print("PER-USER Δ(T3-T1) BY VERSION")
    print("─" * W)
    print()

    header4 = f"{'User':>6}  " + "  ".join(f"{VERSION_META[v]['label']:>8}" for v in VERSIONS) + f"  {'Mean':>8}"
    print(header4)
    print("─" * len(header4))

    for uid in PILOT_USERS:
        cells = []
        vals = []
        for v in VERSIONS:
            r = tertile_results.get(v)
            if r and uid in r["per_user"]:
                d = r["per_user"][uid]["T3"] - r["per_user"][uid]["T1"]
                cells.append(f"{d:>+8.3f}")
                vals.append(d)
            else:
                cells.append(f"{'—':>8}")
        mean_str = f"{np.mean(vals):>+8.3f}" if vals else f"{'—':>8}"
        print(f"{uid:>6}  " + "  ".join(cells) + f"  {mean_str}")

    print()

    # ── Conclusions ───────────────────────────────────────────────────

    print("─" * W)
    print("CONCLUSIONS")
    print("─" * W)
    print()

    sig_raw_versions = []
    sig_holm_versions = []
    for v in VERSIONS:
        r = tertile_results.get(v)
        if r is None:
            continue
        if r["p_value"] < ALPHA:
            sig_raw_versions.append(VERSION_META[v]["label"])
        if p_holm_map.get(v, 1.0) < ALPHA:
            sig_holm_versions.append(VERSION_META[v]["label"])

    # Group by strategy
    agentic = [v for v in VERSIONS if VERSION_META[v]["strategy"] == "Agentic"]
    structured = [v for v in VERSIONS if VERSION_META[v]["strategy"] == "Structured"]

    agentic_avail = [v for v in agentic if tertile_results.get(v)]
    structured_avail = [v for v in structured if tertile_results.get(v)]
    agentic_deltas = [tertile_results[v]["mean_delta"] for v in agentic_avail]
    structured_deltas = [tertile_results[v]["mean_delta"] for v in structured_avail]

    mean_ag = np.mean(agentic_deltas) if agentic_deltas else 0
    mean_st = np.mean(structured_deltas) if structured_deltas else 0

    print(f"  1. Versions with significant improvement (raw p<0.05): {', '.join(sig_raw_versions) or 'None'}")
    print(f"  2. Versions surviving Holm correction (p<0.05):        {', '.join(sig_holm_versions) or 'None'}")
    print()
    print(f"  3. Strategy comparison:")
    print(f"     Agentic  mean Δ = {mean_ag:+.4f}  (versions: {', '.join(VERSION_META[v]['label'] for v in agentic)})")
    print(f"     Struct.  mean Δ = {mean_st:+.4f}  (versions: {', '.join(VERSION_META[v]['label'] for v in structured)})")
    if mean_ag > mean_st:
        print(f"     → Agentic versions show stronger learning effects (+{mean_ag - mean_st:.4f} mean Δ difference)")
    else:
        print(f"     → Structured versions show comparable or stronger effects")
    print()

    # Best & worst
    ranked = sorted(
        [(v, tertile_results[v]["mean_delta"]) for v in VERSIONS if tertile_results.get(v)],
        key=lambda x: x[1], reverse=True
    )
    best_v, best_d = ranked[0]
    worst_v, worst_d = ranked[-1]
    print(f"  4. Strongest learner: {VERSION_META[best_v]['label']} (Δ={best_d:+.4f})")
    print(f"     Weakest learner:  {VERSION_META[worst_v]['label']} (Δ={worst_d:+.4f})")
    print()

    # AR baseline comparison
    print(f"  5. AR baseline context (BA={AR_BASELINE_BA:.3f}):")
    for v in VERSIONS:
        r = tertile_results.get(v)
        if r:
            above_ar_t1 = "yes" if r["mean_T1"] > AR_BASELINE_BA else "no"
            above_ar_t3 = "yes" if r["mean_T3"] > AR_BASELINE_BA else "no"
            if above_ar_t3 == "yes" and above_ar_t1 == "no":
                print(f"     {VERSION_META[v]['label']}: T1={r['mean_T1']:.3f} (below AR) → T3={r['mean_T3']:.3f} (above AR) ★")
            else:
                print(f"     {VERSION_META[v]['label']}: T1={r['mean_T1']:.3f} → T3={r['mean_T3']:.3f} ({'above' if above_ar_t3 == 'yes' else 'below'} AR)")

    print()
    print("  NOTE: V3 results for users 119, 164, 310, 458 use mixed sonnet+haiku checkpoints.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal learning effects")
    parser.add_argument(
        "--checkpoint-dirs", nargs="+", default=None,
        help="Checkpoint directories (searched in order, first match wins)",
    )
    parser.add_argument(
        "--output", default="outputs/pilot_v2/learning_curve_stats.json",
        help="Where to save JSON results",
    )
    args = parser.parse_args()

    cp_dirs = (
        [Path(d) for d in args.checkpoint_dirs]
        if args.checkpoint_dirs
        else [Path(d) for d in DEFAULT_CHECKPOINT_DIRS]
    )
    output_path = Path(args.output)

    print(f"Checkpoint dirs: {[str(d) for d in cp_dirs]}")
    print(f"Users: {PILOT_USERS}")

    # ── 1. Tertile analysis ───────────────────────────────────────────
    print("\nRunning tertile analysis...")
    tertile_results = {}
    for v in VERSIONS:
        r = analyze_version_tertiles(v, cp_dirs)
        tertile_results[v] = r
        if r:
            print(f"  {VERSION_META[v]['label']}: N={r['n_users']}, Δ={r['mean_delta']:+.4f}, p={r['p_value']:.4f}")

    # ── 2. 5-block Spearman trend ─────────────────────────────────────
    print("\nRunning 5-block trend analysis...")
    trend_results = {}
    for v in VERSIONS:
        r = analyze_version_trend(v, cp_dirs)
        trend_results[v] = r
        if r:
            print(f"  {VERSION_META[v]['label']}: ρ={r['mean_rho']:+.3f}, p={r['rho_p']:.4f}")

    # ── 3. Per-target breakdown ───────────────────────────────────────
    print("\nRunning per-target analysis...")
    target_results = analyze_per_target(cp_dirs)

    # ── 4. Friedman test ──────────────────────────────────────────────
    print("\nRunning Friedman test...")
    friedman = friedman_test(tertile_results)

    # ── Output ────────────────────────────────────────────────────────
    print_results(tertile_results, trend_results, target_results, friedman)

    # Save JSON
    save_data = {
        "config": {
            "users": PILOT_USERS,
            "versions": VERSIONS,
            "binary_targets": KEY_BINARY,
            "ar_baseline_ba": AR_BASELINE_BA,
            "alpha": ALPHA,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "tertile": {
            v: {k: r[k] for k in ["n_users", "mean_T1", "mean_T2", "mean_T3",
                                    "mean_delta", "std_delta", "ci_lo", "ci_hi",
                                    "p_value", "cohens_d"]
                }
            for v, r in tertile_results.items() if r
        },
        "trend": {
            v: r for v, r in trend_results.items() if r
        },
        "per_target": target_results,
        "friedman": friedman,
    }

    # Add Holm-corrected p-values
    raw_ps = [tertile_results[v]["p_value"] for v in VERSIONS if tertile_results.get(v)]
    version_order = [v for v in VERSIONS if tertile_results.get(v)]
    corrected = holm_correct(raw_ps)
    for v, p_holm in zip(version_order, corrected):
        save_data["tertile"][v]["p_holm"] = p_holm

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(save_data, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
