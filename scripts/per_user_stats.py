"""Per-user balanced accuracy analysis with statistical significance tests.

Loads all checkpoints for 7 agent versions across 50 users, computes
per-user balanced accuracy (BA) for focus targets and mean BA across
all binary targets, then runs paired Wilcoxon signed-rank tests and
bootstrap confidence intervals.

Usage:
    cd /Users/zwang/projects/proactive-affective-agent
    PYTHONPATH=. python3 scripts/per_user_stats.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

from src.utils.mappings import BINARY_STATE_TARGETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = Path("outputs/pilot_v2/checkpoints")
OUTPUT_PATH = Path("outputs/pilot_v2/statistical_tests.json")

VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
VERSION_LABELS = {
    "callm": "CALLM",
    "v1": "Struct-Sense",
    "v2": "Auto-Sense",
    "v3": "Struct-Multi",
    "v4": "Auto-Multi",
    "v5": "Auto-Sense+",
    "v6": "Auto-Multi+",
}

# 4 focus targets + mean across all binary
FOCUS_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
]

ALL_BINARY_TARGETS = BINARY_STATE_TARGETS + ["INT_availability"]

# Paired comparisons: (version_a, version_b, label)
COMPARISONS = [
    ("v4", "v3", "Auto-Multi vs Struct-Multi"),
    ("v6", "v3", "Auto-Multi+ vs Struct-Multi"),
    ("v2", "v1", "Auto-Sense vs Struct-Sense"),
    ("v4", "callm", "Auto-Multi vs CALLM"),
    ("v6", "callm", "Auto-Multi+ vs CALLM"),
]

N_BOOTSTRAP = 10000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bool(v) -> bool | None:
    """Convert a value to bool, matching evaluate_pilot.py logic."""
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


def find_all_users() -> list[int]:
    """Discover all user IDs from checkpoint filenames."""
    users = set()
    for f in os.listdir(CHECKPOINT_DIR):
        m = re.match(r"\w+_user(\d+)_checkpoint\.json", f)
        if m:
            users.add(int(m.group(1)))
    return sorted(users)


def load_checkpoint(version: str, uid: int) -> tuple[list[dict], list[dict]] | None:
    """Load predictions and ground truths for a single version+user."""
    name = f"{version}_user{uid}_checkpoint.json"
    path = CHECKPOINT_DIR / name
    if not path.exists():
        # Try uppercase for CALLM
        if version.lower() == "callm":
            path = CHECKPOINT_DIR / f"CALLM_user{uid}_checkpoint.json"
        if not path.exists():
            return None
    data = json.loads(path.read_text())
    preds = data.get("predictions", [])
    gts = data.get("ground_truths", [])
    if not preds or not gts:
        return None
    return preds, gts


def compute_user_ba(preds: list[dict], gts: list[dict], target: str) -> float | None:
    """Compute balanced accuracy for a single user on a single target."""
    y, yhat = [], []
    for p, g in zip(preds, gts):
        gv = _to_bool(g.get(target))
        pv = _to_bool(p.get(target))
        if gv is not None and pv is not None:
            y.append(int(gv))
            yhat.append(int(pv))
    if len(y) < 5:
        return None
    # Need at least 2 classes in ground truth for BA to be meaningful
    if len(set(y)) < 2:
        return None
    return float(balanced_accuracy_score(y, yhat))


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 alpha: float = 0.05, seed: int = RANDOM_SEED) -> tuple[float, float, float]:
    """Compute bootstrap mean and (1-alpha) CI."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def main():
    users = find_all_users()
    print(f"Found {len(users)} users")
    print(f"Versions: {VERSIONS}")
    print(f"Focus targets: {FOCUS_TARGETS}")
    print(f"All binary targets ({len(ALL_BINARY_TARGETS)}): {ALL_BINARY_TARGETS}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Compute per-user BA for each version x target
    # Structure: per_user_ba[version][target] = {uid: ba_value}
    # -----------------------------------------------------------------------
    per_user_ba: dict[str, dict[str, dict[int, float]]] = {}

    for version in VERSIONS:
        per_user_ba[version] = {}
        for target in ALL_BINARY_TARGETS:
            per_user_ba[version][target] = {}

        for uid in users:
            result = load_checkpoint(version, uid)
            if result is None:
                continue
            preds, gts = result

            for target in ALL_BINARY_TARGETS:
                ba = compute_user_ba(preds, gts, target)
                if ba is not None:
                    per_user_ba[version][target][uid] = ba

    # -----------------------------------------------------------------------
    # Step 2: Compute per-user mean BA across all binary targets
    # per_user_mean_ba[version] = {uid: mean_ba}
    # -----------------------------------------------------------------------
    per_user_mean_ba: dict[str, dict[int, float]] = {}

    for version in VERSIONS:
        per_user_mean_ba[version] = {}
        for uid in users:
            user_bas = []
            for target in ALL_BINARY_TARGETS:
                ba = per_user_ba[version][target].get(uid)
                if ba is not None:
                    user_bas.append(ba)
            if user_bas:
                per_user_mean_ba[version][uid] = float(np.mean(user_bas))

    # -----------------------------------------------------------------------
    # Step 3: Print per-version summary
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("PER-USER BALANCED ACCURACY SUMMARY")
    print("=" * 90)

    # Focus targets + mean BA
    targets_to_report = FOCUS_TARGETS + ["MEAN_BA"]

    header = f"{'Version':<16}"
    for t in targets_to_report:
        short = t.replace("Individual_level_", "").replace("_State", "")
        header += f"{short:>14}"
    header += f"{'N_users':>10}"
    print(header)
    print("-" * 90)

    version_summaries = {}
    for version in VERSIONS:
        row = f"{VERSION_LABELS[version]:<16}"
        summary = {}
        for target in FOCUS_TARGETS:
            vals = list(per_user_ba[version][target].values())
            if vals:
                mean_val = np.mean(vals)
                summary[target] = {
                    "mean": float(mean_val),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }
                row += f"{mean_val:>14.3f}"
            else:
                row += f"{'--':>14}"

        # Mean BA
        mean_ba_vals = list(per_user_mean_ba[version].values())
        if mean_ba_vals:
            mean_mean_ba = np.mean(mean_ba_vals)
            summary["MEAN_BA"] = {
                "mean": float(mean_mean_ba),
                "std": float(np.std(mean_ba_vals)),
                "median": float(np.median(mean_ba_vals)),
                "n": len(mean_ba_vals),
            }
            row += f"{mean_mean_ba:>14.3f}"
            row += f"{len(mean_ba_vals):>10}"
        else:
            row += f"{'--':>14}{'--':>10}"

        version_summaries[version] = summary
        print(row)

    # -----------------------------------------------------------------------
    # Step 4: Bootstrap 95% CIs
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (10,000 resamples)")
    print("=" * 90)

    ci_results = {}

    for target in targets_to_report:
        short = target.replace("Individual_level_", "").replace("_State", "")
        print(f"\n  {short}:")

        ci_results[target] = {}
        for version in VERSIONS:
            if target == "MEAN_BA":
                vals = np.array(list(per_user_mean_ba[version].values()))
            else:
                vals = np.array(list(per_user_ba[version][target].values()))

            if len(vals) < 3:
                print(f"    {VERSION_LABELS[version]:<16} insufficient data")
                continue

            mean_val, lo, hi = bootstrap_ci(vals)
            ci_results[target][version] = {
                "mean": mean_val,
                "ci_lower": lo,
                "ci_upper": hi,
                "n": len(vals),
            }
            print(f"    {VERSION_LABELS[version]:<16} {mean_val:.3f}  [{lo:.3f}, {hi:.3f}]  (n={len(vals)})")

    # -----------------------------------------------------------------------
    # Step 5: Paired Wilcoxon signed-rank tests
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("PAIRED WILCOXON SIGNED-RANK TESTS")
    print("=" * 90)

    wilcoxon_results = {}

    for va, vb, label in COMPARISONS:
        print(f"\n  {label} ({VERSION_LABELS[va]} vs {VERSION_LABELS[vb]}):")
        wilcoxon_results[label] = {}

        for target in targets_to_report:
            short = target.replace("Individual_level_", "").replace("_State", "")

            if target == "MEAN_BA":
                dict_a = per_user_mean_ba[va]
                dict_b = per_user_mean_ba[vb]
            else:
                dict_a = per_user_ba[va][target]
                dict_b = per_user_ba[vb][target]

            # Find users present in both versions
            common_users = sorted(set(dict_a.keys()) & set(dict_b.keys()))
            if len(common_users) < 5:
                print(f"    {short:<30} insufficient paired data (n={len(common_users)})")
                wilcoxon_results[label][target] = {
                    "status": "insufficient_data",
                    "n_paired": len(common_users),
                }
                continue

            vals_a = np.array([dict_a[u] for u in common_users])
            vals_b = np.array([dict_b[u] for u in common_users])
            diffs = vals_a - vals_b

            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            mean_diff = float(np.mean(diffs))

            # Check if all differences are zero (Wilcoxon can't handle this)
            if np.all(diffs == 0):
                print(f"    {short:<30} all differences are zero (n={len(common_users)})")
                wilcoxon_results[label][target] = {
                    "status": "all_zeros",
                    "n_paired": len(common_users),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": mean_diff,
                }
                continue

            try:
                stat_val, p_val = stats.wilcoxon(vals_a, vals_b, alternative='two-sided')
                # Also compute one-sided (a > b)
                _, p_greater = stats.wilcoxon(vals_a, vals_b, alternative='greater')

                # Effect size: matched-pairs rank-biserial correlation
                n_pairs = len(common_users)
                r_effect = 1 - (2 * stat_val) / (n_pairs * (n_pairs + 1) / 2)

                sig = ""
                if p_val < 0.001:
                    sig = " ***"
                elif p_val < 0.01:
                    sig = " **"
                elif p_val < 0.05:
                    sig = " *"

                print(f"    {short:<30} diff={mean_diff:+.4f}  p={p_val:.4f}{sig}  "
                      f"(p_greater={p_greater:.4f})  r={r_effect:.3f}  n={n_pairs}")

                wilcoxon_results[label][target] = {
                    "status": "ok",
                    "n_paired": n_pairs,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": mean_diff,
                    "wilcoxon_statistic": float(stat_val),
                    "p_value_two_sided": float(p_val),
                    "p_value_greater": float(p_greater),
                    "effect_size_r": float(r_effect),
                }
            except Exception as e:
                print(f"    {short:<30} error: {e}")
                wilcoxon_results[label][target] = {
                    "status": "error",
                    "error": str(e),
                }

    # -----------------------------------------------------------------------
    # Step 6: Per-user BA distributions (for visualization)
    # -----------------------------------------------------------------------
    distributions = {}
    for version in VERSIONS:
        distributions[version] = {}
        for target in targets_to_report:
            if target == "MEAN_BA":
                user_vals = per_user_mean_ba[version]
            else:
                user_vals = per_user_ba[version][target]
            # Store as {uid_str: ba_value} for JSON
            distributions[version][target] = {
                str(uid): val for uid, val in sorted(user_vals.items())
            }

    # -----------------------------------------------------------------------
    # Step 7: Per-user distribution summary stats
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("PER-USER BA DISTRIBUTION SUMMARY (MEAN_BA)")
    print("=" * 90)
    print(f"{'Version':<16} {'Mean':>8} {'Std':>8} {'Min':>8} {'Q1':>8} "
          f"{'Median':>8} {'Q3':>8} {'Max':>8} {'N':>6}")
    print("-" * 90)

    for version in VERSIONS:
        vals = np.array(list(per_user_mean_ba[version].values()))
        if len(vals) == 0:
            continue
        print(f"{VERSION_LABELS[version]:<16} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
              f"{np.min(vals):>8.3f} {np.percentile(vals, 25):>8.3f} "
              f"{np.median(vals):>8.3f} {np.percentile(vals, 75):>8.3f} "
              f"{np.max(vals):>8.3f} {len(vals):>6}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output = {
        "metadata": {
            "n_users": len(users),
            "users": users,
            "versions": VERSIONS,
            "version_labels": VERSION_LABELS,
            "focus_targets": FOCUS_TARGETS,
            "all_binary_targets": ALL_BINARY_TARGETS,
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
        },
        "version_summaries": version_summaries,
        "bootstrap_ci": ci_results,
        "wilcoxon_tests": wilcoxon_results,
        "per_user_distributions": distributions,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
