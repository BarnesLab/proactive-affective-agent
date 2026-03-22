#!/usr/bin/env python3
"""Analyze whether the 50 evaluation users are representative of the full BUCS cohort.

Compares the 50 pilot users (identified from checkpoint filenames) against the
remaining users on key EMA variables, using Mann-Whitney U tests for continuous
measures and chi-squared tests for categorical ones.

Usage:
    PYTHONPATH=. python3 scripts/representativeness_analysis.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "pilot_v2" / "checkpoints"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
BUCS_DATA_DIR = PROJECT_ROOT / "data" / "bucs-data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pilot_v2" / "representativeness.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pilot_users_from_checkpoints() -> set[int]:
    """Extract unique user IDs from checkpoint filenames in pilot_v2/checkpoints/."""
    user_ids: set[int] = set()
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {CHECKPOINT_DIR}")
    for fname in os.listdir(CHECKPOINT_DIR):
        m = re.search(r"user(\d+)_checkpoint\.json", fname)
        if m:
            user_ids.add(int(m.group(1)))
    return user_ids


def load_all_ema() -> pd.DataFrame:
    """Load all EMA data by combining 5 test splits (same as DataLoader.load_all_ema)."""
    dfs = []
    for group in range(1, 6):
        path = SPLITS_DIR / f"group_{group}_test.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError("No test split files found")
    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp_local"] = pd.to_datetime(combined["timestamp_local"])
    combined["date_local"] = pd.to_datetime(combined["date_local"]).dt.date
    combined = combined.drop_duplicates(subset=["Study_ID", "timestamp_local"])
    return combined


def derive_platform(bucs_dir: Path) -> dict[int, str]:
    """Derive platform (iOS/Android) from sensing directory filenames.

    AndroidSleep files -> Android users, MotionActivity files -> iOS users.
    """
    platform: dict[int, str] = {}

    android_dir = bucs_dir / "AndroidSleep"
    if android_dir.exists():
        for f in os.listdir(android_dir):
            if f.endswith(".csv"):
                parts = f.split("_")
                if len(parts) >= 3:
                    try:
                        uid = int(parts[2])
                        platform[uid] = "Android"
                    except ValueError:
                        pass

    ios_dir = bucs_dir / "MotionActivity"
    if ios_dir.exists():
        for f in os.listdir(ios_dir):
            if f.endswith(".csv"):
                parts = f.split("_")
                if len(parts) >= 3:
                    try:
                        uid = int(parts[2])
                        if uid not in platform:  # don't overwrite Android
                            platform[uid] = "iOS"
                    except ValueError:
                        pass

    return platform


def mannwhitney_test(group1: pd.Series, group2: pd.Series, label: str) -> dict:
    """Run Mann-Whitney U test and return summary statistics."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return {"label": label, "error": "insufficient data"}

    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
    n1, n2 = len(g1), len(g2)
    r = 1 - 2 * stat / (n1 * n2)

    return {
        "label": label,
        "pilot_mean": float(g1.mean()),
        "pilot_std": float(g1.std()),
        "pilot_median": float(g1.median()),
        "pilot_n": int(n1),
        "rest_mean": float(g2.mean()),
        "rest_std": float(g2.std()),
        "rest_median": float(g2.median()),
        "rest_n": int(n2),
        "U_statistic": float(stat),
        "p_value": float(p),
        "effect_size_r": float(r),
        "significant_005": p < 0.05,
    }


def chi2_test(pilot_counts: dict, rest_counts: dict, label: str) -> dict:
    """Run chi-squared test for categorical distribution."""
    categories = sorted(set(pilot_counts.keys()) | set(rest_counts.keys()))
    observed_pilot = [pilot_counts.get(c, 0) for c in categories]
    observed_rest = [rest_counts.get(c, 0) for c in categories]

    contingency = np.array([observed_pilot, observed_rest])
    # Remove columns with all zeros
    nonzero = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero]
    categories_used = [c for c, nz in zip(categories, nonzero) if nz]

    if contingency.shape[1] < 2:
        return {"label": label, "error": "fewer than 2 non-empty categories"}

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "label": label,
        "categories": categories_used,
        "pilot_counts": {c: int(v) for c, v in zip(categories_used, contingency[0])},
        "rest_counts": {c: int(v) for c, v in zip(categories_used, contingency[1])},
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "significant_005": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("REPRESENTATIVENESS ANALYSIS: 50 Pilot Users vs Rest of Cohort")
    print("=" * 70)

    # 1. Identify pilot users
    pilot_ids = get_pilot_users_from_checkpoints()
    print(f"\nPilot users (from checkpoints): {len(pilot_ids)}")
    print(f"  IDs: {sorted(pilot_ids)}")

    # 2. Load all EMA data
    ema = load_all_ema()
    all_user_ids = set(ema["Study_ID"].unique())
    rest_ids = all_user_ids - pilot_ids
    print(f"\nTotal users in EMA data: {len(all_user_ids)}")
    print(f"Pilot users found in EMA: {len(pilot_ids & all_user_ids)}")
    print(f"Remaining users: {len(rest_ids)}")

    pilot_ema = ema[ema["Study_ID"].isin(pilot_ids)]
    rest_ema = ema[ema["Study_ID"].isin(rest_ids)]

    results: dict = {
        "n_pilot": len(pilot_ids),
        "n_rest": len(rest_ids),
        "n_total": len(all_user_ids),
        "pilot_ids": sorted(pilot_ids),
        "tests": {},
    }

    # -----------------------------------------------------------------------
    # 3. Per-user summary statistics
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Computing per-user statistics...")

    # 3a. EMA entries per user
    user_ema_count = ema.groupby("Study_ID").size().reset_index(name="ema_count")
    pilot_counts = user_ema_count[user_ema_count["Study_ID"].isin(pilot_ids)]["ema_count"]
    rest_counts = user_ema_count[user_ema_count["Study_ID"].isin(rest_ids)]["ema_count"]

    t = mannwhitney_test(pilot_counts, rest_counts, "EMA entries per user")
    results["tests"]["ema_count"] = t
    print(f"\n  EMA entries per user:")
    print(f"    Pilot: mean={t['pilot_mean']:.1f} (sd={t['pilot_std']:.1f}), median={t['pilot_median']:.0f}")
    print(f"    Rest:  mean={t['rest_mean']:.1f} (sd={t['rest_std']:.1f}), median={t['rest_median']:.0f}")
    print(f"    Mann-Whitney U={t['U_statistic']:.0f}, p={t['p_value']:.4f}, r={t['effect_size_r']:.3f}")
    sig = "*" if t["significant_005"] else ""
    print(f"    {'** SIGNIFICANT at p<0.05 **' if t['significant_005'] else 'Not significant at p<0.05'}")

    # 3b. Diary completion rate (% with emotion_driver text)
    ema["has_diary"] = ema["emotion_driver"].notna() & (ema["emotion_driver"].astype(str).str.strip() != "")
    user_diary_rate = ema.groupby("Study_ID").agg(
        diary_rate=("has_diary", "mean"),
    ).reset_index()
    pilot_diary = user_diary_rate[user_diary_rate["Study_ID"].isin(pilot_ids)]["diary_rate"]
    rest_diary = user_diary_rate[user_diary_rate["Study_ID"].isin(rest_ids)]["diary_rate"]

    t = mannwhitney_test(pilot_diary, rest_diary, "Diary completion rate")
    results["tests"]["diary_completion_rate"] = t
    print(f"\n  Diary completion rate (fraction of EMAs with emotion_driver):")
    print(f"    Pilot: mean={t['pilot_mean']:.3f}, median={t['pilot_median']:.3f}")
    print(f"    Rest:  mean={t['rest_mean']:.3f}, median={t['rest_median']:.3f}")
    print(f"    p={t['p_value']:.4f}, r={t['effect_size_r']:.3f}")
    print(f"    {'** SIGNIFICANT **' if t['significant_005'] else 'Not significant'}")

    # -----------------------------------------------------------------------
    # 4. Base rates for binary focus targets
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Base rates for binary state targets...")

    binary_targets = [
        "Individual_level_PA_State",
        "Individual_level_NA_State",
        "Individual_level_ER_desire_State",
        "INT_availability",
    ]

    for target in binary_targets:
        if target not in ema.columns:
            print(f"  {target}: column not found, skipping")
            continue
        # Convert string yes/no to numeric if needed
        col = ema[target].copy()
        if col.dtype == object:
            col = col.map({"yes": 1, "no": 0, "Yes": 1, "No": 0, True: 1, False: 0})
        col = pd.to_numeric(col, errors="coerce")
        ema[f"_binary_{target}"] = col
        user_rates = ema.groupby("Study_ID")[f"_binary_{target}"].mean().reset_index()
        user_rates.columns = ["Study_ID", "rate"]
        pilot_rates = user_rates[user_rates["Study_ID"].isin(pilot_ids)]["rate"]
        rest_rates = user_rates[user_rates["Study_ID"].isin(rest_ids)]["rate"]

        t = mannwhitney_test(pilot_rates, rest_rates, f"Base rate: {target}")
        results["tests"][f"base_rate_{target}"] = t
        print(f"\n  {target}:")
        print(f"    Pilot: mean={t['pilot_mean']:.3f}, median={t['pilot_median']:.3f}")
        print(f"    Rest:  mean={t['rest_mean']:.3f}, median={t['rest_median']:.3f}")
        print(f"    p={t['p_value']:.4f}, r={t['effect_size_r']:.3f}")
        print(f"    {'** SIGNIFICANT **' if t['significant_005'] else 'Not significant'}")

    # -----------------------------------------------------------------------
    # 5. Mean continuous values per user
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Continuous target distributions...")

    continuous_targets = ["PANAS_Pos", "PANAS_Neg", "ER_desire"]

    for target in continuous_targets:
        if target not in ema.columns:
            print(f"  {target}: column not found, skipping")
            continue
        user_means = ema.groupby("Study_ID")[target].mean().reset_index()
        user_means.columns = ["Study_ID", "mean_val"]
        pilot_vals = user_means[user_means["Study_ID"].isin(pilot_ids)]["mean_val"]
        rest_vals = user_means[user_means["Study_ID"].isin(rest_ids)]["mean_val"]

        t = mannwhitney_test(pilot_vals, rest_vals, f"Mean {target}")
        results["tests"][f"continuous_{target}"] = t
        print(f"\n  {target}:")
        print(f"    Pilot: mean={t['pilot_mean']:.2f} (sd={t['pilot_std']:.2f})")
        print(f"    Rest:  mean={t['rest_mean']:.2f} (sd={t['rest_std']:.2f})")
        print(f"    p={t['p_value']:.4f}, r={t['effect_size_r']:.3f}")
        print(f"    {'** SIGNIFICANT **' if t['significant_005'] else 'Not significant'}")

    # -----------------------------------------------------------------------
    # 6. Platform distribution (iOS vs Android)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Platform distribution (iOS vs Android from sensing data)...")

    platform_map = derive_platform(BUCS_DATA_DIR)
    ema_users_with_platform = {uid: platform_map[uid] for uid in all_user_ids if uid in platform_map}

    if ema_users_with_platform:
        pilot_platform = {uid: p for uid, p in ema_users_with_platform.items() if uid in pilot_ids}
        rest_platform = {uid: p for uid, p in ema_users_with_platform.items() if uid in rest_ids}

        pilot_plat_counts: dict[str, int] = {}
        for p in pilot_platform.values():
            pilot_plat_counts[p] = pilot_plat_counts.get(p, 0) + 1

        rest_plat_counts: dict[str, int] = {}
        for p in rest_platform.values():
            rest_plat_counts[p] = rest_plat_counts.get(p, 0) + 1

        print(f"  Users with platform info: {len(ema_users_with_platform)}/{len(all_user_ids)}")
        print(f"  Pilot platform: {pilot_plat_counts}")
        print(f"  Rest platform:  {rest_plat_counts}")

        t = chi2_test(pilot_plat_counts, rest_plat_counts, "Platform distribution")
        results["tests"]["platform"] = t
        if "error" not in t:
            print(f"  Chi-squared={t['chi2']:.3f}, p={t['p_value']:.4f}")
            print(f"  {'** SIGNIFICANT **' if t['significant_005'] else 'Not significant'}")
        else:
            print(f"  {t['error']}")

        # Also compute percentages
        total_pilot_plat = sum(pilot_plat_counts.values())
        total_rest_plat = sum(rest_plat_counts.values())
        if total_pilot_plat > 0 and total_rest_plat > 0:
            pilot_ios_pct = pilot_plat_counts.get("iOS", 0) / total_pilot_plat * 100
            rest_ios_pct = rest_plat_counts.get("iOS", 0) / total_rest_plat * 100
            print(f"  iOS%: pilot={pilot_ios_pct:.1f}%, rest={rest_ios_pct:.1f}%")
    else:
        print("  No platform info available")
        results["tests"]["platform"] = {"label": "Platform distribution", "error": "no platform data"}

    # -----------------------------------------------------------------------
    # 7. Cross-validation group distribution
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Cross-validation group distribution...")

    if "group" in ema.columns:
        user_group = ema.groupby("Study_ID")["group"].first().reset_index()

        pilot_group_counts: dict[str, int] = {}
        rest_group_counts: dict[str, int] = {}
        for _, row in user_group.iterrows():
            g = str(int(row["group"]))
            if row["Study_ID"] in pilot_ids:
                pilot_group_counts[g] = pilot_group_counts.get(g, 0) + 1
            else:
                rest_group_counts[g] = rest_group_counts.get(g, 0) + 1

        print(f"  Pilot groups: {pilot_group_counts}")
        print(f"  Rest groups:  {rest_group_counts}")

        t = chi2_test(pilot_group_counts, rest_group_counts, "CV group distribution")
        results["tests"]["cv_group"] = t
        if "error" not in t:
            print(f"  Chi-squared={t['chi2']:.3f}, p={t['p_value']:.4f}")
            print(f"  {'** SIGNIFICANT **' if t['significant_005'] else 'Not significant'}")

    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_tests = 0
    n_sig = 0
    for key, t in results["tests"].items():
        if "error" in t:
            continue
        n_tests += 1
        p_key = "p_value"
        if t.get(p_key) is not None and t[p_key] < 0.05:
            n_sig += 1

    print(f"\n  Total tests: {n_tests}")
    print(f"  Significant at p<0.05: {n_sig}")
    print(f"  Not significant: {n_tests - n_sig}")

    if n_sig == 0:
        verdict = "REPRESENTATIVE — No significant differences found between pilot and rest."
    elif n_sig <= 2:
        verdict = "MOSTLY REPRESENTATIVE — Few significant differences, likely due to selection criteria."
    else:
        verdict = "POTENTIALLY BIASED — Multiple significant differences detected."

    results["summary"] = {
        "n_tests": n_tests,
        "n_significant": n_sig,
        "verdict": verdict,
    }

    print(f"\n  Verdict: {verdict}")

    # Highlight significant tests
    if n_sig > 0:
        print("\n  Significant differences:")
        for key, t in results["tests"].items():
            if "error" in t:
                continue
            if t.get("significant_005"):
                label = t.get("label", key)
                p = t.get("p_value", float("nan"))
                r = t.get("effect_size_r", t.get("chi2", float("nan")))
                print(f"    - {label}: p={p:.4f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
