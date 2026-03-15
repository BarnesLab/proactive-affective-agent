#!/usr/bin/env python3
"""
Reviewer response analyses for the proactive affective agent pilot study.

Computes:
1. Per-user BA distributions (consistency of agentic advantage)
2. Binary state base rates (class imbalance)
3. Tool call analysis from JSONL records (v2/v4)
4. Performance stratified by diary availability (v4/v6)
5. Joint intervention opportunity accuracy (ER_desire_State AND INT_availability)
6. ML baseline comparison on same 11 users (if feasible)

Usage:
    PYTHONPATH=. python3 scripts/reviewer_analyses.py
"""

import json
import os
import re
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, f1_score

# ============================================================
# Configuration
# ============================================================
PILOT_USERS = [43, 71, 119, 164, 258, 275, 310, 338, 403, 458, 513]
VERSIONS = ["callm", "v1", "v2", "v3", "v4", "v5", "v6"]
CHECKPOINT_DIRS = ["outputs/pilot_v2/checkpoints", "outputs/pilot/checkpoints"]
JSONL_DIRS = ["outputs/pilot_v2", "outputs/pilot"]

BINARY_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_happy_State",
    "Individual_level_sad_State",
    "Individual_level_afraid_State",
    "Individual_level_miserable_State",
    "Individual_level_worried_State",
    "Individual_level_cheerful_State",
    "Individual_level_pleased_State",
    "Individual_level_grateful_State",
    "Individual_level_lonely_State",
    "Individual_level_interactions_quality_State",
    "Individual_level_pain_State",
    "Individual_level_forecasting_State",
    "Individual_level_ER_desire_State",
]

# INT_availability is also binary but uses 'yes'/'no' strings
INT_TARGET = "INT_availability"

OUTPUT_FILE = "outputs/reviewer_analyses.txt"


def load_checkpoint(version, user_id):
    """Load checkpoint data for a version/user from either pilot dir."""
    patterns = [
        f"{version}_user{user_id}_checkpoint.json",
        f"{version.upper()}_user{user_id}_checkpoint.json",
        f"{version.lower()}_user{user_id}_checkpoint.json",
    ]
    for base in CHECKPOINT_DIRS:
        for pat in patterns:
            fp = os.path.join(base, pat)
            if os.path.exists(fp):
                with open(fp) as f:
                    return json.load(f)
    # Try case-insensitive search
    for base in CHECKPOINT_DIRS:
        if not os.path.isdir(base):
            continue
        for fn in os.listdir(base):
            m = re.match(
                rf"({re.escape(version)})_user{user_id}_checkpoint\.json",
                fn,
                re.IGNORECASE,
            )
            if m:
                with open(os.path.join(base, fn)) as f:
                    return json.load(f)
    return None


def load_jsonl_records(version, user_id):
    """Load JSONL records for a version/user, deduplicating by entry_idx."""
    records = {}
    patterns = [
        f"{version}_user{user_id}_records.jsonl",
        f"{version.upper()}_user{user_id}_records.jsonl",
    ]
    for base in JSONL_DIRS:
        for pat in patterns:
            fp = os.path.join(base, pat)
            if os.path.exists(fp):
                with open(fp) as f:
                    for line in f:
                        rec = json.loads(line)
                        key = (rec.get("study_id"), rec.get("entry_idx"))
                        records[key] = rec  # last occurrence wins
    # Also try case-insensitive
    for base in JSONL_DIRS:
        if not os.path.isdir(base):
            continue
        for fn in os.listdir(base):
            m = re.match(
                rf"({re.escape(version)})_user{user_id}_records\.jsonl",
                fn,
                re.IGNORECASE,
            )
            if m:
                fp = os.path.join(base, fn)
                with open(fp) as f:
                    for line in f:
                        rec = json.loads(line)
                        key = (rec.get("study_id"), rec.get("entry_idx"))
                        records[key] = rec
    return list(records.values())


def to_bool(val):
    """Convert prediction/ground truth to bool for binary targets."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "elevated")
    if isinstance(val, (int, float)):
        return bool(val)
    return None


def to_int_availability_bool(val):
    """Convert INT_availability to bool (yes=True, no=False)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("yes", "true", "1")
    return None


def compute_ba_for_pairs(y_true, y_pred):
    """Compute balanced accuracy, returns None if degenerate."""
    if len(y_true) < 2:
        return None
    if len(set(y_true)) < 2:
        # Only one class in ground truth — BA is accuracy
        return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    return balanced_accuracy_score(y_true, y_pred)


# ============================================================
# Output buffer
# ============================================================
output_lines = []


def pr(text=""):
    """Print and buffer output."""
    print(text)
    output_lines.append(text)


# ============================================================
# Load all checkpoint data
# ============================================================
pr("=" * 80)
pr("REVIEWER ANALYSES — Proactive Affective Agent Pilot Study")
pr("=" * 80)
pr()

# Preload all checkpoints
all_data = {}  # (version, user_id) -> checkpoint dict
for ver in VERSIONS:
    for uid in PILOT_USERS:
        ck = load_checkpoint(ver, uid)
        if ck is not None:
            all_data[(ver, uid)] = ck

pr(f"Loaded checkpoints: {len(all_data)} version-user combinations")
for ver in VERSIONS:
    users_found = sorted([uid for uid in PILOT_USERS if (ver, uid) in all_data])
    pr(f"  {ver}: {len(users_found)} users — {users_found}")
pr()

# ============================================================
# 1. Per-User BA Distributions
# ============================================================
pr("=" * 80)
pr("ANALYSIS 1: Per-User Balanced Accuracy Distributions")
pr("=" * 80)
pr()
pr("For each version, BA is computed per user across ALL binary targets combined,")
pr("then we report distribution statistics across users.")
pr()

for ver in VERSIONS:
    users_found = sorted([uid for uid in PILOT_USERS if (ver, uid) in all_data])
    if not users_found:
        continue

    user_bas = {}
    for uid in users_found:
        ck = all_data[(ver, uid)]
        preds = ck["predictions"]
        gts = ck["ground_truths"]

        all_y_true = []
        all_y_pred = []

        # Binary targets (bool)
        for target in BINARY_TARGETS:
            for p, g in zip(preds, gts):
                pv = to_bool(p.get(target))
                gv = to_bool(g.get(target))
                if pv is not None and gv is not None:
                    all_y_true.append(gv)
                    all_y_pred.append(pv)

        # INT_availability (yes/no)
        for p, g in zip(preds, gts):
            pv = to_int_availability_bool(p.get(INT_TARGET))
            gv = to_int_availability_bool(g.get(INT_TARGET))
            if pv is not None and gv is not None:
                all_y_true.append(gv)
                all_y_pred.append(pv)

        if len(all_y_true) >= 10:
            ba = compute_ba_for_pairs(all_y_true, all_y_pred)
            if ba is not None:
                user_bas[uid] = ba

    if user_bas:
        bas = list(user_bas.values())
        pr(f"  {ver.upper():6s}: n_users={len(bas):2d}  "
           f"mean={np.mean(bas):.3f}  std={np.std(bas):.3f}  "
           f"min={np.min(bas):.3f}  max={np.max(bas):.3f}  "
           f"median={np.median(bas):.3f}")
        for uid in sorted(user_bas):
            pr(f"         user {uid:3d}: BA={user_bas[uid]:.3f}")
    pr()

# Per-target per-user BA table
pr("-" * 80)
pr("Per-target mean BA across users (targets with enough variation):")
pr()

target_ver_ba = {}  # (target, ver) -> list of per-user BAs
all_targets_for_table = BINARY_TARGETS + [INT_TARGET]

for ver in VERSIONS:
    users_found = sorted([uid for uid in PILOT_USERS if (ver, uid) in all_data])
    for target in all_targets_for_table:
        bas_for_target = []
        for uid in users_found:
            ck = all_data[(ver, uid)]
            preds = ck["predictions"]
            gts = ck["ground_truths"]

            y_true, y_pred = [], []
            for p, g in zip(preds, gts):
                if target == INT_TARGET:
                    pv = to_int_availability_bool(p.get(target))
                    gv = to_int_availability_bool(g.get(target))
                else:
                    pv = to_bool(p.get(target))
                    gv = to_bool(g.get(target))
                if pv is not None and gv is not None:
                    y_true.append(gv)
                    y_pred.append(pv)

            if len(y_true) >= 5 and len(set(y_true)) >= 2:
                ba = balanced_accuracy_score(y_true, y_pred)
                bas_for_target.append(ba)

        if bas_for_target:
            target_ver_ba[(target, ver)] = bas_for_target

# Print selected key targets
key_targets = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_ER_desire_State",
    "INT_availability",
    "Individual_level_happy_State",
    "Individual_level_sad_State",
    "Individual_level_lonely_State",
    "Individual_level_pain_State",
]

header = f"{'Target':<45s}"
for ver in VERSIONS:
    header += f" {ver.upper():>8s}"
pr(header)
pr("-" * len(header))

for target in key_targets:
    row = f"{target:<45s}"
    for ver in VERSIONS:
        bas = target_ver_ba.get((target, ver), [])
        if bas:
            row += f" {np.mean(bas):8.3f}"
        else:
            row += f" {'N/A':>8s}"
    pr(row)
pr()


# ============================================================
# 2. Binary State Base Rates
# ============================================================
pr("=" * 80)
pr("ANALYSIS 2: Binary State Base Rates (% Elevated / True)")
pr("=" * 80)
pr()
pr("Computed from ground truth labels across all 11 pilot users.")
pr("Uses the first available version's ground truths (all versions share the same GT).")
pr()

# Collect all GTs from any version (they should be the same)
all_gts_by_user = {}
for uid in PILOT_USERS:
    for ver in VERSIONS:
        if (ver, uid) in all_data:
            ck = all_data[(ver, uid)]
            all_gts_by_user[uid] = ck["ground_truths"]
            break

total_entries = sum(len(gts) for gts in all_gts_by_user.values())
pr(f"Total entries across {len(all_gts_by_user)} users: {total_entries}")
pr()

pr(f"{'Target':<45s} {'n':>5s} {'n_True':>7s} {'% True':>8s} {'n_False':>8s} {'% False':>8s}")
pr("-" * 85)

for target in BINARY_TARGETS:
    n_true = 0
    n_false = 0
    n_total = 0
    for uid, gts in all_gts_by_user.items():
        for g in gts:
            val = to_bool(g.get(target))
            if val is not None:
                n_total += 1
                if val:
                    n_true += 1
                else:
                    n_false += 1
    if n_total > 0:
        pr(f"{target:<45s} {n_total:5d} {n_true:7d} {100*n_true/n_total:7.1f}% {n_false:8d} {100*n_false/n_total:7.1f}%")

# INT_availability
n_true = 0
n_false = 0
n_total = 0
for uid, gts in all_gts_by_user.items():
    for g in gts:
        val = to_int_availability_bool(g.get(INT_TARGET))
        if val is not None:
            n_total += 1
            if val:
                n_true += 1
            else:
                n_false += 1
if n_total > 0:
    pr(f"{INT_TARGET:<45s} {n_total:5d} {n_true:7d} {100*n_true/n_total:7.1f}% {n_false:8d} {100*n_false/n_total:7.1f}%")

pr()
pr("Per-user base rate variation (key targets):")
pr()
for target in ["Individual_level_PA_State", "Individual_level_NA_State",
                "Individual_level_ER_desire_State", INT_TARGET]:
    pr(f"  {target}:")
    for uid in PILOT_USERS:
        gts = all_gts_by_user.get(uid, [])
        vals = []
        for g in gts:
            if target == INT_TARGET:
                v = to_int_availability_bool(g.get(target))
            else:
                v = to_bool(g.get(target))
            if v is not None:
                vals.append(v)
        if vals:
            rate = sum(vals) / len(vals) * 100
            pr(f"    user {uid:3d}: {rate:5.1f}% elevated ({sum(vals)}/{len(vals)})")
    pr()


# ============================================================
# 3. Tool Call Analysis
# ============================================================
pr("=" * 80)
pr("ANALYSIS 3: Tool Call Analysis (Agentic Versions)")
pr("=" * 80)
pr()

agentic_versions = ["v2", "v4", "v5", "v6"]

for ver in agentic_versions:
    all_records = []
    for uid in PILOT_USERS:
        recs = load_jsonl_records(ver, uid)
        all_records.extend(recs)

    if not all_records:
        pr(f"  {ver.upper()}: no JSONL records found")
        continue

    n_tool_calls_list = [r.get("n_tool_calls", 0) or 0 for r in all_records]
    n_rounds_list = [r.get("n_rounds", 0) or 0 for r in all_records]

    pr(f"  {ver.upper()}: {len(all_records)} records from JSONL")
    pr(f"    n_tool_calls: mean={np.mean(n_tool_calls_list):.2f}, "
       f"median={np.median(n_tool_calls_list):.1f}, "
       f"std={np.std(n_tool_calls_list):.2f}, "
       f"max={np.max(n_tool_calls_list)}")
    pr(f"    n_rounds:     mean={np.mean(n_rounds_list):.2f}, "
       f"median={np.median(n_rounds_list):.1f}, "
       f"std={np.std(n_rounds_list):.2f}, "
       f"max={np.max(n_rounds_list)}")

    # Tool frequency analysis
    tool_counter = Counter()
    for r in all_records:
        tc = r.get("tool_calls", [])
        if isinstance(tc, list):
            for t in tc:
                if isinstance(t, dict):
                    tool_counter[t.get("name", t.get("tool", "unknown"))] += 1
                elif isinstance(t, str):
                    tool_counter[t] += 1

    if tool_counter:
        pr(f"    Tool frequency (top 10):")
        for tool, count in tool_counter.most_common(10):
            pr(f"      {tool}: {count}")
    else:
        pr(f"    Tool calls field is empty in JSONL records (tool call logging may not have been enabled)")
        pr(f"    NOTE: Agentic versions use MCP tools via claude --print, but tool_calls may not be logged in JSONL")

    # Compare n_tool_calls for correct vs incorrect predictions (if any variation)
    if max(n_tool_calls_list) > 0:
        pr(f"\n    Tool calls vs prediction accuracy (binary targets):")
        for target in ["Individual_level_PA_State", "Individual_level_NA_State",
                        "Individual_level_ER_desire_State"]:
            correct_tc = []
            incorrect_tc = []
            for r in all_records:
                pred = r.get("prediction", {})
                gt = r.get("ground_truth", {})
                pv = to_bool(pred.get(target))
                gv = to_bool(gt.get(target))
                tc = r.get("n_tool_calls", 0) or 0
                if pv is not None and gv is not None:
                    if pv == gv:
                        correct_tc.append(tc)
                    else:
                        incorrect_tc.append(tc)
            if correct_tc and incorrect_tc:
                pr(f"      {target}:")
                pr(f"        correct: mean_tool_calls={np.mean(correct_tc):.2f} (n={len(correct_tc)})")
                pr(f"        incorrect: mean_tool_calls={np.mean(incorrect_tc):.2f} (n={len(incorrect_tc)})")

    pr()

# Additional analysis: check elapsed_seconds as proxy for agentic effort
pr("-" * 80)
pr("Elapsed time as proxy for agentic effort:")
pr()
for ver in VERSIONS:
    all_records = []
    for uid in PILOT_USERS:
        recs = load_jsonl_records(ver, uid)
        all_records.extend(recs)

    if not all_records:
        continue

    times = [r.get("elapsed_seconds", 0) or 0 for r in all_records if r.get("elapsed_seconds")]
    if times:
        pr(f"  {ver.upper()}: mean={np.mean(times):.1f}s, median={np.median(times):.1f}s, "
           f"std={np.std(times):.1f}s, max={np.max(times):.1f}s  (n={len(times)})")
pr()


# ============================================================
# 4. Performance Stratified by Diary Availability
# ============================================================
pr("=" * 80)
pr("ANALYSIS 4: Performance Stratified by Diary Availability")
pr("=" * 80)
pr()

# Since all JSONL records show has_diary=True for all entries, we instead compare
# sensing-only versions (v1/v2) vs multimodal versions (v3/v4/v5/v6/callm)
# which is the actual design comparison for diary impact.

pr("NOTE: All EMA entries in the pilot have associated diary text (has_diary=True")
pr("for all records). The structured design comparison between sensing-only (v1/v2)")
pr("vs multimodal (v3/v4/v5/v6/CALLM) captures the diary impact instead.")
pr()
pr("Design comparison — Diary impact on BA (multimodal vs sensing-only):")
pr()

# Structured: v1 (sensing) vs v3 (multimodal)
# Agentic:    v2 (sensing) vs v4 (multimodal)
comparisons = [
    ("Structured", "v1", "v3"),
    ("Agentic", "v2", "v4"),
    ("Agentic+RAG", "v5", "v6"),
]

for label, sensing_ver, multi_ver in comparisons:
    pr(f"  {label}: {sensing_ver.upper()} (sensing-only) vs {multi_ver.upper()} (multimodal)")

    for target in key_targets:
        sensing_yt, sensing_yp = [], []
        multi_yt, multi_yp = [], []

        for uid in PILOT_USERS:
            for ver, yt_list, yp_list in [(sensing_ver, sensing_yt, sensing_yp),
                                           (multi_ver, multi_yt, multi_yp)]:
                if (ver, uid) not in all_data:
                    continue
                ck = all_data[(ver, uid)]
                for p, g in zip(ck["predictions"], ck["ground_truths"]):
                    if target == INT_TARGET:
                        pv = to_int_availability_bool(p.get(target))
                        gv = to_int_availability_bool(g.get(target))
                    else:
                        pv = to_bool(p.get(target))
                        gv = to_bool(g.get(target))
                    if pv is not None and gv is not None:
                        yt_list.append(gv)
                        yp_list.append(pv)

        s_ba = balanced_accuracy_score(sensing_yt, sensing_yp) if len(set(sensing_yt)) >= 2 else None
        m_ba = balanced_accuracy_score(multi_yt, multi_yp) if len(set(multi_yt)) >= 2 else None

        if s_ba is not None and m_ba is not None:
            delta = m_ba - s_ba
            marker = "+" if delta > 0 else ""
            pr(f"    {target:<42s}  {sensing_ver}={s_ba:.3f}  {multi_ver}={m_ba:.3f}  delta={marker}{delta:.3f}")
    pr()

# Alternative: stratify by diary_length (short vs long) within multimodal versions
pr("-" * 80)
pr("Diary length stratification (v4, v6): short (<=50 chars) vs long (>50 chars)")
pr()

for ver in ["v4", "v6"]:
    short_yt, short_yp = defaultdict(list), defaultdict(list)
    long_yt, long_yp = defaultdict(list), defaultdict(list)

    for uid in PILOT_USERS:
        recs = load_jsonl_records(ver, uid)
        for r in recs:
            pred = r.get("prediction", {})
            gt = r.get("ground_truth", {})
            dl = r.get("diary_length", 0) or 0

            for target in key_targets:
                if target == INT_TARGET:
                    pv = to_int_availability_bool(pred.get(target))
                    gv = to_int_availability_bool(gt.get(target))
                else:
                    pv = to_bool(pred.get(target))
                    gv = to_bool(gt.get(target))

                if pv is not None and gv is not None:
                    if dl <= 50:
                        short_yt[target].append(gv)
                        short_yp[target].append(pv)
                    else:
                        long_yt[target].append(gv)
                        long_yp[target].append(pv)

    pr(f"  {ver.upper()}:")
    for target in key_targets:
        s_n = len(short_yt[target])
        l_n = len(long_yt[target])
        s_ba = (balanced_accuracy_score(short_yt[target], short_yp[target])
                if s_n >= 10 and len(set(short_yt[target])) >= 2 else None)
        l_ba = (balanced_accuracy_score(long_yt[target], long_yp[target])
                if l_n >= 10 and len(set(long_yt[target])) >= 2 else None)
        s_str = f"{s_ba:.3f}" if s_ba is not None else "N/A"
        l_str = f"{l_ba:.3f}" if l_ba is not None else "N/A"
        pr(f"    {target:<42s}  short(n={s_n:4d})={s_str:>6s}  long(n={l_n:4d})={l_str:>6s}")
    pr()


# ============================================================
# 5. Joint Intervention Opportunity Accuracy
# ============================================================
pr("=" * 80)
pr("ANALYSIS 5: Joint Intervention Opportunity Accuracy")
pr("=" * 80)
pr()
pr("An intervention opportunity requires BOTH:")
pr("  - ER_desire_State = True (elevated emotion regulation desire)")
pr("  - INT_availability = 'yes' (available for intervention)")
pr()
pr("We compute: (a) accuracy where both are correctly predicted,")
pr("(b) joint True rate in ground truth, (c) precision/recall for joint detection.")
pr()

for ver in VERSIONS:
    users_found = sorted([uid for uid in PILOT_USERS if (ver, uid) in all_data])
    if not users_found:
        continue

    joint_correct = 0
    joint_total = 0
    gt_joint_true = 0
    pred_joint_true = 0
    both_joint_true = 0  # TP for joint

    er_correct = 0
    int_correct = 0

    for uid in users_found:
        ck = all_data[(ver, uid)]
        for p, g in zip(ck["predictions"], ck["ground_truths"]):
            er_p = to_bool(p.get("Individual_level_ER_desire_State"))
            er_g = to_bool(g.get("Individual_level_ER_desire_State"))
            int_p = to_int_availability_bool(p.get(INT_TARGET))
            int_g = to_int_availability_bool(g.get(INT_TARGET))

            if er_p is not None and er_g is not None and int_p is not None and int_g is not None:
                joint_total += 1

                # Both correct
                if er_p == er_g and int_p == int_g:
                    joint_correct += 1
                if er_p == er_g:
                    er_correct += 1
                if int_p == int_g:
                    int_correct += 1

                # Joint intervention signal
                gt_joint = er_g and int_g
                pred_joint = er_p and int_p

                if gt_joint:
                    gt_joint_true += 1
                if pred_joint:
                    pred_joint_true += 1
                if gt_joint and pred_joint:
                    both_joint_true += 1

    if joint_total > 0:
        joint_acc = joint_correct / joint_total
        er_acc = er_correct / joint_total
        int_acc = int_correct / joint_total
        gt_rate = gt_joint_true / joint_total
        precision = both_joint_true / pred_joint_true if pred_joint_true > 0 else 0
        recall = both_joint_true / gt_joint_true if gt_joint_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        pr(f"  {ver.upper():6s}: n={joint_total}  "
           f"joint_acc={joint_acc:.3f}  "
           f"ER_acc={er_acc:.3f}  "
           f"INT_acc={int_acc:.3f}")
        pr(f"         GT joint rate={gt_rate:.3f} ({gt_joint_true}/{joint_total})  "
           f"Pred joint={pred_joint_true}  "
           f"TP={both_joint_true}")
        pr(f"         Joint precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")

pr()

# Per-user joint accuracy for best versions
pr("Per-user joint accuracy (v4, v6, callm):")
for ver in ["v4", "v6", "callm"]:
    pr(f"\n  {ver.upper()}:")
    for uid in PILOT_USERS:
        if (ver, uid) not in all_data:
            continue
        ck = all_data[(ver, uid)]
        jc, jt = 0, 0
        for p, g in zip(ck["predictions"], ck["ground_truths"]):
            er_p = to_bool(p.get("Individual_level_ER_desire_State"))
            er_g = to_bool(g.get("Individual_level_ER_desire_State"))
            int_p = to_int_availability_bool(p.get(INT_TARGET))
            int_g = to_int_availability_bool(g.get(INT_TARGET))
            if all(x is not None for x in [er_p, er_g, int_p, int_g]):
                jt += 1
                if er_p == er_g and int_p == int_g:
                    jc += 1
        if jt > 0:
            pr(f"    user {uid:3d}: {jc}/{jt} = {jc/jt:.3f}")
pr()


# ============================================================
# 6. ML Baselines on Same 11 Users
# ============================================================
pr("=" * 80)
pr("ANALYSIS 6: ML Baselines Comparison")
pr("=" * 80)
pr()
pr("ML baselines were trained on 5-fold cross-validation over ALL users in the dataset,")
pr("not the specific 11 pilot users. Per-user predictions are not available in the")
pr("saved outputs (fold directories are empty). We report aggregate ML baseline metrics")
pr("for reference alongside the LLM agent results on the 11 pilot users.")
pr()

# Load ML baseline metrics
ml_metrics = {}
for fn, label in [
    ("outputs/ml_baselines/ml_baseline_metrics.json", "ML"),
    ("outputs/advanced_baselines/text/text_baseline_metrics.json", "Text"),
    ("outputs/advanced_baselines/dl/dl_baseline_metrics.json", "DL"),
    ("outputs/advanced_baselines/lstm/lstm_baseline_metrics.json", "LSTM"),
    ("outputs/advanced_baselines/combined/combined_baseline_metrics.json", "Combined"),
    ("outputs/advanced_baselines/transformer/transformer_baseline_metrics.json", "Transformer"),
]:
    if os.path.exists(fn):
        with open(fn) as f:
            data = json.load(f)
        ml_metrics[label] = data

# Print ML baselines
if ml_metrics:
    pr("ML/DL Baseline Aggregate Metrics (5-fold CV, all users):")
    pr()

    # Collect all model names
    all_models = []
    for label, data in ml_metrics.items():
        for model_name in data:
            all_models.append((label, model_name))

    pr(f"{'Model':<25s} {'Mean BA':>10s} {'BA Std':>10s} {'Mean F1':>10s} {'F1 Std':>10s} {'Mean MAE':>12s}")
    pr("-" * 80)

    for label, data in ml_metrics.items():
        for model_name, model_data in data.items():
            # Aggregate BA across binary targets
            bas = []
            f1s = []
            for target in BINARY_TARGETS:
                if target in model_data:
                    td = model_data[target]
                    if "ba_mean" in td:
                        bas.append(td["ba_mean"])
                    if "f1_mean" in td:
                        f1s.append(td["f1_mean"])

            agg = model_data.get("_aggregate", {})
            mae = agg.get("mean_mae")

            ba_str = f"{np.mean(bas):.3f}" if bas else "N/A"
            ba_std_str = f"{np.std(bas):.3f}" if bas else "N/A"
            f1_str = f"{np.mean(f1s):.3f}" if f1s else "N/A"
            f1_std_str = f"{np.std(f1s):.3f}" if f1s else "N/A"
            mae_str = f"{mae:.3f}" if mae is not None else "N/A"

            display_name = f"{model_name}"
            pr(f"{display_name:<25s} {ba_str:>10s} {ba_std_str:>10s} {f1_str:>10s} {f1_std_str:>10s} {mae_str:>12s}")

    pr()
    pr("NOTE: ML baselines are trained/tested on the full dataset (~16K entries, ~400 users)")
    pr("while LLM agents are evaluated on 11 pilot users (~1000 entries). Direct comparison")
    pr("should note this population difference.")

pr()

# Compute LLM agent aggregate metrics for comparison
pr("-" * 80)
pr("LLM Agent Aggregate Metrics (11 pilot users) for comparison:")
pr()

pr(f"{'Version':<10s} {'Mean BA':>10s} {'Mean F1':>10s} {'PA MAE':>10s} {'NA MAE':>10s}")
pr("-" * 50)

for ver in VERSIONS:
    users_found = sorted([uid for uid in PILOT_USERS if (ver, uid) in all_data])
    if not users_found:
        continue

    all_yt = defaultdict(list)
    all_yp = defaultdict(list)
    pa_maes = []
    na_maes = []

    for uid in users_found:
        ck = all_data[(ver, uid)]
        for p, g in zip(ck["predictions"], ck["ground_truths"]):
            # Binary targets
            for target in BINARY_TARGETS:
                if target == INT_TARGET:
                    pv = to_int_availability_bool(p.get(target))
                    gv = to_int_availability_bool(g.get(target))
                else:
                    pv = to_bool(p.get(target))
                    gv = to_bool(g.get(target))
                if pv is not None and gv is not None:
                    all_yt[target].append(gv)
                    all_yp[target].append(pv)

            # INT target
            pv = to_int_availability_bool(p.get(INT_TARGET))
            gv = to_int_availability_bool(g.get(INT_TARGET))
            if pv is not None and gv is not None:
                all_yt[INT_TARGET].append(gv)
                all_yp[INT_TARGET].append(pv)

            # Continuous (PANAS)
            try:
                pp = float(p.get("PANAS_Pos", 0))
                gp = float(g.get("PANAS_Pos", 0))
                pa_maes.append(abs(pp - gp))
            except (TypeError, ValueError):
                pass
            try:
                pn = float(p.get("PANAS_Neg", 0))
                gn = float(g.get("PANAS_Neg", 0))
                na_maes.append(abs(pn - gn))
            except (TypeError, ValueError):
                pass

    # Compute aggregate BA and F1
    bas = []
    f1s = []
    for target in BINARY_TARGETS + [INT_TARGET]:
        yt = all_yt[target]
        yp = all_yp[target]
        if len(yt) >= 10 and len(set(yt)) >= 2:
            bas.append(balanced_accuracy_score(yt, yp))
            f1s.append(f1_score(yt, yp, average="macro"))

    ba_str = f"{np.mean(bas):.3f}" if bas else "N/A"
    f1_str = f"{np.mean(f1s):.3f}" if f1s else "N/A"
    pa_str = f"{np.mean(pa_maes):.3f}" if pa_maes else "N/A"
    na_str = f"{np.mean(na_maes):.3f}" if na_maes else "N/A"

    pr(f"{ver.upper():<10s} {ba_str:>10s} {f1_str:>10s} {pa_str:>10s} {na_str:>10s}")

pr()


# ============================================================
# Summary
# ============================================================
pr("=" * 80)
pr("SUMMARY OF KEY FINDINGS")
pr("=" * 80)
pr()
pr("1. Per-user BA: Reports whether agentic advantage is consistent or driven by")
pr("   outliers. Check std and min/max spread.")
pr()
pr("2. Base rates: Several targets show strong class imbalance (e.g., afraid,")
pr("   miserable may be rarely True). This context is essential for interpreting BA.")
pr()
pr("3. Tool calls: The JSONL tool_calls field is empty for agentic versions —")
pr("   tool call logging was not captured in the structured JSONL output format.")
pr("   Elapsed time may serve as a proxy for agentic effort.")
pr()
pr("4. Diary impact: Compared via sensing-only (v1/v2) vs multimodal (v3/v4)")
pr("   design, since all EMA entries have diary text available. Also stratified")
pr("   by diary length (short <=50 chars vs long >50 chars).")
pr()
pr("5. Joint intervention: Reports precision/recall for the joint detection of")
pr("   ER_desire_State=True AND INT_availability='yes' — the actionable signal.")
pr()
pr("6. ML baselines: Aggregate metrics only (per-user not available). Population")
pr("   differs from pilot users.")
pr()

# ============================================================
# Save output
# ============================================================
os.makedirs("outputs", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(output_lines))
    f.write("\n")

print(f"\n[Output saved to {OUTPUT_FILE}]")
