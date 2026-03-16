#!/usr/bin/env python3
"""Auto-progress: diagnose gaps, plan next batch, optionally launch."""
import os, glob, re, subprocess, json, sys
from collections import defaultdict

base = '/Users/zwang/Documents/proactive-affective-agent'
versions = ['callm','v1','v2','v3','v4','v5','v6']
gpt_versions = ['gpt-' + v for v in versions]

DRY_RUN = '--dry-run' in sys.argv

def get_all_users():
    """Get all available user IDs from the dataset"""
    try:
        sys.path.insert(0, base)
        from src.data.loader import DataLoader
        dl = DataLoader()
        return sorted(dl.get_user_ids())
    except Exception:
        pass
    # Fallback: extract from parquet filenames
    ids = set()
    for f in glob.glob(os.path.join(base, 'data/processed/hourly/motion/*_motion_hourly.parquet')):
        m = re.search(r'/(\d+)_motion_hourly', f)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)

def get_completed(output_dirs, prefix=''):
    """Get {version: {user_id}} for completed records, with quality check"""
    done = defaultdict(set)
    for d in output_dirs:
        full = os.path.join(base, d)
        for f in glob.glob(full + '/*_records.jsonl'):
            name = os.path.basename(f)
            m = re.search(r'(?:gpt-)?(callm|v[1-6])_user(\d+)_records\.jsonl', name, re.IGNORECASE)
            if m:
                # Quality check: must have >10 records and predictions present
                try:
                    with open(f) as fh:
                        lines = fh.readlines()
                    records = [json.loads(l) for l in lines if l.strip()]
                    if len(records) < 10:
                        continue  # too few records, needs rerun
                    # Check for missing predictions: prediction field should be a non-empty dict
                    missing = sum(1 for r in records if not isinstance(r.get('prediction'), dict) or len(r.get('prediction', {})) == 0)
                    if missing > len(records) * 0.3:
                        continue  # too many missing predictions, needs rerun
                except Exception:
                    continue
                done[m.group(1).lower()].add(int(m.group(2)))
    return done

def get_running():
    """Get currently running tasks as {(model, version, user)}"""
    # Use ps aux and filter for run_pilot.py to avoid pgrep multiline issues
    r = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    running = set()
    for line in r.stdout.strip().split('\n'):
        if 'run_pilot.py' not in line or 'grep' in line:
            continue
        m_ver = re.search(r'--version (\S+)', line)
        m_user = re.search(r'--users (\d+)', line)
        m_model = re.search(r'--model (\S+)', line)
        if m_ver and m_user and m_model:
            running.add((m_model.group(1), m_ver.group(1), int(m_user.group(1))))
    return running

def plan_40_users(sonnet_done, mini_done, all_users):
    """Pick best 40 users — prioritize by EMA sample count (most data per user first),
    with tiebreak by existing experiment completion."""
    # Get EMA sample counts per user
    from src.data.loader import DataLoader
    dl = DataLoader()
    ema = dl.load_all_ema()
    ema_counts = ema.groupby('Study_ID').size().to_dict()

    scores = {}
    for uid in all_users:
        # Primary key: EMA sample count (more samples = better user)
        ema_count = ema_counts.get(uid, 0)
        # Secondary key: existing experiment completion (tiebreak)
        completion = 0
        for v in versions:
            if uid in sonnet_done.get(v, set()): completion += 1
            if uid in mini_done.get(v, set()): completion += 1
        scores[uid] = (ema_count, completion)
    # Sort by EMA count desc, then completion desc
    ranked = sorted(scores.keys(), key=lambda u: (-scores[u][0], -scores[u][1]))
    return ranked[:50]

# === MAIN ===
all_users = get_all_users()
sonnet_done = get_completed(['outputs/pilot', 'outputs/pilot_v2'])
mini_done = get_completed(['outputs/pilot_gpt51mini'])
# Also check archive
for adir in glob.glob(os.path.join(base, 'outputs/_archive/pilot_gpt51codexmini*')):
    rel = os.path.relpath(adir, base)
    extra = get_completed([rel])
    for v, users in extra.items():
        mini_done[v].update(users)

running = get_running()
target_users = plan_40_users(sonnet_done, mini_done, all_users)

print(f"Target {len(target_users)} users: {target_users}")
print(f"Currently running: {len(running)} tasks")

# Find gaps
sonnet_gaps = []
mini_gaps = []
for uid in target_users:
    for v in versions:
        if uid not in sonnet_done.get(v, set()):
            if ('sonnet', v, uid) not in running and ('sonnet', f'v{v}' if v != 'callm' else v, uid) not in running:
                sonnet_gaps.append((v, uid))
        gv = f'gpt-{v}'
        if uid not in mini_done.get(v, set()):
            if ('gpt-5.1-codex-mini', gv, uid) not in running:
                mini_gaps.append((gv, uid))

print(f"\nSonnet gaps: {len(sonnet_gaps)} tasks needed")
print(f"Mini gaps:   {len(mini_gaps)} tasks needed")

# How many slots available?
current_running = len(running)
max_parallel = 15
available = max(0, max_parallel - current_running)

print(f"\nRunning: {current_running}, Max: {max_parallel}, Available slots: {available}")

# USER-FIRST SCHEDULING: maximize fully completed users PER MODEL
# Each model independently: user is "complete" when all 7 versions done.
# Sonnet: maximize complete users. Mini: same, independently.

def user_first_schedule_per_model(gaps, model_name, target_users):
    """For one model, return tasks sorted: users closest to 7/7 first."""
    user_gaps = defaultdict(list)
    for v, uid in gaps:
        user_gaps[uid].append((model_name, v, uid))
    
    # Sort users by fewest remaining gaps (closest to completion)
    users_sorted = sorted(
        [u for u in target_users if u in user_gaps],
        key=lambda u: len(user_gaps[u])
    )
    
    # Report
    total = len(target_users)
    complete = [u for u in target_users if u not in user_gaps]
    print(f"  {model_name}: {len(complete)}/{total} users fully complete (7/7)")
    for uid in users_sorted[:5]:
        remaining = len(user_gaps[uid])
        print(f"    User {uid}: {7-remaining}/7 ({remaining} remaining)")
    if len(users_sorted) > 5:
        print(f"    ... and {len(users_sorted)-5} more users with gaps")
    
    # Flatten: all gaps for closest user first, then next user, etc.
    ordered = []
    for uid in users_sorted:
        ordered.extend(user_gaps[uid])
    return ordered

print("\n--- User-first scheduling (per model) ---")
sonnet_ordered = user_first_schedule_per_model(sonnet_gaps, 'sonnet', target_users)
mini_ordered = user_first_schedule_per_model(mini_gaps, 'gpt-5.1-codex-mini', target_users)

# Interleave: alternate between models to keep both progressing
to_launch = []
si, mi = 0, 0
while len(to_launch) < available:
    added = False
    if si < len(sonnet_ordered):
        to_launch.append(sonnet_ordered[si])
        si += 1
        added = True
    if len(to_launch) >= available:
        break
    if mi < len(mini_ordered):
        to_launch.append(mini_ordered[mi])
        mi += 1
        added = True
    if not added:
        break

print(f"\nWill launch {len(to_launch)} new tasks:")
for model, ver, uid in to_launch:
    print(f"  {model} {ver} user {uid}")

if DRY_RUN:
    print("\n[DRY RUN] Not launching anything.")
    sys.exit(0)

# Launch
venv_python = os.path.join(base, '.venv/bin/python')
for model, ver, uid in to_launch:
    if model == 'sonnet':
        out_dir = os.path.join(base, 'outputs/pilot_v2')
        cmd = f'cd {base} && {venv_python} scripts/run_pilot.py --version {ver} --users {uid} --model sonnet --delay 1.0 --output-dir {out_dir} --verbose'
    else:
        out_dir = os.path.join(base, 'outputs/pilot_gpt51mini')
        cmd = f'cd {base} && {venv_python} scripts/run_pilot.py --version {ver} --users {uid} --model gpt-5.1-codex-mini --delay 1.0 --output-dir {out_dir} --verbose'
    
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{ver}_user{uid}.log')
    
    full_cmd = f'nohup {cmd} > {log_file} 2>&1 &'
    print(f"  Launching: {model} {ver} user {uid}")
    os.system(full_cmd)

print(f"\nLaunched {len(to_launch)} tasks.")

# Save state
state = {
    'target_users': target_users,
    'sonnet_done_count': {v: len(users) for v, users in sonnet_done.items()},
    'mini_done_count': {v: len(users) for v, users in mini_done.items()},
    'running': current_running + len(to_launch),
    'sonnet_gaps_remaining': len(sonnet_gaps) - sum(1 for m,_,_ in to_launch if m == 'sonnet'),
    'mini_gaps_remaining': len(mini_gaps) - sum(1 for m,_,_ in to_launch if m == 'gpt-5.1-codex-mini'),
}
state_path = os.path.expanduser('~/.openclaw/workspace/memory/project-progress.json')
os.makedirs(os.path.dirname(state_path), exist_ok=True)
with open(state_path, 'w') as f:
    json.dump(state, f, indent=2)
print(f"State saved to {state_path}")
