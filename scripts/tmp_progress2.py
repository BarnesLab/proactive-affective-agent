import os, glob, re, subprocess
from collections import defaultdict

base = '/Users/zwang/projects/proactive-affective-agent'
versions = ['callm','v1','v2','v3','v4','v5','v6']

def count_records(pattern):
    """Count users with records files and line counts"""
    results = defaultdict(dict)  # version -> {user: lines}
    for f in glob.glob(pattern):
        name = os.path.basename(f)
        # Match both "callm_user71" and "gpt-callm_user71"
        m = re.search(r'(?:gpt-)?(callm|v[1-6])_user(\d+)_records\.jsonl', name)
        if m:
            ver = m.group(1)
            uid = int(m.group(2))
            lines = sum(1 for _ in open(f))
            results[ver][uid] = lines
    return results

def count_predictions(pattern):
    """Count users in prediction CSVs"""
    results = defaultdict(set)
    for f in glob.glob(pattern):
        name = os.path.basename(f)
        m = re.search(r'(?:gpt-)?(callm|v[1-6])_predictions\.csv', name)
        if m:
            ver = m.group(1)
            import csv
            with open(f) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    uid_col = 'Study_ID' if 'Study_ID' in row else 'study_id' if 'study_id' in row else None
                    if uid_col:
                        results[ver].add(int(float(row[uid_col])))
    return results

# Sonnet
print("=" * 60)
print("SONNET (Claude) - outputs/pilot + outputs/pilot_v2")
print("=" * 60)
sonnet_records = defaultdict(dict)
for root in ['outputs/pilot', 'outputs/pilot_v2']:
    r = count_records(os.path.join(base, root, '*_records.jsonl'))
    for v, users in r.items():
        sonnet_records[v].update(users)

sonnet_preds = defaultdict(set)
for root in ['outputs/pilot', 'outputs/pilot_v2']:
    p = count_predictions(os.path.join(base, root, '*_predictions.csv'))
    for v, users in p.items():
        sonnet_preds[v].update(users)

all_sonnet_users = set()
for v in versions:
    users = sorted(sonnet_records.get(v, {}).keys())
    all_sonnet_users.update(users)
    pred_users = sonnet_preds.get(v, set())
    print(f"  {v}: {len(users)} users with records, {len(pred_users)} in predictions")
    print(f"       users: {users}")
print(f"  ALL UNIQUE: {len(all_sonnet_users)} -> {sorted(all_sonnet_users)}")

# GPT 5.1 mini
print("\n" + "=" * 60)
print("GPT 5.1 CODEX MINI - outputs/pilot_gpt51mini + archive")
print("=" * 60)
mini_records = count_records(os.path.join(base, 'outputs/pilot_gpt51mini', '*_records.jsonl'))
# Also check archive
for adir in glob.glob(os.path.join(base, 'outputs/_archive/pilot_gpt51codexmini*')):
    r = count_records(os.path.join(adir, '*_records.jsonl'))
    for v, users in r.items():
        for u, lines in users.items():
            if u not in mini_records[v] or lines > mini_records[v][u]:
                mini_records[v][u] = lines

all_mini_users = set()
for v in versions:
    users = sorted(mini_records.get(v, {}).keys())
    all_mini_users.update(users)
    print(f"  {v}: {len(users)} users -> {users}")
print(f"  ALL UNIQUE: {len(all_mini_users)} -> {sorted(all_mini_users)}")

# Currently running
print("\n" + "=" * 60)
print("CURRENTLY RUNNING")
print("=" * 60)
r = subprocess.run(['pgrep', '-afl', 'scripts/run_pilot.py'], capture_output=True, text=True)
running = defaultdict(list)
for line in r.stdout.strip().split('\n'):
    if not line: continue
    m_ver = re.search(r'--version (gpt-\w+|callm|v\d)', line)
    m_user = re.search(r'--users (\d+)', line)
    m_model = re.search(r'--model (\S+)', line)
    if m_ver and m_user:
        model = m_model.group(1) if m_model else '?'
        running[model].append(f"{m_ver.group(1)}/user{m_user.group(1)}")
for model, tasks in running.items():
    print(f"  {model}: {len(tasks)} tasks")
    for t in sorted(tasks):
        print(f"    {t}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY: 30-user target")
print("=" * 60)
# Users that appear in ALL 7 versions for sonnet
sonnet_per_ver = [set(sonnet_records.get(v, {}).keys()) for v in versions]
sonnet_all7 = set.intersection(*sonnet_per_ver) if all(sonnet_per_ver) else set()
mini_per_ver = [set(mini_records.get(v, {}).keys()) for v in versions]
mini_all7 = set.intersection(*mini_per_ver) if all(mini_per_ver) else set()

print(f"Sonnet: {len(sonnet_all7)} users fully done (all 7 versions) -> {sorted(sonnet_all7)}")
print(f"Mini:   {len(mini_all7)} users fully done (all 7 versions) -> {sorted(mini_all7)}")
overlap = sonnet_all7 & mini_all7
print(f"Both:   {len(overlap)} users done in BOTH -> {sorted(overlap)}")
print(f"Need:   {max(0, 30 - len(overlap))} more users in both models")
