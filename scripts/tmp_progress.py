import os, glob, re, json
from collections import defaultdict

base = '/Users/zwang/projects/proactive-affective-agent'
versions = ['callm','v1','v2','v3','v4','v5','v6']

# 1) Sonnet (Claude) progress - outputs/pilot and outputs/pilot_v2
sonnet_done = defaultdict(set)
for root in ['outputs/pilot', 'outputs/pilot_v2']:
    full = os.path.join(base, root)
    for f in glob.glob(full + '/**/*_checkpoint.json', recursive=True):
        m = re.search(r'(callm|v[1-6])_user(\d+)_checkpoint\.json', os.path.basename(f))
        if m:
            with open(f) as fh:
                try:
                    ck = json.load(fh)
                    total = ck.get('total_entries', 0)
                    done = ck.get('completed_entries', 0)
                    if done >= total and total > 0:
                        sonnet_done[m.group(1)].add(int(m.group(2)))
                except:
                    pass

# 2) GPT 5.1 mini progress - outputs/pilot_gpt51mini
mini_done = defaultdict(set)
mini_running = defaultdict(dict)
mini_root = os.path.join(base, 'outputs/pilot_gpt51mini')
for f in glob.glob(mini_root + '/*_checkpoint.json'):
    m = re.search(r'((?:gpt-)?(?:callm|v[1-6]))_user(\d+)_checkpoint\.json', os.path.basename(f))
    if m:
        ver = m.group(1).replace('gpt-','')
        uid = int(m.group(2))
        with open(f) as fh:
            try:
                ck = json.load(fh)
                total = ck.get('total_entries', 0)
                done = ck.get('completed_entries', 0)
                if done >= total and total > 0:
                    mini_done[ver].add(uid)
                else:
                    mini_running[ver][uid] = f"{done}/{total}"
            except:
                pass

# Also check _archive for completed mini runs
for root in glob.glob(os.path.join(base, 'outputs/_archive/pilot_gpt51codexmini*')):
    for f in glob.glob(root + '/*_checkpoint.json'):
        m = re.search(r'((?:gpt-)?(?:callm|v[1-6]))_user(\d+)_checkpoint\.json', os.path.basename(f))
        if m:
            ver = m.group(1).replace('gpt-','')
            uid = int(m.group(2))
            with open(f) as fh:
                try:
                    ck = json.load(fh)
                    total = ck.get('total_entries', 0)
                    done = ck.get('completed_entries', 0)
                    if done >= total and total > 0:
                        mini_done[ver].add(uid)
                except:
                    pass

print("=== SONNET (Claude) completed users per version ===")
all_sonnet = set()
for v in versions:
    users = sorted(sonnet_done.get(v, set()))
    all_sonnet.update(users)
    print(f"  {v}: {len(users)} users -> {users}")
print(f"  UNION: {len(all_sonnet)} unique users -> {sorted(all_sonnet)}")

print("\n=== GPT 5.1 CODEX MINI completed users per version ===")
all_mini = set()
for v in versions:
    users = sorted(mini_done.get(v, set()))
    all_mini.update(users)
    print(f"  {v}: {len(users)} users -> {users}")
print(f"  UNION: {len(all_mini)} unique users -> {sorted(all_mini)}")

print("\n=== GPT 5.1 CODEX MINI in-progress ===")
for v in versions:
    if v in mini_running:
        for uid, prog in sorted(mini_running[v].items()):
            print(f"  {v} user {uid}: {prog}")

print(f"\n=== SUMMARY ===")
print(f"Sonnet: {len(all_sonnet)}/30 users with at least some version complete")
print(f"Mini:   {len(all_mini)}/30 users with at least some version complete")
# Users done in ALL 7 versions
sonnet_all7 = set.intersection(*[sonnet_done.get(v, set()) for v in versions]) if all(v in sonnet_done for v in versions) else set()
mini_all7 = set.intersection(*[mini_done.get(v, set()) for v in versions]) if all(v in mini_done for v in versions) else set()
print(f"Sonnet fully done (all 7 versions): {len(sonnet_all7)} -> {sorted(sonnet_all7)}")
print(f"Mini fully done (all 7 versions):   {len(mini_all7)} -> {sorted(mini_all7)}")

# Target 30 overlap
print(f"\n=== OVERLAP (users done in BOTH models, all 7 versions) ===")
both = sonnet_all7 & mini_all7
print(f"Both done: {len(both)} -> {sorted(both)}")
print(f"Need {30 - len(both)} more to reach 30")
