import os,glob,re,collections
base='/Users/zwang/projects/proactive-affective-agent'
roots=[os.path.join(base,'outputs/pilot'), os.path.join(base,'outputs/pilot_v2')]
seen=collections.defaultdict(set)
for root in roots:
    for path in glob.glob(root+'/**/*_checkpoint.json', recursive=True):
        name=os.path.basename(path)
        m=re.search(r'((?:callm|v[1-6]))_user(\d+)_checkpoint\.json', name)
        if m:
            seen[m.group(1)].add(int(m.group(2)))
    for path in glob.glob(root+'/**/*_user*_records.jsonl', recursive=True):
        name=os.path.basename(path)
        m=re.search(r'((?:callm|v[1-6]))_user(\d+)_records\.jsonl', name)
        if m:
            seen[m.group(1)].add(int(m.group(2)))
print('sonnet/claude users by version:')
for v in sorted(seen):
    print(v, sorted(seen[v]))
all_users=sorted(set().union(*seen.values()) if seen else set())
print('ALL', all_users)
