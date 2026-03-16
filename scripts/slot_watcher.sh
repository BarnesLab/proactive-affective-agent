#!/bin/bash
cd /Users/zwang/Documents/proactive-affective-agent
source .venv/bin/activate

MAX_PARALLEL=15
LOG="/tmp/slot_watcher.log"
LOCKFILE="/tmp/auto_progress.lock"

echo "[$(date)] Slot watcher started (max=$MAX_PARALLEL)" >> "$LOG"

while true; do
    # Dedup: kill duplicate processes
    python3 -c "
import subprocess, re
from collections import defaultdict
r = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
tasks = defaultdict(list)
for line in r.stdout.strip().split('\n'):
    if 'run_pilot.py' not in line or 'grep' in line: continue
    pid = int(line.split()[1])
    m_ver = re.search(r'--version (\S+)', line)
    m_user = re.search(r'--users (\d+)', line)
    m_model = re.search(r'--model (\S+)', line)
    if m_ver and m_user and m_model:
        key = f'{m_model.group(1)}_{m_ver.group(1)}_user{m_user.group(1)}'
        tasks[key].append(pid)
for key, pids in tasks.items():
    for pid in pids[1:]:
        subprocess.run(['kill', str(pid)])
" 2>/dev/null

    RUNNING=$(ps aux | grep "run_pilot.py" | grep -v grep | wc -l | tr -d ' ')
    if [ "$RUNNING" -lt "$MAX_PARALLEL" ]; then
        # Lock to prevent concurrent auto_progress calls
        if mkdir "$LOCKFILE" 2>/dev/null; then
            SLOTS=$((MAX_PARALLEL - RUNNING))
            echo "[$(date)] $RUNNING running, $SLOTS slots free — filling" >> "$LOG"
            python scripts/auto_progress.py >> "$LOG" 2>&1
            rmdir "$LOCKFILE" 2>/dev/null
        else
            echo "[$(date)] Lock held, skipping" >> "$LOG"
        fi
    else
        echo "[$(date)] $RUNNING running, full" >> "$LOG"
    fi
    sleep 120
done
