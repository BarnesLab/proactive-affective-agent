#!/bin/bash
# Sonnet watcher: maintains exactly TARGET sonnet processes
# Runs as a daemon, checks every 30 seconds
# Immune to SIGHUP (survives gateway restarts)

trap '' HUP  # Ignore SIGHUP so gateway restart won't kill us

TARGET=15
PROJ_DIR="$HOME/Documents/proactive-affective-agent"
OUTDIR="$PROJ_DIR/outputs/pilot_v2"
VENV="$PROJ_DIR/.venv/bin/python"
LOG="$OUTDIR/logs/watcher.log"

mkdir -p "$OUTDIR/logs"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG"; }

is_peak_hours() {
    # Peak: weekdays 8AM-2PM ET (Claude promo: 2026-03-13 to 2026-03-27)
    local dow=$(date +%u)  # 1=Mon, 7=Sun
    local hour=$(date +%H)
    # Weekday (1-5) AND 8-13 (8AM to 1:59PM)
    if [ "$dow" -le 5 ] && [ "$hour" -ge 8 ] && [ "$hour" -lt 14 ]; then
        return 0  # true = peak
    fi
    return 1  # false = off-peak
}

log "Watcher started (target=$TARGET)"

while true; do
    # Check if Claude is rate-limited. Flag file contains expiry ISO date.
    RATE_LIMIT_FLAG="$OUTDIR/.rate_limited"
    if [ -f "$RATE_LIMIT_FLAG" ]; then
        expiry=$(head -1 "$RATE_LIMIT_FLAG" | tr -d '[:space:]')
        expiry_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S%z" "$expiry" "+%s" 2>/dev/null || echo 0)
        now_epoch=$(date +%s)
        if [ "$now_epoch" -lt "$expiry_epoch" ]; then
            sleep 1800  # Sleep 30 min, check again
            continue
        fi
        log "Rate limit expired, removing flag"
        rm -f "$RATE_LIMIT_FLAG"
    fi

    # Skip during Claude peak hours (weekdays 8AM-2PM ET) to save tokens
    if is_peak_hours; then
        current=$(ps aux | grep "run_pilot.*sonnet" | grep -v grep | grep python | wc -l | tr -d ' ')
        if [ "$current" -gt 0 ]; then
            log "Peak hours detected ($(date '+%H:%M')), stopping $current sonnet processes"
            pkill -f "run_pilot.*sonnet" 2>/dev/null
        fi
        sleep 300  # Check every 5 min during peak
        continue
    fi

    current=$(ps aux | grep "run_pilot.*sonnet" | grep -v grep | grep python | wc -l | tr -d ' ')
    
    if [ "$current" -gt "$TARGET" ]; then
        log "Sonnet processes: $current/$TARGET, converging (no new spawns, waiting for $((current - TARGET)) to finish)"
        # Don't kill, just don't spawn — let running tasks finish naturally
    elif [ "$current" -lt "$TARGET" ]; then
        needed=$((TARGET - current))
        log "Sonnet processes: $current/$TARGET, launching $needed"
        
        # Call auto_progress to fill slots
        cd "$PROJ_DIR"
        source .venv/bin/activate 2>/dev/null
        
        # Get next tasks from python
        $VENV -c "
import os, glob, subprocess, sys
sys.path.insert(0, '.')
from src.data.loader import DataLoader

versions = ['callm','v1','v2','v3','v4','v5','v6']
target_users = [399, 258, 43, 71, 211, 505, 513, 363, 275, 437, 362, 86, 24, 164, 169, 119, 99, 61, 458, 403, 503, 41, 310, 338, 25, 40, 89, 232, 242, 299, 455, 187, 499, 320, 257, 361, 95, 103, 75, 83, 464, 335, 392, 351, 60, 82, 260, 189, 140, 98]

# What's done
done = set()
for f in glob.glob('outputs/pilot_v2/*_records.jsonl') + glob.glob('outputs/pilot/*_records.jsonl'):
    done.add(os.path.basename(f).replace('_records.jsonl','').lower())

# What's running
running = set()
r = subprocess.run(['ps','aux'], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if 'run_pilot' in line and 'sonnet' in line and 'grep' not in line and 'python' in line.lower():
        import re
        mv = re.search(r'--version (\S+)', line)
        mu = re.search(r'--users (\d+)', line)
        if mv and mu:
            running.add(f'{mv.group(1)}_user{mu.group(1)}')

# User-first: find users closest to completion
user_completion = {}
for u in target_users:
    completed = sum(1 for v in versions if f'{v}_user{u}' in done)
    in_progress = sum(1 for v in versions if f'{v}_user{u}' in running)
    remaining = 7 - completed - in_progress
    if remaining > 0:
        user_completion[u] = (completed, remaining)

# Sort: most completed first (closest to done)
sorted_users = sorted(user_completion.keys(), key=lambda u: -user_completion[u][0])

# Priority order: v6 first, then v5, v4, ... (finish higher versions first)
priority_versions = list(reversed(versions))

needed = $needed
launched = 0
for v in priority_versions:
    if launched >= needed:
        break
    for u in sorted_users:
        if launched >= needed:
            break
        key = f'{v}_user{u}'
        if key not in done and key not in running:
            outdir = '$OUTDIR'
            logf = f'{outdir}/logs/{v}_user{u}.log'
            cmd = f'nohup $VENV scripts/run_pilot.py --version {v} --users {u} --model sonnet --delay 1.0 --output-dir {outdir} --verbose > {logf} 2>&1 &'
            os.system(cmd)
            print(f'Launched: {v} user{u}')
            launched += 1
" 2>> "$LOG"
    fi
    
    # Use project venv python to avoid homebrew python3 hanging in launchd
    $VENV -c "import json,time;f='$HOME/.openclaw/pulse.json';d=json.load(open(f)) if __import__('os').path.exists(f) else {};d['sonnet-watcher']=int(time.time());json.dump(d,open(f,'w'),indent=2)" 2>/dev/null
    sleep 30
done
