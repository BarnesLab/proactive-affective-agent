#!/usr/bin/env bash
# Re-run V2/V4 for users that had rate-limit fallback issues.
# Runs 5 processes at a time per user's request.
# Checkpoints have been cleared; JSONL records for affected users deleted.

set -euo pipefail
cd "$(dirname "$0")/.."

# Unset CLAUDECODE to allow claude --print subprocess
unset CLAUDECODE 2>/dev/null || true

LOG_DIR="outputs/pilot/logs"
mkdir -p "$LOG_DIR"

# Users that need re-run (both V2 and V4): high fallback rate
RERUN_USERS=(43 71 211 258 275 362 363 399 505 513)

echo "=== Starting V2/V4 re-runs (5 concurrent processes max) ==="
echo "Start time: $(date)"

run_batch() {
    local ver=$1
    shift
    local users=("$@")
    local pids=()

    for uid in "${users[@]}"; do
        echo "  Launching ${ver} user ${uid}..."
        PYTHONPATH=. python3 scripts/run_pilot.py \
            --version "$ver" --users "$uid" --model sonnet --delay 3 \
            > "$LOG_DIR/rerun_${ver}_user${uid}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all and report
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            ((failed++))
        fi
    done
    echo "  Batch done (${#users[@]} launched, $failed failed): $(date)"
}

# V2 batches
echo ""
echo "--- V2 Batch 1: users 43,71,211,258,275 ---"
run_batch v2 43 71 211 258 275

echo ""
echo "--- V2 Batch 2: users 362,363,399,505,513,86 ---"
run_batch v2 362 363 399 505 513

# Also re-run V2 user86 (had only 23 records)
echo "  + Launching V2 user 86..."
PYTHONPATH=. python3 scripts/run_pilot.py \
    --version v2 --users 86 --model sonnet --delay 3 \
    > "$LOG_DIR/rerun_v2_user86.log" 2>&1
echo "  V2 user86 done: $(date)"

# V4 batches
echo ""
echo "--- V4 Batch 1: users 43,71,211,258,275 ---"
run_batch v4 43 71 211 258 275

echo ""
echo "--- V4 Batch 2: users 362,363,399,505,513 ---"
run_batch v4 362 363 399 505 513

echo ""
echo "=== ALL BATCHES COMPLETE: $(date) ==="

# Quick validation
echo ""
echo "=== Validation ==="
python3 -c "
import json
from pathlib import Path

users = [43, 71, 86, 211, 258, 275, 362, 363, 399, 505, 513]
for ver in ['v2', 'v4']:
    print(f'\n{ver.upper()}:')
    for uid in users:
        f = Path(f'outputs/pilot/{ver}_user{uid}_records.jsonl')
        if not f.exists():
            if ver == 'v4' and uid == 86:
                continue  # v4_user86 not in re-run list
            print(f'  user{uid}: MISSING!')
            continue
        entries = [json.loads(l) for l in f.read_text().strip().split('\n') if l.strip()]
        total = len(entries)
        fallback = sum(1 for e in entries if e.get('confidence', 1) <= 0.1)
        print(f'  user{uid}: {total} entries, {fallback} fallback ({fallback*100//max(total,1)}%)')
"
