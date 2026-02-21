#!/bin/bash
# Launch 25 parallel experiments: 5 versions × 5 users
# Each process handles one version + one user independently.

cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/pilot"
MODEL="sonnet"
DELAY="1.0"
USERS=(71 164 119 458 310)
VERSIONS=(callm v1 v2 v3 v4)

mkdir -p "$OUTPUT_DIR/logs"

N_JOBS=$(( ${#VERSIONS[@]} * ${#USERS[@]} ))
echo "=== Launching $N_JOBS parallel experiments ==="
echo "  Versions: ${VERSIONS[*]}"
echo "  Users: ${USERS[*]}"
echo "  Model: $MODEL"
echo ""

PIDS=()

for v in "${VERSIONS[@]}"; do
    for u in "${USERS[@]}"; do
        LOG="$OUTPUT_DIR/logs/${v}_user${u}.log"
        echo "Starting $v user $u -> $LOG"
        nohup python3 scripts/run_pilot.py \
            --version "$v" \
            --users "$u" \
            --model "$MODEL" \
            --delay "$DELAY" \
            --output-dir "$OUTPUT_DIR" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
done

echo ""
echo "=== All $N_JOBS processes launched ==="
echo "PIDs: ${PIDS[*]}"
echo ""
echo "Monitor with:"
echo "  watch 'for v in callm v1 v2 v3 v4; do for u in 71 164 119 458 310; do c=\$(ls outputs/pilot/traces/\${v}_user\${u}_entry*.json 2>/dev/null | wc -l); echo \"\$v user\$u: \$c entries\"; done; done'"
echo ""
echo "Or check logs:"
echo "  tail -1 outputs/pilot/logs/*.log"
