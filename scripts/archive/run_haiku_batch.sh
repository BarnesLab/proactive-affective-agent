#!/bin/bash
# Launch 7 versions × 2 user groups = 14 parallel processes with Haiku
# Group A: 71,458,310  |  Group B: 164,119
# This keeps concurrent claude CLI calls at ~14
# Output: outputs/pilot_haiku/

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

GROUP_A="71,458,310"
GROUP_B="164,119"
VERSIONS=(callm v1 v2 v3 v4 v5 v6)
MODEL="haiku"
DELAY="0.5"
OUTPUT_DIR="outputs/pilot_haiku"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

echo "=== Haiku Batch Launch (14 parallel processes) ==="
echo "Group A: $GROUP_A"
echo "Group B: $GROUP_B"
echo "Versions: ${VERSIONS[*]}"
echo "Model: $MODEL"
echo ""

COUNT=0
for VERSION in "${VERSIONS[@]}"; do
    # Group A
    nohup python3 scripts/run_pilot.py \
        --version "$VERSION" \
        --users "$GROUP_A" \
        --model "$MODEL" \
        --delay "$DELAY" \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/${VERSION}_groupA.log" 2>&1 &
    COUNT=$((COUNT + 1))

    # Group B
    nohup python3 scripts/run_pilot.py \
        --version "$VERSION" \
        --users "$GROUP_B" \
        --model "$MODEL" \
        --delay "$DELAY" \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_DIR/${VERSION}_groupB.log" 2>&1 &
    COUNT=$((COUNT + 1))

    sleep 2
done

echo "$COUNT processes launched."
echo ""
echo "Monitor: ps aux | grep run_pilot | grep -v grep | wc -l"
