#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

MODEL="gpt-5.1-codex-mini"
DELAY="0.8"
GROUP_A="71,458,310"
GROUP_B="164,119"
VERSIONS=(v2 v4 v5 v6)
LOG_ROOT="outputs/pilot_codexmini/repair_logs_parallel"
mkdir -p "$LOG_ROOT"

echo "[repair-parallel] start $(date)"
PIDS=()
for V in "${VERSIONS[@]}"; do
  echo "[repair-parallel] launch gpt-$V groupA/groupB"
  uv run python scripts/run_pilot.py \
    --version "gpt-$V" \
    --users "$GROUP_A" \
    --model "$MODEL" \
    --delay "$DELAY" \
    --output-dir outputs/pilot_codexmini/groupA \
    > "$LOG_ROOT/${V}_groupA.log" 2>&1 &
  PIDS+=("$!")

  uv run python scripts/run_pilot.py \
    --version "gpt-$V" \
    --users "$GROUP_B" \
    --model "$MODEL" \
    --delay "$DELAY" \
    --output-dir outputs/pilot_codexmini/groupB \
    > "$LOG_ROOT/${V}_groupB.log" 2>&1 &
  PIDS+=("$!")
done

echo "[repair-parallel] launched ${#PIDS[@]} jobs"
FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "[repair-parallel] job failed pid=$PID"
    FAIL=1
  fi
done

echo "[repair-parallel] end $(date) fail=$FAIL"
exit $FAIL
