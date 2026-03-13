#!/bin/bash
# Run GPT-5.1-codex-mini pilot for all 10 Claude-matched users.
# Splits into 2 batches of 5 users, each batch runs 7 versions in parallel.
# Checkpoint-based: safe to re-run if interrupted.
set -euo pipefail
cd /Users/zwang/Documents/proactive-affective-agent
source .venv/bin/activate

MODEL="gpt-5.1-codex-mini"
OUT="outputs/pilot_gpt51mini_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/logs" "$OUT/traces" "$OUT/memory"

# Match Claude pilot_v2 users
USERS=(43 71 258 275 338 362 399 403 437 513)
VERSIONS=(gpt-callm gpt-v1 gpt-v2 gpt-v3 gpt-v4 gpt-v5 gpt-v6)

printf "%s\n" "${VERSIONS[@]}" > "$OUT/versions.txt"
printf "%s\n" "${USERS[@]}" > "$OUT/users.txt"

: > "$OUT/launch_plan.txt"
for user in "${USERS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    echo "$user $version" >> "$OUT/launch_plan.txt"
  done
done

echo "[$(date)] Starting GPT-5.1-codex-mini pilot: ${#USERS[@]} users x ${#VERSIONS[@]} versions = $(( ${#USERS[@]} * ${#VERSIONS[@]} )) jobs"
echo "Output dir: $OUT"

export OUT MODEL
cat "$OUT/launch_plan.txt" | xargs -n 2 -P 7 bash -c '
  set -euo pipefail
  user="$1"
  version="$2"
  log="$OUT/logs/${version}_user${user}.log"
  echo "[$(date)] START $version user $user (model=$MODEL)" >> "$log"
  if PYTHONPATH=. python scripts/run_pilot.py \
    --version "$version" \
    --users "$user" \
    --model "$MODEL" \
    --delay 1.0 \
    --output-dir "$OUT" \
    --verbose >> "$log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[$(date)] END $version user $user exit=$code" >> "$log"
  exit $code
' _

echo "[$(date)] All jobs finished."
