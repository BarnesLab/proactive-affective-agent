#!/bin/bash
set -euo pipefail
cd /Users/zwang/projects/proactive-affective-agent
source .venv/bin/activate
OUT="outputs/pilot_gpt54_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/logs"
VERSIONS=(gpt-callm gpt-v1 gpt-v2 gpt-v3 gpt-v4 gpt-v5 gpt-v6)
USERS=(275 513 362 437 24)
printf "%s\n" "${VERSIONS[@]}" > "$OUT/versions.txt"
printf "%s\n" "${USERS[@]}" > "$OUT/users.txt"
: > "$OUT/launch_plan.txt"
for user in "${USERS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    echo "$user $version" >> "$OUT/launch_plan.txt"
  done
done
export OUT
cat "$OUT/launch_plan.txt" | xargs -n 2 -P 10 bash -c '
  set -euo pipefail
  user="$1"
  version="$2"
  log="$OUT/logs/${version}_user${user}.log"
  echo "[$(date)] START $version user $user" >> "$log"
  if PYTHONPATH=. python scripts/run_pilot.py --version "$version" --users "$user" --model gpt-5.4 --delay 1.0 --output-dir "$OUT" --verbose >> "$log" 2>&1; then
    code=0
  else
    code=$?
  fi
  echo "[$(date)] END $version user $user exit=$code" >> "$log"
  exit $code
' _
