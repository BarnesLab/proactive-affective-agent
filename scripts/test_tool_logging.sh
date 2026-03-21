#!/bin/bash
# Independent test: verify MCP tool call logging works
# Run this OUTSIDE of Claude Code (e.g., from a raw terminal)
# to avoid nested session env vars blocking MCP.

set -e
cd "$(dirname "$0")/.."
PROJ=$(pwd)

# Clean env
unset CLAUDE_CODE CLAUDECODE CLAUDE_CODE_SESSION_ID

# Clear pycache to pick up latest code
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Clear old tool logs
rm -f outputs/pilot_v2/tool_logs/*.jsonl 2>/dev/null

# Use a test user (user 24) with a separate output dir
TEST_DIR="/tmp/paa_tool_log_test"
rm -rf "$TEST_DIR"

echo "=== Starting tool logging test ==="
echo "Output: $TEST_DIR"
echo "Tool logs: $PROJ/outputs/pilot_v2/tool_logs/"

# Run 3 entries only (enough to verify logging)
PYTHONPATH="$PROJ" .venv/bin/python scripts/run_pilot.py \
  --version v6 --users 24 --model sonnet \
  --output-dir "$TEST_DIR" --verbose 2>&1 &
PID=$!

echo "PID: $PID"
echo "Waiting 3 minutes for entries to complete..."
sleep 180

echo ""
echo "=== RESULTS ==="

# Check tool logs
echo "Tool log files:"
ls -la outputs/pilot_v2/tool_logs/*.jsonl 2>/dev/null || echo "  NONE - FAILED"

echo ""
echo "Tool calls logged:"
cat outputs/pilot_v2/tool_logs/*.jsonl 2>/dev/null | python3 -c "
import sys, json
count = 0
for line in sys.stdin:
    d = json.loads(line)
    count += 1
    print(f'  {d[\"tool\"]}: user{d[\"study_id\"]} {d[\"ema_date\"]} result_len={d[\"result_length\"]}')
print(f'Total: {count} tool calls')
" 2>/dev/null || echo "  (empty)"

echo ""
echo "Trace check:"
python3 -c "
import json, glob
for f in sorted(glob.glob('$TEST_DIR/traces/v6_user24_entry*.json'))[:3]:
    d = json.load(open(f))
    tc = d.get('_tool_calls', [])
    print(f'  {f.split(\"/\")[-1]}: n_tool={d.get(\"_n_tool_calls\")}, detail={len(tc)}')
    for t in tc:
        print(f'    -> {t.get(\"name\")}: result_len={t.get(\"result_length\")}')
" 2>/dev/null || echo "  No traces yet"

kill $PID 2>/dev/null
echo ""
echo "Done. If tool calls show above, logging works!"
