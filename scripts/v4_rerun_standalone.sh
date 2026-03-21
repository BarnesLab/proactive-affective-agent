#!/bin/bash
# V4 Rerun with fixed MCP - run from a CLEAN terminal (NOT Claude Code)
# Usage: bash scripts/v4_rerun_standalone.sh
set -ex

cd "$(dirname "$0")/.."
echo "Working dir: $(pwd)"
PROJ=$(pwd)
OUTDIR="$PROJ/outputs/pilot_v3_v4_rerun_mar_21"
VENV="$PROJ/.venv/bin/python"
VERSION="v4"

# Strip ALL Claude Code env vars
unset CLAUDE_CODE CLAUDECODE CLAUDE_CODE_SESSION_ID CLAUDE_CODE_ENTRYPOINT CLAUDE_AUTOCOMPACT_PCT_OVERRIDE 2>/dev/null || true

# Clear pycache
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Ensure dirs exist
mkdir -p "$OUTDIR"/{checkpoints,logs,traces,tool_logs,memory}

echo "=== V4 Rerun (standalone) ==="

# Quick MCP test: 1 entry
echo "--- MCP Connection Test ---"
rm -rf /tmp/paa_mcp_verify
PYTHONPATH="$PROJ" "$VENV" scripts/run_pilot.py \
  --version v4 --users 24 --model sonnet \
  --output-dir /tmp/paa_mcp_verify --verbose 2>&1 &
TEST_PID=$!
echo "Test PID: $TEST_PID, waiting 5min..."
sleep 300
kill $TEST_PID 2>/dev/null || true

# Check MCP
python3 -c "
import json, glob, sys
traces = sorted(glob.glob('/tmp/paa_mcp_verify/traces/v4_user24_entry*.json'))
if not traces:
    print('NO TRACES - test failed')
    sys.exit(1)
with open(traces[0]) as f:
    d = json.load(f)
resp = d.get('_full_response', '')
mcp_fail = 'not available' in resp[:300] or \"aren't\" in resp[:300]
conv = d.get('_conversation_length', 0)
if mcp_fail:
    print(f'MCP FAILED (conv={conv})')
    print(f'Response: {resp[:300]}')
    sys.exit(1)
else:
    print(f'MCP OK! conv={conv}. Proceeding.')
"
MCP_OK=$?
if [ $MCP_OK -ne 0 ]; then
    echo "MCP test failed. NOT starting rerun."
    exit 1
fi

echo ""
echo "--- Starting Full V4 Rerun ---"
PYTHONPATH="$PROJ" "$VENV" scripts/v4_rerun_watcher.py
