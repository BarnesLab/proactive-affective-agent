#!/bin/bash
# Phase 1 batch processing — light sensor modalities
# Motion runs first (sequential) because screen and keyinput may cross-reference
# its output for the device_missing flag.  The remaining four run in parallel.
set -e
cd /Users/zwang/Documents/proactive-affective-agent

echo "=============================================="
echo "Phase 1 — Offline batch processing"
echo "=============================================="

echo ""
echo "Running motion processing..."
python scripts/offline/process_motion.py

echo ""
echo "Running screen/app processing..."
python scripts/offline/process_screen_app.py &
PID_SCREEN=$!

echo "Running keyinput processing..."
python scripts/offline/process_keyinput.py &
PID_KEY=$!

echo "Running music processing..."
python scripts/offline/process_mus.py &
PID_MUS=$!

echo "Running light processing..."
python scripts/offline/process_light.py &
PID_LIGHT=$!

# Wait for all parallel jobs and capture exit codes
FAIL=0
wait $PID_SCREEN || { echo "[ERROR] process_screen_app.py failed"; FAIL=1; }
wait $PID_KEY    || { echo "[ERROR] process_keyinput.py failed"; FAIL=1; }
wait $PID_MUS    || { echo "[ERROR] process_mus.py failed"; FAIL=1; }
wait $PID_LIGHT  || { echo "[ERROR] process_light.py failed"; FAIL=1; }

echo ""
echo "=============================================="
echo "Phase 1 (light sensors) complete."
echo "NOTE: Run process_accel.py and process_gps.py separately (heavy, run overnight)"
echo "=============================================="

exit $FAIL
